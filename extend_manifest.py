"""
Extend Manifest
=================
Manually add newly uploaded Kaggle dataset shards to the frozen manifest
without invalidating past runs. Creates a new version while keeping the
previous version accessible at manifest/split_manifest_vN.json.

Must be run deliberately — never called automatically.

Usage (on Kaggle, after attaching new datasets):
    !python extend_manifest.py

What it does:
  1. Downloads current manifest from HF
  2. Discovers mounted shards NOT already in shards_used
  3. Builds stratified train/val/test split for new images only
  4. Appends to images dict, bumps version, updates split_sizes
  5. Archives old manifest as manifest/split_manifest_vN.json
  6. Uploads new manifest as manifest/split_manifest.json
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_config import (
    HF_MANIFEST_REPO, MANIFEST_PATH_IN_REPO, KAGGLE_INPUT_ROOT,
    SEED, LABEL_MAP, CLASS_NAMES, IMG_EXTENSIONS, KAGGLE_DATASETS,
)
from split_manifest_manager import (
    stable_image_id, compute_manifest_sha256,
)


def extend_manifest(
    hf_token: str = None,
    input_root: str = None,
    local_csv: str = None,
    new_val_ratio: float = 0.15,
    new_test_ratio: float = 0.15,
):
    """
    Extend the current manifest with newly discovered shards.

    Args:
        hf_token: HF token. Auto-discovered if None.
        input_root: Kaggle input root. Default /kaggle/input.
        local_csv: Optional path to a CSV with new images.
        new_val_ratio: Val ratio for new images (default 0.15).
        new_test_ratio: Test ratio for new images (default 0.15).
    """
    print("=" * 60)
    print("Extending Split Manifest")
    print("=" * 60)

    # Auto-discover token
    if hf_token is None:
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            env_path = Path(__file__).parent / ".env"
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        if line.startswith("hf="):
                            hf_token = line.strip().split("=", 1)[1]
                            break
        if not hf_token:
            raise ValueError("HF_TOKEN not found")

    from huggingface_hub import hf_hub_download, HfApi

    # 1. Download current manifest
    print("\n[1] Downloading current manifest from HF...")
    path = hf_hub_download(
        repo_id=HF_MANIFEST_REPO,
        filename=MANIFEST_PATH_IN_REPO,
        repo_type="model",
        token=hf_token,
    )
    with open(path) as f:
        current_manifest = json.load(f)
    current_sha256 = compute_manifest_sha256(current_manifest)

    print(f"  Version: {current_manifest['version']}")
    print(f"  Images: {current_manifest.get('total_images', '?')}")
    print(f"  Shards used: {current_manifest['shards_used']}")
    print(f"  Split sizes: {current_manifest['split_sizes']}")

    existing_ids = set(current_manifest["images"].keys())

    # 2. Discover new images (not in current manifest)
    print("\n[2] Discovering new images...")
    from sklearn.model_selection import train_test_split

    new_rows = []
    if local_csv:
        import csv
        with open(local_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fn = row.get("filename", "")
                lbl = row.get("label", "real")
                shard = row.get("source", "unknown")
                img_id = stable_image_id(fn, shard, lbl)
                if img_id not in existing_ids:
                    new_rows.append({
                        "image_id": img_id,
                        "shard": shard,
                        "label": lbl,
                        "label_int": LABEL_MAP.get(lbl, 0),
                    })
    else:
        root = Path(input_root) if input_root else KAGGLE_INPUT_ROOT
        for slug, (label, subdir) in KAGGLE_DATASETS.items():
            mount_dir = root / slug
            if not mount_dir.exists():
                # Fuzzy match
                if root.exists():
                    actual = [d.name for d in root.iterdir() if d.is_dir()]
                    matches = [m for m in actual if slug in m or m in slug]
                    if matches:
                        mount_dir = root / min(matches, key=len)
                    else:
                        continue
                else:
                    continue
            image_root = mount_dir / subdir
            if not image_root.exists():
                image_root = mount_dir

            count = 0
            for img_path in image_root.rglob("*"):
                if img_path.suffix.lower() in IMG_EXTENSIONS and img_path.is_file():
                    if img_path.stat().st_size == 0:
                        continue
                    img_id = stable_image_id(img_path.name, slug, label)
                    if img_id not in existing_ids:
                        new_rows.append({
                            "image_id": img_id,
                            "shard": slug,
                            "label": label,
                            "label_int": LABEL_MAP[label],
                        })
                        count += 1
            if count > 0:
                print(f"  {slug}: {count} NEW images ({label})")

    if not new_rows:
        print("\n  No new images found. Manifest unchanged.")
        return current_manifest

    print(f"\n  Total new images: {len(new_rows)}")

    # 3. Stratified split for new images only
    print("\n[3] Building stratified split for new images...")
    labels = [r["label_int"] for r in new_rows]
    holdout = new_val_ratio + new_test_ratio

    if len(new_rows) >= 3:
        train_val_idx, test_idx = train_test_split(
            list(range(len(new_rows))), test_size=holdout,
            stratify=labels, random_state=SEED,
        )
        rv_labels = [labels[i] for i in train_val_idx]
        val_ratio_adj = new_val_ratio / holdout
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_ratio_adj,
            stratify=rv_labels, random_state=SEED,
        )
    else:
        train_idx = list(range(len(new_rows)))
        val_idx, test_idx = [], []

    split_map = {}
    for i in train_idx:
        split_map[i] = "train"
    for i in val_idx:
        split_map[i] = "val"
    for i in test_idx:
        split_map[i] = "test"

    for i, row in enumerate(new_rows):
        row["split"] = split_map.get(i, "train")

    # 4. Append to manifest
    print("\n[4] Appending to manifest...")
    old_version = current_manifest["version"]
    new_version = old_version + 1

    for row in new_rows:
        current_manifest["images"][row["image_id"]] = {
            "shard": row["shard"],
            "label": row["label"],
            "label_int": row["label_int"],
            "split": row["split"],
        }

    # Update shards_used
    new_shards = set(row["shard"] for row in new_rows)
    for s in new_shards:
        if s not in current_manifest["shards_used"]:
            current_manifest["shards_used"].append(s)

    # Update split_sizes
    for row in new_rows:
        s = row["split"]
        current_manifest["split_sizes"][s] = current_manifest["split_sizes"].get(s, 0) + 1

    # BUG 2 fix: recompute total_images
    current_manifest["total_images"] = len(current_manifest["images"])

    current_manifest["version"] = new_version

    # Recompute overall class distribution
    all_labels = [v["label"] for v in current_manifest["images"].values()]
    current_manifest["class_distribution"] = {}
    for lbl_name, lbl_int in LABEL_MAP.items():
        count = sum(1 for l in all_labels if l == lbl_name)
        current_manifest["class_distribution"][lbl_name] = round(count / len(all_labels), 4)

    # Recompute manifest hash
    new_sha256 = compute_manifest_sha256(current_manifest)
    current_manifest["manifest_sha256"] = new_sha256

    # Print updated stats
    print(f"\n  New version: {new_version}")
    print(f"  Total images: {current_manifest['total_images']}")
    print(f"  Split sizes: {current_manifest['split_sizes']}")
    print(f"  Class distribution: {current_manifest['class_distribution']}")
    print(f"  Shards used: {current_manifest['shards_used']}")

    # 5. Archive old manifest, upload new
    print(f"\n[5] Archiving old manifest (v{old_version}) and uploading new (v{new_version})...")
    api = HfApi(token=hf_token)
    api.create_repo(HF_MANIFEST_REPO, repo_type="model", exist_ok=True)

    # Save old manifest locally, upload as archive
    old_path = Path(f"/kaggle/working/split_manifest_v{old_version}.json")
    with open(old_path, "w") as f:
        json.dump(current_manifest, f, indent=2, default=str)

    api.upload_file(
        path_or_fileobj=str(old_path),
        path_in_repo=f"manifest/split_manifest_v{old_version}.json",
        repo_id=HF_MANIFEST_REPO,
        repo_type="model",
    )
    print(f"  Archived: manifest/split_manifest_v{old_version}.json")

    # Upload new manifest to main path
    new_path = Path("/kaggle/working/split_manifest_new.json")
    with open(new_path, "w") as f:
        json.dump(current_manifest, f, indent=2, default=str)

    api.upload_file(
        path_or_fileobj=str(new_path),
        path_in_repo=MANIFEST_PATH_IN_REPO,
        repo_id=HF_MANIFEST_REPO,
        repo_type="model",
    )
    print(f"  Uploaded: {MANIFEST_PATH_IN_REPO} (v{new_version})")

    print(f"\n{'='*60}")
    print(f"Done! Manifest extended from v{old_version} to v{new_version}")
    print(f"New SHA-256: {new_sha256[:16]}...")
    print(f"{'='*60}")

    return current_manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-input", default="/kaggle/input")
    parser.add_argument("--local-csv", default=None)
    parser.add_argument("--new-val-ratio", type=float, default=0.15)
    parser.add_argument("--new-test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    extend_manifest(
        input_root=args.kaggle_input,
        local_csv=args.local_csv,
        new_val_ratio=args.new_val_ratio,
        new_test_ratio=args.new_test_ratio,
    )
