"""
Build Master Manifest from Kaggle Mounted Datasets
====================================================
Scans all mounted Kaggle datasets, discovers image paths, assigns labels,
creates deterministic train/val/test splits, and uploads to HuggingFace.

Run this on FIRST Kaggle session with ALL 11 datasets attached.
Subsequent sessions download the manifest from HF instead of re-scanning.

Usage (in Kaggle notebook):
    # Cell 1: Copy this file or !cp /kaggle/input/.../build_master_manifest.py .
    # Cell 2: !python build_master_manifest.py

Or as a standalone script:
    python build_master_manifest.py --kaggle-input /kaggle/input
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np

# ============================================================
# DATASET CONFIGURATION
# ============================================================
# Maps Kaggle dataset slug -> (default_label, image_subdir)
# label: "real" | "ai_generated" | "deepfake"
# image_subdir: subdirectory within the mount that contains images
KAGGLE_DATASETS = {
    "stable-diffusion": {
        "label": "ai_generated",
        "image_subdir": "Stable Diffusion/images",
        "description": "Stable Diffusion generated images",
    },
    "places365": {
        "label": "real",
        "image_subdir": "train",
        "description": "Places365 real scene images",
    },
    "open-images-v7-dataset": {
        "label": "real",
        "image_subdir": "Open-Images-V7-Dataset/open-images-v7/train/images",
        "description": "Open Images V7 real images",
    },
    "ntire2026": {
        "label": "real",
        "image_subdir": "NTIRE2026",
        "description": "NTIRE 2026 competition images (mostly real)",
    },
    "midjourney": {
        "label": "ai_generated",
        "image_subdir": "Midjourney/Datasetfordream",
        "description": "Midjourney generated images",
    },
    "mfft-real": {
        "label": "real",
        "image_subdir": "Mfft_real",
        "description": "Real photographs (Pexels, augmented)",
    },
    "genimage-ai": {
        "label": "ai_generated",
        "image_subdir": "genimage_ai",
        "description": "GenImage AI (BigGAN, Glide, etc.)",
    },
    "faceforensics": {
        "label": "deepfake",
        "image_subdir": "cropped_images",
        "description": "FaceForensics++ deepfake faces",
    },
    "dfdc-faces-of-the-train-sample": {
        "label": "deepfake",
        "image_subdir": "train/fake",
        "description": "DFDC deepfake face samples",
    },
    "dall-e3": {
        "label": "ai_generated",
        "image_subdir": "DALL-E3",
        "description": "DALL-E 3 generated images",
    },
    "celebdf-v2image-dataset": {
        "label": "deepfake",
        "image_subdir": "Celeb_V2",
        "description": "Celeb-DF V2 deepfake faces",
    },
}

LABEL_MAP = {"real": 0, "ai_generated": 1, "deepfake": 2}
CLASS_NAMES = ["real", "ai_generated", "deepfake"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
HF_MANIFEST_REPO = "studyhub991/mfft-master-manifest"


def file_hash(path: Path, chunk_size: int = 8192) -> str:
    """Fast partial hash for unique image IDs."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        data = f.read(chunk_size)
        h.update(data)
    return h.hexdigest()[:12]


def discover_mounted_datasets(input_root: Path) -> dict:
    """Find all mounted datasets that match our expected slugs."""
    mounted = {}
    if not input_root.exists():
        print(f"ERROR: {input_root} does not exist")
        return mounted

    for slug, config in KAGGLE_DATASETS.items():
        mount_dir = input_root / slug
        if mount_dir.exists():
            mounted[slug] = {"path": mount_dir, **config}
            print(f"  Found: {slug}")
        else:
            print(f"  MISSING: {slug}")

    return mounted


def scan_dataset(slug: str, info: dict) -> list:
    """Scan a single mounted dataset and return list of image records."""
    mount_dir = info["path"]
    image_subdir = info["image_subdir"]
    default_label = info["label"]
    label_int = LABEL_MAP[default_label]

    # Find the actual image root
    image_root = mount_dir / image_subdir
    if not image_root.exists():
        # Try fallback: scan entire mount
        image_root = mount_dir

    rows = []
    for img_path in image_root.rglob("*"):
        if img_path.suffix.lower() in IMG_EXTENSIONS and img_path.is_file():
            if img_path.stat().st_size == 0:
                continue

            # Relative path from the mount root (what appears in /kaggle/input/)
            relative = img_path.relative_to(mount_dir)

            rows.append({
                "image_id": file_hash(img_path),
                "relative_path": str(relative),
                "label": label_int,
                "label_name": CLASS_NAMES[label_int],
                "source": slug,
                "kaggle_mount_slug": slug,
            })

    return rows


def create_deterministic_split(
    n_samples: int,
    labels: list,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
    seed: int = SEED,
) -> dict:
    """Create deterministic train/val/test split indices."""
    from sklearn.model_selection import train_test_split

    indices = list(range(n_samples))
    holdout = val_split + test_split

    train_idx, rest_idx = train_test_split(
        indices, test_size=holdout, stratify=labels, random_state=seed,
    )
    rest_labels = [labels[i] for i in rest_idx]
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=test_split / holdout, stratify=rest_labels, random_state=seed,
    )

    return {
        "seed": seed,
        "val_split": val_split,
        "test_split": test_split,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": len(test_idx),
        "train_class_dist": dict(Counter(labels[i] for i in train_idx)),
        "val_class_dist": dict(Counter(labels[i] for i in val_idx)),
        "test_class_dist": dict(Counter(labels[i] for i in test_idx)),
    }


def build_manifest(input_root: str = "/kaggle/input", upload: bool = True) -> dict:
    """Main entry: scan all datasets, build manifest, optionally upload to HF."""
    print("=" * 60)
    print("Building Master Manifest from Kaggle Datasets")
    print("=" * 60)

    input_path = Path(input_root)

    # 1. Discover mounted datasets
    print("\n1. Discovering mounted datasets...")
    mounted = discover_mounted_datasets(input_path)

    if len(mounted) == 0:
        raise RuntimeError(
            f"No datasets found at {input_path}\n"
            "Attach all 11 Kaggle datasets via the Input panel."
        )

    # 2. Scan each dataset
    print(f"\n2. Scanning {len(mounted)} datasets...")
    all_rows = []
    for slug, info in mounted.items():
        rows = scan_dataset(slug, info)
        all_rows.extend(rows)
        print(f"  {slug}: {len(rows)} images ({info['label']})")

    if len(all_rows) == 0:
        raise RuntimeError("No images found in any mounted dataset")

    print(f"\n  Total: {len(all_rows)} images")

    # 3. Create deterministic split
    print("\n3. Creating deterministic split...")
    labels = [r["label"] for r in all_rows]
    split = create_deterministic_split(len(all_rows), labels)

    for split_name in ["train", "val", "test"]:
        count = split[f"{split_name}_count"]
        dist = {CLASS_NAMES[k]: v for k, v in split[f"{split_name}_class_dist"].items()}
        print(f"  {split_name}: {count} | {dist}")

    # 4. Build manifest
    print("\n4. Building manifest...")
    manifest = {
        "version": "2.0",
        "created_at": datetime.now().isoformat(),
        "seed": SEED,
        "class_names": CLASS_NAMES,
        "label_map": LABEL_MAP,
        "kaggle_datasets": {
            slug: {
                "label": info["label"],
                "image_subdir": info["image_subdir"],
                "description": info["description"],
            }
            for slug, info in mounted.items()
        },
        "split": split,
        "samples": all_rows,
    }

    # Compute hash
    manifest_str = json.dumps(manifest, sort_keys=True, default=str)
    manifest["manifest_hash"] = hashlib.sha256(manifest_str.encode()).hexdigest()[:16]

    # 5. Save locally
    manifest_path = Path("/kaggle/working/master_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n  Saved: {manifest_path}")

    # 6. Upload to HuggingFace
    if upload:
        print("\n5. Uploading to HuggingFace Hub...")
        upload_manifest_to_hf(manifest_path, manifest)

    print(f"\n{'='*60}")
    print(f"Manifest ready! Hash: {manifest['manifest_hash']}")
    print(f"{'='*60}")

    return manifest


def upload_manifest_to_hf(local_path: Path, manifest: dict):
    """Upload manifest to HuggingFace Hub."""
    # Get HF token
    hf_token = None

    # Try Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        print("  HF token from Kaggle Secrets")
    except Exception:
        pass

    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            print("  HF token from environment")

    if not hf_token:
        # Try .env
        for env_path in [Path("/kaggle/working/.env"), Path(__file__).parent / ".env"]:
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        if line.startswith("hf="):
                            hf_token = line.strip().split("=", 1)[1]
                            break
            if hf_token:
                break

    if not hf_token:
        raise ValueError("HF_TOKEN not found. Set as Kaggle Secret or in .env")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        os.system("pip install huggingface_hub -q")
        from huggingface_hub import HfApi

    api = HfApi(token=hf_token)

    # Create repo
    try:
        api.create_repo(HF_MANIFEST_REPO, repo_type="model", exist_ok=True)
        print(f"  Repo: {HF_MANIFEST_REPO}")
    except Exception as e:
        print(f"  Repo: {e}")

    # Upload manifest
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo="master_manifest.json",
        repo_id=HF_MANIFEST_REPO,
        repo_type="model",
    )
    print(f"  Uploaded: master_manifest.json")

    # Upload summary (lightweight)
    summary = {k: v for k, v in manifest.items() if k != "samples"}
    summary["sample_count"] = len(manifest["samples"])
    summary_path = Path("/kaggle/working/manifest_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    api.upload_file(
        path_or_fileobj=str(summary_path),
        path_in_repo="manifest_summary.json",
        repo_id=HF_MANIFEST_REPO,
        repo_type="model",
    )
    print(f"  Uploaded: manifest_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-input", default="/kaggle/input")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    build_manifest(input_root=args.kaggle_input, upload=not args.no_upload)
