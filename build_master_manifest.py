"""
Build Master Manifest for Kaggle Multi-Session Training
=========================================================
Scans the local dataset, generates deterministic train/val/test splits,
and uploads to HuggingFace as the single source of truth.

Every training session (any model, any Kaggle account, any session)
downloads this manifest and uses the exact same split indices.

Usage:
    python build_master_manifest.py --upload
    python build_master_manifest.py --dry-run
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_IMAGES_DIR = PROJECT_ROOT / "dataset" / "images"
LOCAL_METADATA_DIR = PROJECT_ROOT / "dataset" / "metadata"
HF_REPO_ID = "studyhub991/mfft-master-manifest"  # will create if not exists
SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Mapping: Kaggle dataset slug -> (local_dir_name, default_label)
KAGGLE_TO_LOCAL = {
    "mihmahmud/stable-diffusion":   ("Stable Diffusion", "ai_generated"),
    "benjaminkz/places365":         ("Places365", "real"),
    "mihmahmud/open-images-v7-dataset": ("Open-Images-V7-Dataset", "real"),
    "mihmahmud/ntire2026":          ("NTIRE2026", "mixed"),
    "mihmahmud/midjourney":         ("Midjourney", "ai_generated"),
    "studyhub991/mfft-real":        ("real", "real"),
    "studyhub991/genimage-ai":      ("genimage_ai", "ai_generated"),
    "greatgamedota/faceforensics":  ("FaceForensics", "deepfake"),
    "itamargr/dfdc-faces-of-the-train-sample": ("DFDC", "deepfake"),
    "mihmahmud/dall-e3":            ("DALL-E3", "ai_generated"),
    "pranabr0y/celebdf-v2image-dataset": ("CelebDF_V2", "deepfake"),
    "awsaf49/artifact-dataset":     (None, "real"),  # AFHQ, unused
}

# Kaggle slug -> expected mount path on Kaggle
KAGGLE_MOUNT_SLUGS = {
    "mihmahmud/stable-diffusion":   "stable-diffusion",
    "benjaminkz/places365":         "places365",
    "mihmahmud/open-images-v7-dataset": "open-images-v7-dataset",
    "mihmahmud/ntire2026":          "ntire2026",
    "mihmahmud/midjourney":         "midjourney",
    "studyhub991/mfft-real":        "mfft-real",
    "studyhub991/genimage-ai":      "genimage-ai",
    "greatgamedota/faceforensics":  "faceforensics",
    "itamargr/dfdc-faces-of-the-train-sample": "dfdc-faces-of-the-train-sample",
    "mihmahmud/dall-e3":            "dall-e3",
    "pranabr0y/celebdf-v2image-dataset": "celebdf-v2image-dataset",
}

LABEL_MAP = {"real": 0, "ai_generated": 1, "deepfake": 2}
CLASS_NAMES = ["real", "ai_generated", "deepfake"]


def file_hash(path: Path, chunk_size: int = 8192) -> str:
    """Fast partial hash for image deduplication."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        data = f.read(chunk_size)
        h.update(data)
    return h.hexdigest()[:12]


def scan_local_dataset() -> pd.DataFrame:
    """Scan local dataset/images/ and build a DataFrame with all images."""
    rows = []

    for subdir_name, (local_dir_name, default_label) in KAGGLE_TO_LOCAL.items():
        if local_dir_name is None:
            continue

        local_dir = LOCAL_IMAGES_DIR / local_dir_name
        if not local_dir.exists():
            print(f"  Skipping {local_dir_name}: not found locally")
            continue

        # For NTIRE2026, check for metadata to determine labels
        if default_label == "mixed":
            # Try to load NTIRE metadata if available
            ntire_meta = LOCAL_METADATA_DIR / "ntire2026_metadata.csv"
            if ntire_meta.exists():
                ntire_df = pd.read_csv(ntire_meta)
                # Use metadata labels
                for _, row in ntire_df.iterrows():
                    img_path = local_dir / row.get("filename", "")
                    if img_path.exists() and img_path.stat().st_size > 0:
                        label_str = str(row.get("label", "real")).lower()
                        label_int = LABEL_MAP.get(label_str, 0)
                        rows.append({
                            "image_id": file_hash(img_path),
                            "relative_path": f"{local_dir_name}/{img_path.relative_to(local_dir)}",
                            "label": label_int,
                            "label_name": CLASS_NAMES[label_int],
                            "source": subdir_name,
                            "kaggle_mount_slug": KAGGLE_MOUNT_SLUGS.get(subdir_name, ""),
                        })
                continue
            else:
                # Default: treat NTIRE as real (most are real images)
                default_label = "real"

        # Scan all images in the directory
        img_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        count = 0
        for img_path in local_dir.rglob("*"):
            if img_path.suffix.lower() in img_extensions and img_path.stat().st_size > 0:
                label_int = LABEL_MAP[default_label]
                rows.append({
                    "image_id": file_hash(img_path),
                    "relative_path": f"{local_dir_name}/{img_path.relative_to(local_dir)}",
                    "label": label_int,
                    "label_name": CLASS_NAMES[label_int],
                    "source": subdir_name,
                    "kaggle_mount_slug": KAGGLE_MOUNT_SLUGS.get(subdir_name, ""),
                })
                count += 1

        print(f"  {local_dir_name}: {count} images ({default_label})")

    return pd.DataFrame(rows)


def create_deterministic_split(
    df: pd.DataFrame,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
    seed: int = SEED,
) -> dict:
    """
    Create deterministic train/val/test split using stratified sampling.
    Same seed + same data = same split every time, on every machine.
    """
    from sklearn.model_selection import train_test_split

    indices = list(range(len(df)))
    labels = df["label"].tolist()

    holdout = val_split + test_split
    train_idx, rest_idx = train_test_split(
        indices, test_size=holdout, stratify=labels, random_state=seed,
    )
    rest_labels = [labels[i] for i in rest_idx]
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=test_split / holdout, stratify=rest_labels, random_state=seed,
    )

    # Verify balance
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    test_labels = [labels[i] for i in test_idx]

    split_info = {
        "seed": seed,
        "val_split": val_split,
        "test_split": test_split,
        "total_samples": len(df),
        "train": {
            "indices": train_idx,
            "count": len(train_idx),
            "class_distribution": dict(Counter(train_labels)),
        },
        "val": {
            "indices": val_idx,
            "count": len(val_idx),
            "class_distribution": dict(Counter(val_labels)),
        },
        "test": {
            "indices": test_idx,
            "count": len(test_idx),
            "class_distribution": dict(Counter(test_labels)),
        },
    }

    return split_info


def build_manifest(dry_run: bool = False) -> dict:
    """Build the complete master manifest."""
    print("Scanning local dataset...")
    df = scan_local_dataset()

    if len(df) == 0:
        raise RuntimeError("No images found in local dataset")

    print(f"\nTotal images: {len(df)}")
    print(f"Label distribution: {dict(df['label_name'].value_counts())}")

    print("\nCreating deterministic split...")
    split_info = create_deterministic_split(df)

    for split_name in ["train", "val", "test"]:
        s = split_info[split_name]
        dist = {CLASS_NAMES[k]: v for k, v in s["class_distribution"].items()}
        print(f"  {split_name}: {s['count']} samples | {dist}")

    # Build manifest
    manifest = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "seed": SEED,
        "class_names": CLASS_NAMES,
        "label_map": LABEL_MAP,
        "kaggle_datasets": {
            kaggle_slug: {
                "mount_slug": KAGGLE_MOUNT_SLUGS.get(kaggle_slug, ""),
                "local_dir": local_dir,
                "default_label": label,
            }
            for kaggle_slug, (local_dir, label) in KAGGLE_TO_LOCAL.items()
            if local_dir is not None
        },
        "split": {
            "seed": split_info["seed"],
            "val_split": split_info["val_split"],
            "test_split": split_info["test_split"],
            "train_indices": split_info["train"]["indices"],
            "val_indices": split_info["val"]["indices"],
            "test_indices": split_info["test"]["indices"],
            "train_count": split_info["train"]["count"],
            "val_count": split_info["val"]["count"],
            "test_count": split_info["test"]["count"],
            "train_class_dist": split_info["train"]["class_distribution"],
            "val_class_dist": split_info["val"]["class_distribution"],
            "test_class_dist": split_info["test"]["class_distribution"],
        },
        "samples": df.to_dict(orient="records"),
    }

    # Compute manifest hash for verification
    manifest_str = json.dumps(manifest, sort_keys=True, default=str)
    manifest["manifest_hash"] = hashlib.sha256(manifest_str.encode()).hexdigest()[:16]

    # Save locally
    manifest_path = LOCAL_METADATA_DIR / "master_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest saved locally: {manifest_path}")

    if not dry_run:
        upload_to_huggingface(manifest)

    return manifest


def upload_to_huggingface(manifest: dict):
    """Upload manifest to HuggingFace Hub."""
    print("\nUploading to HuggingFace Hub...")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub -q")
        from huggingface_hub import HfApi

    # Get HF token from .env or environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("hf="):
                        hf_token = line.strip().split("=", 1)[1]
                        break

    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env or environment")

    api = HfApi(token=hf_token)

    # Create repo if it doesn't exist
    try:
        api.create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
        print(f"  Repo: {HF_REPO_ID}")
    except Exception as e:
        print(f"  Repo creation: {e}")

    # Upload manifest
    manifest_path = LOCAL_METADATA_DIR / "master_manifest.json"
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="master_manifest.json",
        repo_id=HF_REPO_ID,
        repo_type="model",
    )
    print(f"  Uploaded: master_manifest.json to {HF_REPO_ID}")

    # Also upload a lightweight version (without full samples list) for quick checks
    light_manifest = {k: v for k, v in manifest.items() if k != "samples"}
    light_manifest["sample_count"] = len(manifest["samples"])
    light_path = LOCAL_METADATA_DIR / "manifest_summary.json"
    with open(light_path, "w") as f:
        json.dump(light_manifest, f, indent=2, default=str)

    api.upload_file(
        path_or_fileobj=str(light_path),
        path_in_repo="manifest_summary.json",
        repo_id=HF_REPO_ID,
        repo_type="model",
    )
    print(f"  Uploaded: manifest_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Build manifest but don't upload")
    parser.add_argument("--hf-repo", type=str, default=HF_REPO_ID)
    args = parser.parse_args()

    HF_REPO_ID = args.hf_repo
    build_manifest(dry_run=args.dry_run)
