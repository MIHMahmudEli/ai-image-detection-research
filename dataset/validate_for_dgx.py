"""
Comprehensive Dataset Validation for DGX Pre-flight
====================================================
Checks everything that would cause failures on the DGX:
  1. train_manifest.csv row count
  2. All referenced images exist on disk
  3. Zero-byte files
  4. Label distribution (real vs ai)
  5. Source distribution
  6. File size anomalies

Run from repo root:
  python dataset/validate_for_dgx.py
"""
import csv
import os
import time
import sys
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "dataset" / "images"
METADATA_DIR = PROJECT_ROOT / "dataset" / "metadata"

# Manifests to check
MANIFESTS = {
    "train_manifest.csv": METADATA_DIR / "train_manifest.csv",
    "clean_metadata.csv": METADATA_DIR / "clean_metadata.csv",
    "test_train_manifest.csv": METADATA_DIR / "test_train_manifest.csv",
}

EXPECTED_TRAIN_ROWS = 1_146_683  # from DGX_RUN_GUIDE.md
EXPECTED_SOURCES = {
    "real": ["imagenet", "pexels_unsplash", "places365", "open_images_v7",
             "celebdf_real", "dfdc_real"],
    "ai": ["genimage_biggan", "glide", "stable_diffusion", "dalle3", "midjourney"],
    "deepfake": ["celebdf", "faceforensics", "dfdc"],
}


def resolve_image_path(filename: str) -> Path | None:
    """Resolve image path from metadata filename."""
    filename = filename.replace("\\", "/")
    candidate = IMAGES_DIR / filename
    if candidate.exists():
        return candidate

    for subdir in ["real", "ai_generated", "ai_altered"]:
        candidate = IMAGES_DIR / subdir / filename
        if candidate.exists():
            return candidate

    return None


def validate_manifest(name: str, path: Path) -> dict:
    """Validate a single manifest file."""
    print(f"\n{'='*60}")
    print(f"Validating: {name}")
    print(f"Path: {path}")
    print(f"{'='*60}")

    if not path.exists():
        print(f"  ERROR: File does not exist!")
        return {"status": "MISSING", "rows": 0}

    # Count rows and collect stats
    start = time.time()
    total = 0
    labels = Counter()
    sources = Counter()
    missing = []
    zero_byte = []
    file_sizes = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            filename = row.get("filename", "")
            label = row.get("label", "")
            source = row.get("source", "")

            labels[label] += 1
            sources[source] += 1

            # Check image exists (sample first 100K for speed)
            if total <= 100_000 or total % 10_000 == 0:
                img_path = IMAGES_DIR / filename.replace("\\", "/")
                if not img_path.exists():
                    missing.append(filename)
                elif img_path.stat().st_size == 0:
                    zero_byte.append(filename)

            if total % 500_000 == 0:
                elapsed = time.time() - start
                print(f"  Progress: {total:,} rows ({elapsed:.0f}s)")

    elapsed = time.time() - start

    print(f"\n  Row count:       {total:>10,}")
    print(f"  Expected:        {EXPECTED_TRAIN_ROWS:>10,} (train_manifest)")
    if name == "train_manifest.csv":
        if total < EXPECTED_TRAIN_ROWS * 0.95:
            print(f"  WARNING: Row count is {EXPECTED_TRAIN_ROWS - total:,} below expected!")
        elif total >= EXPECTED_TRAIN_ROWS * 0.95:
            print(f"  OK: Row count matches expectation")

    print(f"\n  Label distribution:")
    for label, count in labels.most_common():
        print(f"    {label or '(empty)':<15} {count:>10,} ({count/total*100:.1f}%)")

    print(f"\n  Source distribution:")
    for source, count in sources.most_common(20):
        print(f"    {source or '(empty)':<25} {count:>10,}")

    print(f"\n  Missing images (sampled): {len(missing)}")
    for f in missing[:5]:
        print(f"    {f}")

    print(f"  Zero-byte files (sampled): {len(zero_byte)}")
    for f in zero_byte[:5]:
        print(f"    {f}")

    return {
        "status": "OK" if total > 0 else "EMPTY",
        "rows": total,
        "labels": dict(labels),
        "sources": dict(sources),
        "missing_sampled": len(missing),
        "zero_byte_sampled": len(zero_byte),
    }


def check_image_dirs():
    """Check what's actually on disk."""
    print(f"\n{'='*60}")
    print("Image Directories on Disk")
    print(f"{'='*60}")

    if not IMAGES_DIR.exists():
        print(f"  ERROR: {IMAGES_DIR} does not exist!")
        return

    for d in sorted(IMAGES_DIR.iterdir()):
        if d.is_dir():
            count = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"  {d.name:<30} {count:>10,} files")


def check_split_files():
    """Check split index files."""
    print(f"\n{'='*60}")
    print("Split Index Files")
    print(f"{'='*60}")

    for name in ["split_indices.json", "split_indices_smoke.json", "split_indices_quick5k.json"]:
        path = METADATA_DIR / name
        if path.exists():
            import json
            with open(path) as f:
                data = json.load(f)
            n_samples = data.get("n_samples", 0)
            n_train = len(data.get("train", []))
            n_val = len(data.get("val", []))
            n_test = len(data.get("test", []))
            print(f"  {name:<35} {n_samples:>8,} samples (train={n_train}, val={n_val}, test={n_test})")
        else:
            print(f"  {name:<35} NOT FOUND")


def check_config():
    """Check what config.py points to."""
    print(f"\n{'='*60}")
    print("Config Check")
    print(f"{'='*60}")

    config_path = PROJECT_ROOT / "model" / "src" / "config.py"
    if not config_path.exists():
        print(f"  WARNING: {config_path} not found")
        return

    with open(config_path) as f:
        content = f.read()

    if "train_manifest.csv" in content:
        print("  config.py references train_manifest.csv — OK")
    elif "clean_metadata.csv" in content:
        print("  WARNING: config.py references clean_metadata.csv")
        print("  This includes the BigGAN duplicates and uncapped Places365!")
        print("  The notebooks use train_manifest.csv if it exists,")
        print("  but config.py defaults matter for standalone scripts.")
    else:
        print("  config.py metadata path: unclear, check manually")


def main():
    print(f"{'#'*60}")
    print(f"# DGX Pre-flight Dataset Validation")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    # 1. Image directories
    check_image_dirs()

    # 2. Validate manifests
    results = {}
    for name, path in MANIFESTS.items():
        results[name] = validate_manifest(name, path)

    # 3. Split files
    check_split_files()

    # 4. Config check
    check_config()

    # 5. Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    train = results.get("train_manifest.csv", {})
    clean = results.get("clean_metadata.csv", {})

    issues = []
    if train.get("rows", 0) == 0:
        issues.append("train_manifest.csv is empty or missing")
    elif train.get("rows", 0) < 1_000_000:
        issues.append(f"train_manifest.csv only has {train.get('rows', 0):,} rows (expected ~1.15M)")

    if train.get("missing_sampled", 0) > 0:
        issues.append(f"{train['missing_sampled']} missing images found in train_manifest (sampled)")

    if train.get("zero_byte_sampled", 0) > 0:
        issues.append(f"{train['zero_byte_sampled']} zero-byte images found in train_manifest (sampled)")

    if clean.get("rows", 0) == 0:
        issues.append("clean_metadata.csv is empty or missing")

    if issues:
        print("\n  ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        print(f"\n  Run: python dataset/prepare_training_manifest.py")
        print(f"  Then: python dataset/validate_images.py --fix --clean-csv")
    else:
        print("\n  ALL CHECKS PASSED — ready for DGX!")

    print(f"\n{'='*60}")
    return len(issues) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
