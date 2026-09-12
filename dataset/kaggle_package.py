"""
Package a Kaggle-ready subset of the dataset.
Creates a kaggle_dataset/ directory with images + metadata,
ready to upload as a Kaggle Dataset.

Usage:
  python dataset/kaggle_package.py --max-images 50000 --output ./kaggle_dataset
"""
import os
import csv
import json
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "dataset" / "images"
METADATA_DIR = PROJECT_ROOT / "dataset" / "metadata"


def load_metadata(metadata_path: Path) -> list:
    """Load metadata CSV into a list of dicts."""
    rows = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def resolve_image_path(row: dict) -> Path | None:
    """Resolve the actual image file path from a metadata row."""
    filename = row.get("filename", "")
    if not filename:
        return None

    # Normalize path separators
    filename = filename.replace("\\", "/")

    # Try direct path from images root
    candidate = IMAGES_DIR / filename
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate

    # Try subdirectories
    for subdir in ["real", "ai_generated", "ai_altered"]:
        candidate = IMAGES_DIR / subdir / filename
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    return None


def create_kaggle_dataset(
    max_images: int = 50000,
    output_dir: str = "./kaggle_dataset",
    seed: int = 42,
    train_manifest: str = None,
):
    """
    Create a balanced, stratified Kaggle dataset.
    - 50% real, 50% AI-generated (or as balanced as possible)
    - Images copied to output_dir/images/
    - Metadata CSV written to output_dir/
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)

    # Load metadata
    meta_path = Path(train_manifest) if train_manifest else METADATA_DIR / "train_manifest.csv"
    if not meta_path.exists():
        meta_path = METADATA_DIR / "clean_metadata.csv"
    print(f"Loading metadata from: {meta_path}")
    rows = load_metadata(meta_path)
    print(f"Total metadata rows: {len(rows)}")

    # Separate by label
    real_rows = [r for r in rows if r.get("label") == "real"]
    ai_rows = [r for r in rows if r.get("label") == "ai_generated" or r.get("label") == "ai_altered"]
    print(f"Real: {len(real_rows)}, AI: {len(ai_rows)}")

    # Stratified sample
    rng = random.Random(seed)
    per_class = max_images // 2

    rng.shuffle(real_rows)
    rng.shuffle(ai_rows)

    selected_real = real_rows[:per_class]
    selected_ai = ai_rows[:per_class]
    selected = selected_real + selected_ai
    rng.shuffle(selected)

    print(f"Selected: {len(selected_real)} real + {len(selected_ai)} AI = {len(selected)} total")

    # Copy images and build new metadata
    new_metadata = []
    copied = 0
    skipped = 0

    for i, row in enumerate(selected):
        src = resolve_image_path(row)
        if src is None:
            skipped += 1
            continue

        # Create a flat filename to avoid deep directory structures
        ext = src.suffix.lower()
        dst_name = f"{row.get('image_id', f'img_{i:06d}')}_{row.get('label', 'unknown')}{ext}"
        dst = output / "images" / dst_name

        try:
            shutil.copy2(src, dst)
            copied += 1

            new_metadata.append({
                "image_id": row.get("image_id", f"img_{i:06d}"),
                "filename": dst_name,
                "label": row.get("label", "unknown"),
                "source": row.get("source", "unknown"),
                "generator": row.get("generator", ""),
            })

            if (copied % 1000) == 0:
                print(f"  Copied {copied}/{len(selected)} images...")

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Warning: failed to copy {src.name}: {e}")

    # Write metadata CSV
    meta_out = output / "metadata.csv"
    with open(meta_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "filename", "label", "source", "generator"])
        writer.writeheader()
        writer.writerows(new_metadata)

    # Write dataset description for Kaggle
    readme = output / "README.md"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(f"""# MFFT AI Image Detection Dataset (Subset)

## Description
Balanced subset for training the Multi-Frequency Fusion Transformer (MFFT).
Contains {len(new_metadata)} images (50% real, 50% AI-generated).

## Structure
```
images/          - All images (flattened)
metadata.csv     - Labels and source information
```

## Labels
- `real` - Authentic photographs (label=0)
- `ai_generated` - AI-generated images (label=1)
- `ai_altered` - AI-manipulated deepfakes (label=1)

## Sources
Real: Unsplash, Pexels, Pixabay, ImageNet, Places365
AI: BigGAN, Stable Diffusion, DALL-E3, Midjourney, Glide, etc.

## Usage
```python
import pandas as pd
df = pd.read_csv("metadata.csv")
```

## Statistics
- Total images: {len(new_metadata)}
- Real: {sum(1 for r in new_metadata if r['label'] == 'real')}
- AI: {sum(1 for r in new_metadata if r['label'] != 'real')}
""")

    # Summary
    total_size = sum(f.stat().st_size for f in (output / "images").iterdir() if f.is_file())
    print(f"\n{'='*60}")
    print(f"Kaggle dataset created: {output}")
    print(f"  Images: {copied} copied, {skipped} skipped")
    print(f"  Size: {total_size / 1e6:.1f} MB")
    print(f"  Metadata: {meta_out}")
    print(f"{'='*60}")
    print(f"\nTo upload:")
    print(f"  1. Install Kaggle CLI: pip install kaggle")
    print(f"  2. Create dataset metadata: kaggle datasets init -p {output}")
    print(f"  3. Edit {output}/dataset-metadata.json")
    print(f"  4. Upload: kaggle datasets create -p {output} --dir-mode zip")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package Kaggle dataset")
    parser.add_argument("--max-images", type=int, default=50000, help="Max images per class")
    parser.add_argument("--output", type=str, default="./kaggle_dataset", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-manifest", type=str, default=None, help="Path to train_manifest.csv")
    args = parser.parse_args()

    create_kaggle_dataset(
        max_images=args.max_images,
        output_dir=args.output,
        seed=args.seed,
        train_manifest=args.train_manifest,
    )
