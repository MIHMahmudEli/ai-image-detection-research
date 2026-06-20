"""
Smart collector: calculates how many images we already have
and runs the 50K collector only for the remaining needed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from complete_dataset_collector import DatasetCollectorV2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.jpg")) + list(directory.glob("*.png")) + list(directory.glob("*.webp")))


def main():
    output_dir = Path("./ai_dataset_50k")
    real_dir = output_dir / "real_images"
    ai_dir = output_dir / "ai_generated_images"
    altered_dir = output_dir / "ai_altered_images"
    ai_gen_alt = Path("./ai_generated_dataset") / "images"
    ai_altered_alt = Path("./ai_altered_dataset") / "images"

    # Count existing
    existing_real = count_images(real_dir)
    existing_ai = count_images(ai_dir) + count_images(ai_gen_alt)
    existing_altered = count_images(altered_dir) + count_images(ai_altered_alt)

    TARGET_REAL = 25000
    TARGET_AI = 17500
    TARGET_ALTERED = 7500

    needed_real = max(0, TARGET_REAL - existing_real)
    needed_ai = max(0, TARGET_AI - existing_ai)
    needed_altered = max(0, TARGET_ALTERED - existing_altered)

    print(f"\n{'='*60}")
    print(f"Current dataset status:")
    print(f"{'='*60}")
    print(f"  Real photos:     {existing_real:>6,} / {TARGET_REAL:,}  (need {needed_real:,} more)")
    print(f"  AI-generated:    {existing_ai:>6,} / {TARGET_AI:,}  (need {needed_ai:,} more)")
    print(f"  AI-altered:      {existing_altered:>6,} / {TARGET_ALTERED:,}  (need {needed_altered:,} more)")
    print(f"  {'-'*37}")
    print(f"  Total:           {existing_real + existing_ai + existing_altered:>6,} / {TARGET_REAL + TARGET_AI + TARGET_ALTERED:,}")
    need_total = needed_real + needed_ai + needed_altered

    if need_total == 0:
        print(f"\n Dataset already complete at 50K! No collection needed.")
        return

    print(f"\nCollection needed: ~{need_total:,} images")

    # Import and consolidate any AI images from alternate folders
    if existing_ai > count_images(ai_dir):
        import shutil
        ai_gen_alt.mkdir(parents=True, exist_ok=True)
        count_copied = 0
        for f in ai_gen_alt.glob("*"):
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                dest = ai_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                    count_copied += 1
        if count_copied > 0:
            logger.info(f"  Consolidated {count_copied} AI images from alternate folder")
            existing_ai = count_images(ai_dir)
            needed_ai = max(0, TARGET_AI - existing_ai)

    # Run collection for remaining
    if need_total > 0:
        print(f"\n{'='*60}")
        print(f"Starting collection for {need_total:,} remaining images...")
        print(f"{'='*60}")

        collector = DatasetCollectorV2(output_dir=output_dir)

        # Only collect what's needed
        if needed_real > 0:
            api_keys = collector.load_api_keys()
            collector.collect_real_images(api_keys, needed_real)

        if needed_ai > 0:
            api_keys = collector.load_api_keys()
            collector.collect_ai_images(api_keys, needed_ai)

        if needed_altered > 0:
            collector.collect_ai_altered_images(real_dir, needed_altered)

    # Final counts
    final_real = count_images(real_dir)
    final_ai = count_images(ai_dir)
    final_altered = count_images(altered_dir)
    final_total = final_real + final_ai + final_altered

    print(f"\n{'='*60}")
    print(f"FINAL DATASET STATUS")
    print(f"{'='*60}")
    print(f"  Real photos:     {final_real:>6,}")
    print(f"  AI-generated:    {final_ai:>6,}")
    print(f"  AI-altered:      {final_altered:>6,}")
    print(f"  ")
    print(f"  Total:           {final_total:>6,}")
    print(f"{'='*60}")

    if final_total >= 50000:
        print(f" 50K dataset complete!")
    else:
        print(f" {final_total:,} / 50,000 collected. "
              f"Need {50000 - final_total:,} more. "
              f"Run this script again to continue.")

    # Generate consolidated metadata
    if final_total > 0:
        from complete_dataset_collector import ImageMetadata, ImageType, ImageSource
        import json, hashlib, io
        from datetime import datetime
        from PIL import Image
        import pandas as pd

        all_metadata = []

        for img_path in list(real_dir.glob("*")) + list(ai_dir.glob("*")) + list(altered_dir.glob("*")):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
                continue
            try:
                data = img_path.read_bytes()
                md5 = hashlib.md5(data).hexdigest()
                img = Image.open(io.BytesIO(data))
                img_type = "real" if img_path.parent.name == "real_images" else \
                           "ai_generated" if img_path.parent.name == "ai_generated_images" else "ai_altered"
                all_metadata.append({
                    "image_id": img_path.stem,
                    "filename": img_path.name,
                    "source": "consolidated",
                    "source_url": "",
                    "download_date": datetime.now().isoformat(),
                    "image_type": img_type,
                    "width": img.size[0],
                    "height": img.size[1],
                    "format": img.format or "",
                    "file_size_kb": round(len(data) / 1024, 2),
                    "color_space": img.mode,
                    "source_metadata": {},
                    "md5_hash": md5,
                    "is_duplicate": False,
                    "duplicate_of": None,
                })
            except:
                continue

        df = pd.DataFrame(all_metadata)
        df.to_csv(output_dir / "dataset_metadata_all.csv", index=False)
        logger.info(f"   Consolidated metadata: {output_dir / 'dataset_metadata_all.csv'} ({len(df)} records)")


if __name__ == "__main__":
    main()

