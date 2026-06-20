"""Consolidate images from alternate folders and generate metadata CSV."""
from pathlib import Path
import shutil, hashlib, io, json
from datetime import datetime
from PIL import Image
import pandas as pd

v3 = Path(__file__).parent
output_dir = v3 / "ai_dataset_50k"

real_dir = output_dir / "real_images"
ai_dir = output_dir / "ai_generated_images"
altered_dir = output_dir / "ai_altered_images"

# Consolidate from alternate folders
for src_name, dst_dir in [("ai_generated_dataset/images", ai_dir),
                           ("ai_altered_dataset/images", altered_dir)]:
    src_dir = v3 / src_name
    if src_dir.exists():
        for f in src_dir.iterdir():
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                dest = dst_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)

# Generate metadata
all_rows = []
for img_dir, img_type in [(real_dir, "real"),
                           (ai_dir, "ai_generated"),
                           (altered_dir, "ai_altered")]:
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
            continue
        try:
            data = img_path.read_bytes()
            md5 = hashlib.md5(data).hexdigest()
            img = Image.open(io.BytesIO(data))
            all_rows.append({
                "image_id": img_path.stem,
                "filename": img_path.name,
                "source": "consolidated",
                "source_url": "",
                "download_date": datetime.now().isoformat(),
                "image_type": img_type,
                "width": img.size[0], "height": img.size[1],
                "format": img.format or "",
                "file_size_kb": round(len(data) / 1024, 2),
                "color_space": img.mode,
                "source_metadata": json.dumps({}),
                "md5_hash": md5,
                "is_duplicate": False, "duplicate_of": None,
            })
        except:
            continue

df = pd.DataFrame(all_rows)
csv_path = output_dir / "dataset_metadata_all.csv"
df.to_csv(csv_path, index=False)

real_c = len(list(real_dir.iterdir()))
ai_c = len(list(ai_dir.iterdir()))
alt_c = len(list(altered_dir.iterdir()))
total = real_c + ai_c + alt_c
print(f"Real: {real_c}  AI: {ai_c}  Altered: {alt_c}  TOTAL: {total}")
print(f"Metadata saved: {csv_path}")
