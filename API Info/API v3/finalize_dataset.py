"""Consolidate images and generate metadata without PIL (faster)."""
from pathlib import Path
import shutil, hashlib, csv, os
from datetime import datetime

v3 = Path(os.path.dirname(os.path.abspath(__file__)))
output_dir = v3 / "ai_dataset_50k"
real_dir = output_dir / "real_images"
ai_dir = output_dir / "ai_generated_images"
altered_dir = output_dir / "ai_altered_images"

# Consolidate from alternate folders
for src_dir, dst_dir in [(v3 / "ai_generated_dataset", ai_dir),
                          (v3 / "ai_altered_dataset", altered_dir)]:
    if src_dir.exists():
        copied = 0
        for f in src_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                dest = dst_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                    copied += 1
        if copied:
            print(f"Consolidated {copied} from {src_dir.name}")

# Generate metadata
now = datetime.now().isoformat()
csv_path = output_dir / "dataset_metadata_all.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "filename", "source", "source_url", "download_date",
                "image_type", "width", "height", "format", "file_size_kb",
                "color_space", "md5_hash", "is_duplicate", "duplicate_of"])
    for img_dir, img_type in [(real_dir, "real"), (ai_dir, "ai_generated"), (altered_dir, "ai_altered")]:
        for fpath in img_dir.iterdir():
            if not fpath.is_file() or fpath.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
                continue
            try:
                data = fpath.read_bytes()
                w.writerow([fpath.stem, fpath.name, "consolidated", "", now,
                           img_type, 0, 0, "", round(len(data)/1024, 2),
                           "", hashlib.md5(data).hexdigest(), "False", ""])
            except:
                continue

real_c = sum(1 for _ in real_dir.iterdir())
ai_c = sum(1 for _ in ai_dir.iterdir())
alt_c = sum(1 for _ in altered_dir.iterdir())
print(f"Real: {real_c}  AI: {ai_c}  Altered: {alt_c}  TOTAL: {real_c+ai_c+alt_c}")
print(f"Metadata: {csv_path}")
