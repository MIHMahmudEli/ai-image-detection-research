"""
Run this AFTER downloading and extracting GenImage to dataset/images/genimage_ai/
Regenerates clean_metadata.csv with all images.
"""
import csv, hashlib
from pathlib import Path
from PIL import Image

GENIMAGE_DIR = Path("dataset/images/genimage_ai")
METADATA = Path("dataset/metadata/clean_metadata.csv")
AI_DIR = Path("dataset/images/ai_generated")

# Update metadata: add genimage images as ai_generated
print(f"Scanning {GENIMAGE_DIR} for new images...")
rows = []
for img_path in sorted(GENIMAGE_DIR.rglob("*")):
    if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        continue
    try:
        img = Image.open(img_path)
        w, h = img.size
        data = img_path.read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        rows.append({
            "image_id": img_path.stem,
            "filename": str(img_path),
            "label": "ai_generated",
            "width": w,
            "height": h,
            "file_size_bytes": len(data),
            "md5": md5,
            "source": "genimage",
        })
    except:
        pass

print(f"Found {len(rows)} GenImage images")

# Append to existing clean_metadata.csv
if METADATA.exists():
    import pandas as pd
    existing = pd.read_csv(METADATA)
    print(f"Existing metadata: {len(existing)} rows")
    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(METADATA, index=False)
    print(f"Updated metadata: {len(combined)} rows")
    print(f"  Real: {len(combined[combined['label']=='real'])}")
    print(f"  AI-Gen: {len(combined[combined['label']=='ai_generated'])}")
    print(f"  AI-Alt: {len(combined[combined['label']=='ai_altered'])}")
else:
    print(f"No existing metadata found. Created new file.")
    import pandas as pd
    pd.DataFrame(rows).to_csv(METADATA, index=False)
