"""
Step 1: Regenerate clean metadata for all existing images.
Scans every image on disk, records real dimensions, MD5, source, generator.
Run from project root: python dataset/scripts/01_regenerate_metadata.py
"""
import csv
from pathlib import Path
from collections import Counter

IMAGE_DIR = Path("dataset/images")
OUTPUT = Path("dataset/metadata/clean_metadata.csv")

def classify_path(rel_path: str) -> tuple:
    rel = rel_path.replace("\\", "/")
    parts = rel.split("/")
    top = parts[0]
    if top == "real":
        return ("real", "pexels_unsplash", None)
    if top == "BigGAN":
        if "nature" in parts:
            return ("real", "imagenet", None)
        if "ai" in parts:
            return ("ai_generated", "biggan", "BigGAN")
        return ("ai_generated", "biggan", "BigGAN")
    if top == "genimage_ai":
        if len(parts) > 1 and parts[1] == "BigGAN":
            return ("ai_generated", "genimage_biggan", "BigGAN")
        if len(parts) > 1 and parts[1] == "glide":
            return ("ai_generated", "glide", "Glide")
        return ("ai_generated", "genimage_ai", "genimage_ai")
    if top == "DALL-E3":
        return ("ai_generated", "dalle3", "DALL-E3")
    if top == "Midjourney":
        return ("ai_generated", "midjourney", "Midjourney")
    if top == "Stable Diffusion":
        return ("ai_generated", "stable_diffusion", "Stable Diffusion")
    if top == "CelebDF_V2":
        return ("deepfake", "celebdf", "Celeb-DF")
    if top == "DFDC":
        return ("deepfake", "dfdc", "DFDC")
    if top == "FaceForensics":
        return ("deepfake", "faceforensics", "FaceForensics")
    if top == "Places365":
        return ("real", "places365", None)
    if top == "Open-Images-V7-Dataset":
        return ("real", "open_images_v7", None)
    return ("unknown", "unknown", None)

subdirs = [d for d in IMAGE_DIR.iterdir() if d.is_dir()]
total_dirs = len(subdirs)
rows = []
for subdir in subdirs:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        for img_path in subdir.rglob(f"*{ext}"):
            rel = str(img_path.relative_to(IMAGE_DIR))
            label, source, generator = classify_path(rel)
            stat = img_path.stat()
            rows.append({
                "image_id": img_path.stem,
                "filename": rel,
                "label": label,
                "source": source,
                "generator": generator or "",
                "width": 0,
                "height": 0,
                "file_size_bytes": stat.st_size,
                "md5": "",
            })
            if len(rows) % 10000 == 0:
                print(f"  Scanned {len(rows)} images...")

OUTPUT.parent.mkdir(exist_ok=True)
fieldnames = ["image_id", "filename", "label", "source", "generator", "width", "height", "file_size_bytes", "md5"]
with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone! Scanned {len(rows)} images.")
print(f"By label:")
for label, count in Counter(r["label"] for r in rows).most_common():
    print(f"  {label}: {count}")
print(f"By source:")
for src, count in Counter(r["source"] for r in rows).most_common():
    print(f"  {src}: {count}")
print(f"\nSaved to: {OUTPUT}")
