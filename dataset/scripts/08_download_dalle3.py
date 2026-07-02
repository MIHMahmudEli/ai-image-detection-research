"""
Download DALL-E 3 dataset from HuggingFace — batch mode.
Processes a range of parquet files so it can run within time limits.

Usage:
  python dataset/scripts/08_download_dalle3.py          # all files
  python dataset/scripts/08_download_dalle3.py 0 10     # files 0-9
"""
import csv, sys, tempfile
from pathlib import Path
import requests
import pyarrow.parquet as pq

OUTPUT_DIR = Path("dataset/images/DALL-E3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_PATH = Path("dataset/metadata/dalle3_metadata.csv")
TMP = Path(tempfile.gettempdir())
BASE_URL = "https://huggingface.co/api/datasets/OpenDatasets/dalle-3-dataset/parquet/default/train"
NUM_FILES = 67

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_FILES

existing = {f.stem for f in OUTPUT_DIR.glob("*")}
rows = []

for i in range(start, end):
    url = f"{BASE_URL}/{i}.parquet"
    print(f"[{i+1}/{NUM_FILES}] {url}", end=" ... ")
    resp = requests.get(url, timeout=180)
    if resp.status_code != 200:
        print(f"FAILED HTTP {resp.status_code}")
        continue
    p = TMP / f"dalle3_{i}.parquet"
    with open(p, "wb") as f:
        f.write(resp.content)
    table = pq.read_table(p)
    p.unlink(missing_ok=True)
    df = table.to_pandas()
    n_new = 0
    for _, row in df.iterrows():
        img_data = row["image"]
        if isinstance(img_data, dict):
            img_bytes = img_data.get("bytes", b"")
        elif isinstance(img_data, bytes):
            img_bytes = img_data
        else:
            continue
        if len(img_bytes) < 100:
            continue
        img_id = row["image_hash"]
        if img_id in existing:
            continue
        ext = "jpg" if img_bytes[:3] == b"\xff\xd8\xff" else "png"
        fname = f"{img_id}.{ext}"
        (OUTPUT_DIR / fname).write_bytes(img_bytes)
        rows.append({
            "image_id": img_id,
            "filename": f"dataset/images/DALL-E3/{fname}",
            "label": "ai_generated",
            "generator": "dalle3",
            "width": 1024,
            "height": 1024,
            "caption": (row.get("caption") or "")[:500],
            "link": row.get("link") or "",
        })
        n_new += 1
    print(f"{len(df)} rows, {n_new} new, total {len(rows)}")

# Merge with existing metadata
existing_rows = []
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)

all_rows = existing_rows + rows
with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=[
        "image_id", "filename", "label", "generator",
        "width", "height", "caption", "link"
    ])
    w.writeheader()
    w.writerows(all_rows)

total_imgs = len({f.stem for f in OUTPUT_DIR.glob("*")})
print(f"\nDone [{start}-{end-1}]. Total on disk: {total_imgs}, metadata rows: {len(all_rows)}")
