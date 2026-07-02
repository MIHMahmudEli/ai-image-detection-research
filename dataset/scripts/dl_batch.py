"""Download a small batch of DALL-E3 parquet files."""
import sys, requests, pyarrow.parquet as pq, tempfile, csv
from pathlib import Path

TMP = Path(tempfile.gettempdir())
out = Path("dataset/images/DALL-E3")
out.mkdir(parents=True, exist_ok=True)
meta_path = Path("dataset/metadata/dalle3_metadata.csv")

start, end = int(sys.argv[1]), int(sys.argv[2])
existing = {f.stem for f in out.glob("*")}
print(f"Existing: {len(existing)}")

new_rows = []
for i in range(start, end):
    url = f"https://huggingface.co/api/datasets/OpenDatasets/dalle-3-dataset/parquet/default/train/{i}.parquet"
    print(f"[{i}] Downloading...", end=" ", flush=True)
    resp = requests.get(url, timeout=600)
    if resp.status_code != 200:
        print(f"FAILED {resp.status_code}")
        continue
    print(f"{len(resp.content)/1e6:.0f}MB", end=" ")
    p = TMP / f"dalle3_{i}.parquet"
    with open(p, "wb") as f:
        f.write(resp.content)
    table = pq.read_table(p)
    df = table.to_pandas()
    p.unlink()
    print(f"{len(df)} rows", end=" ")
    n = 0
    for _, row in df.iterrows():
        img = row["image"]
        b = img.get("bytes", b"") if isinstance(img, dict) else (img if isinstance(img, bytes) else b"")
        if len(b) < 100:
            continue
        img_id = row["image_hash"]
        if img_id in existing:
            continue
        ext = "jpg" if b[:3] == b"\xff\xd8\xff" else "png"
        (out / f"{img_id}.{ext}").write_bytes(b)
        new_rows.append({
            "image_id": img_id,
            "filename": f"dataset/images/DALL-E3/{img_id}.{ext}",
            "label": "ai_generated",
            "generator": "dalle3",
            "width": 1024,
            "height": 1024,
            "caption": (row.get("caption") or "")[:500],
            "link": row.get("link") or "",
        })
        n += 1
    existing.add(row["image_hash"] for _ in range(0))
    existing = {f.stem for f in out.glob("*")}
    print(f"{n} new")

# Merge with existing metadata
all_rows = new_rows[:]
if meta_path.exists():
    with open(meta_path, "r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f)) + new_rows
with open(meta_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["image_id","filename","label","generator","width","height","caption","link"])
    w.writeheader()
    w.writerows(all_rows)
print(f"Total: {len(existing)} images, {len(all_rows)} metadata rows")
