import requests, time, tempfile, pyarrow.parquet as pq
from pathlib import Path

TMP = Path(tempfile.gettempdir())
start = 3
end = 6
existing = {f.stem for f in Path("dataset/images/DALL-E3").glob("*")}
print(f"Existing images: {len(existing)}")

for i in range(start, end):
    url = f"https://huggingface.co/api/datasets/OpenDatasets/dalle-3-dataset/parquet/default/train/{i}.parquet"
    print(f"\n[{i}] Downloading...", end=" ")
    t0 = time.time()
    resp = requests.get(url, timeout=600)
    print(f"{len(resp.content)/1e6:.0f}MB in {time.time()-t0:.0f}s")

    p = TMP / f"dalle3_{i}.parquet"
    with open(p, "wb") as f:
        f.write(resp.content)
    t1 = time.time()
    table = pq.read_table(p)
    df = table.to_pandas()
    print(f"  Read {len(df)} rows in {time.time()-t1:.0f}s")
    p.unlink()

    new_count = 0
    for _, row in df.iterrows():
        img_data = row["image"]
        if isinstance(img_data, dict):
            b = img_data.get("bytes", b"")
        elif isinstance(img_data, bytes):
            b = img_data
        else:
            continue
        if len(b) < 100:
            continue
        img_id = row["image_hash"]
        if img_id in existing:
            continue
        ext = "jpg" if b[:3] == b"\xff\xd8\xff" else "png"
        (Path("dataset/images/DALL-E3") / f"{img_id}.{ext}").write_bytes(b)
        new_count += 1

    print(f"  New images: {new_count}")
    existing = {f.stem for f in Path("dataset/images/DALL-E3").glob("*")}
    print(f"  Total now: {len(existing)}")

print(f"\nDone. Total: {len(existing)}")
