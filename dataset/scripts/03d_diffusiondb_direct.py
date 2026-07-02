"""
Direct download from DiffusionDB on HuggingFace.
Lists files and downloads individual images from the large_random_1k subset.
"""
import requests, time, hashlib, json
from pathlib import Path

OUT = Path("dataset/images/ai_generated")
OUT.mkdir(parents=True, exist_ok=True)

# The large_random_1k images are stored at specific paths on HuggingFace
# Let's try downloading them directly
base = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images_random_large"

print("Trying direct download...")
t0 = time.time()
downloaded = 0
errors = 0

for i in range(1000):
    fname = f"{i:05d}.jpg"
    url = f"{base}/{fname}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            out_name = f"diffusiondb_{i:05d}.jpg"
            (OUT / out_name).write_bytes(r.content)
            downloaded += 1
            errors = 0
        else:
            errors += 1
        
        if errors > 20:
            print(f"Too many errors (last: {r.status_code}), trying parquet method...")
            break
        
        if downloaded % 100 == 0 and downloaded > 0:
            print(f"  {downloaded} images ({downloaded/(time.time()-t0):.1f}/s)")
    except Exception as e:
        errors += 1
        time.sleep(0.5)

if downloaded == 0:
    # Try parquet approach - download metadata parquet and get image paths
    print("Direct download failed, trying parquet metadata...")
    url = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet"
    resp = requests.get(url, timeout=60)
    if resp.status_code == 200:
        import pyarrow.parquet as pq
        import io
        table = pq.read_table(io.BytesIO(resp.content))
        df = table.to_pandas()
        print(f"Parquet columns: {list(df.columns)}")
        print(f"Rows: {len(df)}")
        print(df.head(3).to_string())
    else:
        print(f"Parquet download failed: {resp.status_code}")

print(f"\nDownloaded: {downloaded} images in {time.time()-t0:.0f}s")
total = len(list(OUT.glob("diffusiondb_*")))
print(f"Total DiffusionDB in dir: {total}")
