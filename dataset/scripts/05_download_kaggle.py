"""Download from Kaggle using working Bearer token"""
import os, requests, zipfile, io, time, json
from pathlib import Path

OUT = Path("dataset/images/kaggle_ai")
OUT.mkdir(parents=True, exist_ok=True)

token = "KGAT_100897ec213fc73a81649eb3d20cea99"
headers = {"Authorization": f"Bearer {token}"}

# Download ai-vs-real-images-dataset (237 MB)
ds = "rhythmghai/ai-vs-real-images-dataset"
url = f"https://www.kaggle.com/api/v1/datasets/{ds}/download"

print(f"Downloading {ds}...")
t0 = time.time()
r = requests.get(url, headers=headers, timeout=600, stream=True)
print(f"Status: {r.status_code}")

if r.status_code == 200:
    total = int(r.headers.get("content-length", 0))
    print(f"Size: {total/1024/1024:.0f} MB")
    
    content = b""
    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            content += chunk
    print(f"Downloaded {len(content)/1024/1024:.0f} MB in {time.time()-t0:.0f}s")
    
    if content[:2] == b"PK":
        print("Extracting...")
        count = 0
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                if name.endswith((".jpg", ".jpeg", ".png")):
                    # Determine if real or AI
                    clean = name.replace("\\", "/")
                    (OUT / Path(name).name).write_bytes(zf.read(name))
                    count += 1
        print(f"Extracted {count} images to {OUT}")
    else:
        print(f"Not ZIP. First 200 bytes: {content[:200]}")
elif r.status_code == 302:
    print(f"Redirect to: {r.headers.get('location', '')[:80]}")
else:
    print(f"Failed: {r.text[:300]}")
