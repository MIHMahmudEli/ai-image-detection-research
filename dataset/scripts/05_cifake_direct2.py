"""Simpler: Get download URL from Kaggle API then download"""
import os, requests, zipfile, io, time
from pathlib import Path

OUT = Path("dataset/images/cifake_real")
OUT_FAKE = Path("dataset/images/cifake_ai")
OUT.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)

token = os.environ["KAGGLE_API_TOKEN"]
headers = {"Authorization": f"Bearer {token}"}

# Get signed download URL from Kaggle API
print("Requesting download URL...")
resp = requests.post(
    "https://www.kaggle.com/api/v1/datasets/sagnik1511/cifake-ai-image-detection-dataset/download.zip",
    headers=headers, timeout=30
)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    total = int(resp.headers.get("content-length", 0))
    print(f"Content-Length: {total/1024/1024:.0f} MB")
    
    t0 = time.time()
    content = resp.content
    print(f"Downloaded {len(content)/1024/1024:.0f} MB in {time.time()-t0:.0f}s")
    
    if content[:2] == b"PK":
        print("Valid ZIP, extracting...")
        real, fake = 0, 0
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                if name.endswith((".jpg", ".png")):
                    data = zf.read(name)
                    p = name.replace("\\", "/")
                    if "/REAL/" in p or p.startswith("REAL/"):
                        (OUT / Path(name).name).write_bytes(data)
                        real += 1
                    elif "/FAKE/" in p or p.startswith("FAKE/"):
                        (OUT_FAKE / Path(name).name).write_bytes(data)
                        fake += 1
        print(f"Real: {real}, AI: {fake}")
    else:
        print(f"Not a ZIP. First bytes: {content[:100]}")
else:
    print(f"Failed: {resp.status_code}")
    print(resp.text[:300])
