"""
Download CIFAKE from Kaggle using direct API (Bearer token).
Doesn't rely on kagglehub library auth.
"""
import os, requests, json, zipfile, io, time
from pathlib import Path

OUT = Path("dataset/images/cifake_real")
OUT_FAKE = Path("dataset/images/cifake_ai")
OUT.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)

token = os.environ.get("KAGGLE_API_TOKEN", "")
headers = {"Authorization": f"Bearer {token}"}

# Step 1: Get dataset download URL
print("Getting CIFAKE download info...")
r = requests.post(
    "https://www.kaggle.com/api/v1/datasets/sagnik1511/cifake-ai-image-detection-dataset/download",
    headers=headers, timeout=30
)
print(f"Status: {r.status_code}")
if r.status_code == 200 or r.status_code == 302:
    # Follow redirect
    url = r.url if r.status_code == 302 else r.json().get("url", "")
    print(f"Download URL: {url[:80]}...")
else:
    # Try alternate API endpoint
    r2 = requests.get(
        "https://www.kaggle.com/api/v1/datasets/sagnik1511/cifake-ai-image-detection-dataset/download.zip",
        headers=headers, timeout=30, stream=True
    )
    print(f"Direct download: {r2.status_code}")
    if r2.status_code == 200:
        total = int(r2.headers.get("content-length", 0))
        print(f"Size: {total/1024/1024:.0f} MB")
        print("Downloading...")
        t0 = time.time()
        content = b""
        for chunk in r2.iter_content(chunk_size=8192):
            if chunk:
                content += chunk
        print(f"Downloaded {len(content)/1024/1024:.0f} MB in {time.time()-t0:.0f}s")
        
        # Extract
        print("Extracting...")
        count_real = 0
        count_fake = 0
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                if name.endswith(".jpg") or name.endswith(".png"):
                    data = zf.read(name)
                    if "/REAL/" in name.replace("\\", "/"):
                        (OUT / Path(name).name).write_bytes(data)
                        count_real += 1
                    elif "/FAKE/" in name.replace("\\", "/"):
                        (OUT_FAKE / Path(name).name).write_bytes(data)
                        count_fake += 1
        
        print(f"Real: {count_real}, AI: {count_fake}")
else:
    print(f"Failed: {r.text[:200]}")
