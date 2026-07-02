"""
Download CIFAKE from Kaggle using Bearer token + direct URL.
Gets the signed download URL first, then downloads.
"""
import os, requests, zipfile, io, time
from pathlib import Path

OUT = Path("dataset/images/cifake_ai")
OUT.mkdir(parents=True, exist_ok=True)

token = "KGAT_100897ec213fc73a81649eb3d20cea99"
headers = {"Authorization": f"Bearer {token}"}

# Get dataset info first to find files
print("Getting dataset info...")
r = requests.get(
    "https://www.kaggle.com/api/v1/datasets/sagnik1511/cifake-ai-image-detection-dataset",
    headers=headers, timeout=15
)
print(f"Dataset info: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"Files: {len(data.get('datasetFiles', []))}")
    for f in data.get("datasetFiles", []):
        print(f"  {f.get('name')} - {f.get('totalBytes', 0)/1024/1024:.0f}MB")
else:
    # Try alternative: download the zip directly
    print("Trying direct download...")
    r2 = requests.get(
        "https://storage.googleapis.com/kaggle-data-sets/...", 
        # This won't work without the signed URL
        headers=headers, timeout=15
    )
    print(r2.status_code)
