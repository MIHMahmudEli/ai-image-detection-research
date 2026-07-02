"""
Download CIFAKE dataset from Kaggle (60K real + 60K AI images).
Uses kagglehub library — no API key needed for public datasets.
"""
import time, shutil, hashlib, csv
from pathlib import Path

print("=" * 60)
print("Downloading CIFAKE dataset from Kaggle...")
print("=" * 60)

t0 = time.time()
import kagglehub
path = kagglehub.dataset_download("sagnik1511/cifake-ai-image-detection-dataset")
print(f"Downloaded to: {path}")
print(f"Time: {time.time()-t0:.0f}s")
print()

# List downloaded files
files = list(Path(path).rglob("*"))
for f in files:
    size = f.stat().st_size if f.is_file() else 0
    print(f"  {f.relative_to(path)} ({size/1024/1024:.0f} MB)" if f.is_file() else f"  {f.relative_to(path)}/")
