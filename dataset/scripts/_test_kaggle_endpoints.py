"""Try different Kaggle API endpoints"""
import os, requests

token = os.environ["KAGGLE_API_TOKEN"]
headers = {"Authorization": f"Bearer {token}"}
owner = "sagnik1511"
ds = "cifake-ai-image-detection-dataset"

# Try various endpoints
endpoints = [
    f"https://www.kaggle.com/api/v1/datasets/{owner}/{ds}/download",
    f"https://www.kaggle.com/api/v1/datasets/{owner}/{ds}/download.zip",
    f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{ds}",
]

for url in endpoints:
    r = requests.get(url, headers=headers, timeout=15, allow_redirects=False)
    print(f"GET {url.split('/api/')[1]:60s} -> {r.status_code}")
    if r.status_code in (200, 302):
        loc = r.headers.get("location", "")
        print(f"  Location: {loc[:80]}")
        cl = r.headers.get("content-length", "?")
        print(f"  Size: {cl} bytes")
