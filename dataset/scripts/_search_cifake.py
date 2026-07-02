"""Search CIFAKE on Kaggle API"""
import os, requests

headers = {"Authorization": f"Bearer {os.environ['KAGGLE_API_TOKEN']}"}
r = requests.get("https://www.kaggle.com/api/v1/datasets/search?search=cifake", headers=headers, timeout=15)
if r.status_code == 200:
    results = r.json()
    for ds in results.get("results", [])[:5]:
        ref = ds.get("ref", "")
        title = ds.get("title", "")
        url = ds.get("downloadUrl", "")
        print(f"{ref} - {title}")
        print(f"  URL: {url}")
else:
    print(r.status_code, r.text[:200])
