"""Search for AI image datasets on Kaggle using working API"""
import os, requests, json

headers = {"Authorization": "Bearer KGAT_100897ec213fc73a81649eb3d20cea99"}

# Try search via the working API
r = requests.get(
    "https://www.kaggle.com/api/v1/datasets/list",
    headers=headers,
    params={"search": "ai generated image detection", "sortBy": "hottest", "page": 1, "pageSize": 20},
    timeout=15
)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    if isinstance(data, list):
        for ds in data:
            ref = ds.get("ref", ds.get("dataset", ""))
            size = ds.get("totalBytes", 0)
            print(f"{ref:60s} {size/1024/1024:.0f} MB")
    elif isinstance(data, dict):
        print(json.dumps(data, indent=2)[:1000])
else:
    print(r.text[:500])
