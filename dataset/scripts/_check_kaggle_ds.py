"""Get dataset details from Kaggle API"""
import requests

headers = {"Authorization": "Bearer KGAT_100897ec213fc73a81649eb3d20cea99"}

# Try to get dataset info
ref = "rhythmghai/ai-vs-real-images-dataset"
r = requests.get(f"https://www.kaggle.com/api/v1/datasets/{ref}", headers=headers, timeout=15)
print(f"Info: {r.status_code}")
if r.status_code == 200:
    print(list(r.json().keys())[:10])
else:
    print(r.text[:200])

# Try the files listing endpoint
r2 = requests.get(f"https://www.kaggle.com/api/v1/datasets/{ref}/files", headers=headers, timeout=15)
print(f"Files: {r2.status_code}")
if r2.status_code == 200:
    print(r2.json()[:3])
else:
    print(r2.text[:200])
