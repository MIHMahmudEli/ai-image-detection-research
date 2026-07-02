"""List accessible Kaggle datasets about AI image detection"""
import os, requests

headers = {"Authorization": "Bearer KGAT_100897ec213fc73a81649eb3d20cea99"}

# Try to list datasets I can access
r = requests.get("https://www.kaggle.com/api/v1/datasets/list", headers=headers, timeout=15)
print(f"List: {r.status_code}")

# Try user-specific endpoint
r2 = requests.get("https://www.kaggle.com/api/v1/account", headers=headers, timeout=15)
print(f"Account: {r2.status_code}")
if r2.status_code == 200:
    print(r2.json())
