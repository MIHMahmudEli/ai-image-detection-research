"""Check Kaggle auth status"""
import os, requests
from pathlib import Path

# Method 1: Check with Bearer token
token = "KGAT_100897ec213fc73a81649eb3d20cea99"
headers = {"Authorization": f"Bearer {token}"}
r = requests.get("https://www.kaggle.com/api/v1/account", headers=headers, timeout=10)
print(f"Account info: {r.status_code}")
if r.status_code == 200:
    print(r.json())

# Method 2: Check what username is in kaggle.json
kf = Path.home() / ".kaggle" / "kaggle.json"
if kf.exists():
    import json
    data = json.loads(kf.read_text())
    print(f"kaggle.json username: {data.get('username')}")
