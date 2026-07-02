"""Test Kaggle API auth methods"""
import os, requests
from kagglehub import dataset_download

token = os.environ.get("KAGGLE_API_TOKEN", "")
print(f"Token: {token[:20]}...")

# Method 1: Direct API with Bearer token
headers = {"Authorization": f"Bearer {token}"}
r = requests.get("https://www.kaggle.com/api/v1/competitions/list", headers=headers, timeout=10)
print(f"Bearer token: {r.status_code}")

# Method 2: Try dataset download with env var set
try:
    p = dataset_download("sagnik1511/cifake-ai-image-detection-dataset")
    print(f"Success: {p}")
except Exception as e:
    print(f"Failed: {str(e)[:100]}")
