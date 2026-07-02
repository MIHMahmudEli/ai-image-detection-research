"""Test DiffusionDB correct URL with range request"""
import requests

url = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-000001.zip"
headers = {"Range": "bytes=0-1000000"}
r = requests.get(url, headers=headers, timeout=30)
print(f"HTTP {r.status_code}")
cr = r.headers.get("content-range", "N/A")
print(f"Content-Range: {cr}")
print(f"Downloaded {len(r.content)/1024:.1f} KB")
if r.content[:2] == b"PK":
    print("VALID ZIP header - file exists!")
else:
    print(f"First bytes: {r.content[:50]}")
