"""Find the correct DiffusionDB download URLs"""
import requests

# Test different possible URL patterns
base = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main"
paths = [
    "part_00000.zip",
    "images/part_00000.zip",
    "data/part_00000.zip",
    "image/part_00000.zip",
    "diffusiondb/part_00000.zip",
]

for path in paths:
    url = f"{base}/{path}"
    try:
        r = requests.head(url, timeout=10, allow_redirects=True)
        print(f"{path:40s} -> HTTP {r.status_code} ({r.headers.get('content-length', '?')} bytes)")
    except Exception as e:
        print(f"{path:40s} -> ERROR: {e}")
