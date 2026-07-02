import time, requests
from pathlib import Path
from PIL import Image

OUT = Path("dataset/images/ai_generated")
OUT.mkdir(parents=True, exist_ok=True)

# Get image URLs from API
resp = requests.get("https://civitai.com/api/v1/images?limit=5&page=1&nsfw=false", timeout=15)
items = resp.json().get("items", [])

for i, item in enumerate(items):
    img_url = item.get("url", "")
    print(f"{i+1}. URL: {img_url[:80]}...")
    t0 = time.time()
    img_resp = requests.get(img_url, timeout=30)
    dt = time.time() - t0
    print(f"   Downloaded {len(img_resp.content)/1024:.0f} KB in {dt:.1f}s")
    fname = f"test_{i}.jpg"
    (OUT / fname).write_bytes(img_resp.content)
    img = Image.open(OUT / fname)
    print(f"   Size: {img.size}, Mode: {img.mode}")
print("Done")
