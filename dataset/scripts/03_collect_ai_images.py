"""
Step 3: Collect more AI-generated images (parallel downloads).
Uses CivitAI API with 10 parallel workers for speed.
"""
import time, json, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

OUTPUT_DIR = Path("dataset/images/ai_generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_FILE = Path("dataset/metadata/ai_generated_new.json")

def download_image(item):
    img_url = item.get("url", "")
    if not img_url:
        return None
    try:
        img_resp = requests.get(img_url, timeout=30)
        if img_resp.status_code != 200:
            return None
        img_id = item.get("id", hash(img_url) % 1000000)
        fname = f"civitai_{img_id}.jpg"
        img_data = img_resp.content
        (OUTPUT_DIR / fname).write_bytes(img_data)
        return {
            "filename": f"dataset/images/ai_generated/{fname}",
            "source": "civitai",
            "generator": item.get("model", {}).get("name", "unknown"),
            "md5": hashlib.md5(img_data).hexdigest(),
        }
    except:
        return None

# Collect image URLs from CivitAI API
all_items = []
target = 2000
page = 1
while len(all_items) < target:
    url = f"https://civitai.com/api/v1/images?limit=100&page={page}&sort=Newest&nsfw=false"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            break
        items = resp.json().get("items", [])
        if not items:
            break
        all_items.extend(items)
        print(f"  Fetched page {page}: {len(items)} items (total: {len(all_items)})")
        page += 1
        time.sleep(0.3)
    except Exception as e:
        print(f"  Error on page {page}: {e}")
        break

all_items = all_items[:target]
print(f"\nDownloading {len(all_items)} images with 10 parallel workers...")

metadata = []
done = 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(download_image, item): item for item in all_items}
    for f in as_completed(futures):
        result = f.result()
        if result:
            metadata.append(result)
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                print(f"  Downloaded {done}/{target} ({rate:.1f} img/s)")

elapsed = time.time() - t0
METADATA_FILE.write_text(json.dumps(metadata, indent=2))
print(f"\nDone! Downloaded {done} images in {elapsed:.0f}s ({done/elapsed:.1f} img/s)")
print(f"Metadata saved to: {METADATA_FILE}")
