"""
Download more CivitAI images. Prints progress every 50 images.
Safe to stop/resume — skips already-downloaded files.
"""
import time, json, hashlib
from pathlib import Path
import requests

OUT = Path("dataset/images/ai_generated")
OUT.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Get existing CivitAI IDs to skip duplicates
existing_ids = set()
for f in OUT.iterdir():
    if f.name.startswith("civitai_"):
        try:
            fid = f.name.replace("civitai_", "").replace(".jpg", "")
            existing_ids.add(fid)
        except:
            pass

print(f"Already have {len(existing_ids)} CivitAI images")
target = 7000  # get up to 7K total CivitAI
needed = target - len(existing_ids)
print(f"Need {needed} more")

if needed <= 0:
    print("Already at or above target")
    exit(0)

downloaded = 0
page = 1
errors = 0
t0 = time.time()

while downloaded < needed and errors < 20:
    url = f"https://civitai.com/api/v1/images?limit=100&page={page}&sort=Newest&nsfw=false"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            errors += 1
            time.sleep(2)
            continue
        items = resp.json().get("items", [])
        if not items:
            break
        
        page += 1
        for item in items:
            if downloaded >= needed:
                break
            img_id = str(item.get("id", ""))
            if img_id in existing_ids:
                continue
            
            img_url = item.get("url", "")
            if not img_url:
                continue
            
            try:
                img_resp = requests.get(img_url, headers=HEADERS, timeout=15)
                if img_resp.status_code != 200:
                    continue
                
                fname = f"civitai_{img_id}.jpg"
                (OUT / fname).write_bytes(img_resp.content)
                existing_ids.add(img_id)
                downloaded += 1
                
                if downloaded % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {downloaded}/{needed} ({downloaded/elapsed:.1f} img/s)")
            except:
                continue
        
        time.sleep(0.3)
        errors = 0
    except Exception as e:
        errors += 1
        print(f"  Error: {e}, retrying...")
        time.sleep(3)

elapsed = time.time() - t0
print(f"\nDone: {downloaded} new images in {elapsed:.0f}s")
total = len([f for f in OUT.iterdir() if f.name.startswith("civitai_")])
print(f"Total CivitAI: {total}")
