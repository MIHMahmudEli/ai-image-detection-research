"""
Step 3a: Download CivitAI images (small batch to test)
Downloads 500 images from CivitAI as a test.
"""
import time, json, hashlib
from pathlib import Path
import requests

OUTPUT_DIR = Path("dataset/images/ai_generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = Path("dataset/metadata/civitai_batch1.json")

metadata = []
downloaded = 0
target = 500

for page in range(1, 20):
    if downloaded >= target:
        break
    
    url = f"https://civitai.com/api/v1/images?limit=100&page={page}&sort=Newest&nsfw=false"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            break
        items = resp.json().get("items", [])
        if not items:
            break
        
        for item in items:
            if downloaded >= target:
                break
            img_url = item.get("url", "")
            if not img_url:
                continue
            
            try:
                img_resp = requests.get(img_url, timeout=10)
                if img_resp.status_code != 200:
                    continue
                
                img_id = item.get("id", downloaded)
                fname = f"civitai_{img_id}.jpg"
                img_data = img_resp.content
                (OUTPUT_DIR / fname).write_bytes(img_data)
                
                metadata.append({
                    "filename": f"dataset/images/ai_generated/{fname}",
                    "source": "civitai",
                    "generator": item.get("model", {}).get("name", "unknown"),
                    "md5": hashlib.md5(img_data).hexdigest(),
                })
                downloaded += 1
                if downloaded % 50 == 0:
                    print(f"  Downloaded {downloaded}/{target}")
            except:
                continue
        
        time.sleep(0.3)
    except Exception as e:
        print(f"  Page {page} error: {e}")
        time.sleep(2)

LOG_FILE.write_text(json.dumps(metadata, indent=2))
print(f"\nDone: {downloaded} images from CivitAI")
