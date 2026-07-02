"""
Step 3b: Download from Pollinations.ai in parallel with proper headers.
"""
import time, json, hashlib, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

OUTPUT_DIR = Path("dataset/images/ai_generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

prompts = [
    "realistic portrait photo professional lighting",
    "mountain landscape at sunset photography",
    "cat sitting on windowsill photorealistic",
    "busy city street at night realistic",
    "plate of gourmet food on wooden table",
    "forest path with sun rays photorealistic",
    "modern living room interior photography",
    "flower with dew drops macro photography",
    "person walking on beach at sunrise",
    "old vintage car on cobblestone street",
    "aerial view tropical island clear water",
    "street photography market small town",
    "baby laughing candid moment natural light",
    "snow capped mountains reflecting in lake",
    "coffee cup on rustic table morning light",
    "garden colorful flowers spring bloom",
    "black and white portrait elderly person",
    "night sky stars over desert landscape",
    "fresh bakery items on counter warm light",
    "dog running in park action shot",
]

existing = len([f for f in OUTPUT_DIR.iterdir() if f.name.startswith("pollinations_")])
needed = 2000 - existing
print(f"Already have {existing} Pollinations images, need {needed} more")

def download(i):
    prompt = prompts[i % len(prompts)]
    seed = abs(hash(f"p{i}")) % 1000000
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?seed={seed}&width=512&height=512"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=120)
        if resp.status_code != 200:
            return None
        fname = f"pollinations_{i:05d}.jpg"
        (OUTPUT_DIR / fname).write_bytes(resp.content)
        return True
    except:
        return None

done = 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=5) as ex:
    futures = {ex.submit(download, i): i for i in range(existing, 2000)}
    for f in as_completed(futures):
        if f.result():
            done += 1
        if done % 50 == 0 and done > 0:
            print(f"  {done}/{needed} ({done/(time.time()-t0):.1f} img/s)")

print(f"Downloaded {done} new images in {time.time()-t0:.0f}s")
print(f"Total: {len([f for f in OUTPUT_DIR.iterdir() if f.name.startswith('pollinations_')])}")
