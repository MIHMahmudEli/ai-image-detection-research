"""Debug: show actual errors from parallel Pollinations requests"""
import time, urllib.parse, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

OUT = Path("dataset/images/ai_generated")
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def dl(i):
    url = f"https://image.pollinations.ai/prompt/test%20image?seed={i}&width=512&height=512"
    try:
        r = requests.get(url, headers=H, timeout=30)
        if r.status_code != 200:
            return (i, f"HTTP {r.status_code}: {r.text[:100]}")
        (OUT / f"poll_debug2_{i}.jpg").write_bytes(r.content)
        return (i, f"OK {len(r.content)} bytes")
    except Exception as e:
        return (i, f"{type(e).__name__}: {e}")

with ThreadPoolExecutor(max_workers=3) as ex:
    for f in as_completed({ex.submit(dl, i): i for i in range(10)}):
        i, msg = f.result()
        print(f"{i}: {msg}")
