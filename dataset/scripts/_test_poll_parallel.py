"""Quick test: download 20 Pollinations images with parallel workers"""
import time, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

OUT = Path("dataset/images/ai_generated")
OUT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
prompts = ["realistic portrait", "mountain landscape", "cat photorealistic", "city street night", "food photography"]

def dl(i):
    p = prompts[i % len(prompts)]
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(p)}?seed={i}&width=512&height=512"
    try:
        r = requests.get(url, headers=H, timeout=120)
        if r.status_code == 200:
            (OUT / f"poll_test_{i}.jpg").write_bytes(r.content)
            return (i, len(r.content), time.time())
        return (i, None, None)
    except Exception as e:
        return (i, str(e), None)

t0 = time.time()
with ThreadPoolExecutor(max_workers=5) as ex:
    for f in as_completed({ex.submit(dl, i): i for i in range(20)}):
        i, result, t = f.result()
        if result and isinstance(result, int):
            print(f"{i}: {result/1024:.0f}KB at {t-t0:.1f}s")
        else:
            print(f"{i}: FAILED {result}")
