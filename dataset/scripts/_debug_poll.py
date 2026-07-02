"""Debug Pollinations API - try 10 sequential downloads with error reporting"""
import requests, time, urllib.parse
from pathlib import Path

OUT = Path("dataset/images/ai_generated")
OUT.mkdir(parents=True, exist_ok=True)

prompts = [
    "realistic portrait photo professional lighting",
    "mountain landscape at sunset photography",
    "cat sitting on windowsill photorealistic",
]

for i in range(10):
    prompt = prompts[i % len(prompts)]
    seed = abs(hash(f"p{i}")) % 1000000
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?seed={seed}&width=512&height=512"
    try:
        t0 = time.time()
        resp = requests.get(url, timeout=120)
        dt = time.time() - t0
        print(f"{i}: Status={resp.status_code}, Size={len(resp.content)/1024:.0f}KB, Time={dt:.1f}s")
        if resp.status_code == 200:
            fname = f"poll_debug_{i}.jpg"
            (OUT / fname).write_bytes(resp.content)
    except Exception as e:
        print(f"{i}: ERROR: {e}")
