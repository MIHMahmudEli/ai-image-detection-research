import requests, time, os, sys
from pathlib import Path

BASE_URL = 'https://huggingface.co/datasets/bitmind/GenImage_glide/resolve/main/data'
DL_DIR = Path('genimage_zips/hf_shards/glide')
DL_DIR.mkdir(parents=True, exist_ok=True)

# List all shards via HF API
url = 'https://huggingface.co/api/datasets/bitmind/GenImage_glide'
r = requests.get(url)
data = r.json()
siblings = data.get('siblings', [])
shards = sorted([s['rfilename'] for s in siblings if s['rfilename'].endswith('.parquet')])
print(f'Downloading {len(shards)} glide shards...')

for i, shard_name in enumerate(shards):
    filename = shard_name.split('/')[-1]
    out = DL_DIR / filename
    if out.exists():
        sz = out.stat().st_size
        print(f'[{i+1}/{len(shards)}] {filename} exists ({sz/1e6:.1f} MB), skipping')
        continue

    tmp = str(out) + '.part'
    shard_url = f'{BASE_URL}/{filename}'
    print(f'[{i+1}/{len(shards)}] Downloading {filename}...')
    t0 = time.time()
    r = requests.get(shard_url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(tmp, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    elapsed = time.time() - t0
    speed = total / elapsed / 1e6
    os.rename(tmp, out)
    print(f'  Done: {total/1e6:.1f} MB in {elapsed:.0f}s ({speed:.1f} MB/s)')

print(f'\nAll {len(shards)} shards downloaded')
