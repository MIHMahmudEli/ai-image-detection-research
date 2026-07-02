"""
Download DiffusionDB 2M part files with resume support.
Downloads in chunks so it can survive network issues.
"""
import requests, zipfile, io, time, json
from pathlib import Path

OUT = Path("dataset/images/ai_generated")
OUT.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks

# Download just 2 parts (about 2000 images, ~1.2 GB total)
parts = [1, 2]
base_url = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images"

for part_num in parts:
    url = f"{base_url}/part-{part_num:06d}.zip"
    local_zip = Path(f"dataset/scripts/part-{part_num:06d}.zip")
    
    print(f"\n{'='*60}")
    print(f"Downloading part-{part_num:06d}.zip ...")
    
    # Check if already partially downloaded
    downloaded_bytes = 0
    if local_zip.exists():
        downloaded_bytes = local_zip.stat().st_size
        print(f"  Already have {downloaded_bytes/1024/1024:.0f} MB, resuming...")
    
    headers = {}
    if downloaded_bytes > 0:
        headers["Range"] = f"bytes={downloaded_bytes}-"
    
    t0 = time.time()
    try:
        resp = requests.get(url, headers=headers, timeout=30, stream=True)
        if resp.status_code in (200, 206):
            total = int(resp.headers.get("content-length", 0)) + downloaded_bytes
            mode = "ab" if downloaded_bytes > 0 else "wb"
            
            with open(local_zip, mode) as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
            
            elapsed = time.time() - t0
            print(f"  Downloaded {downloaded_bytes/1024/1024:.0f}/{total/1024/1024:.0f} MB in {elapsed:.0f}s")
            
            # Extract images
            print(f"  Extracting images...")
            count = 0
            with zipfile.ZipFile(local_zip) as zf:
                for name in zf.namelist():
                    if name.lower().endswith((".jpg", ".jpeg", ".png")):
                        data = zf.read(name)
                        fname = f"diffusiondb_{part_num:03d}_{Path(name).name}"
                        (OUT / fname).write_bytes(data)
                        count += 1
            print(f"  Extracted {count} images")
            
            # Clean up zip
            local_zip.unlink()
            print(f"  Removed zip file")
        else:
            print(f"  HTTP {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Partial file saved at {local_zip}, will resume next time")

print(f"\nDone! Total DiffusionDB images: {len(list(OUT.glob('diffusiondb_*')))}")
print(f"Total AI-generated images: {len(list(OUT.glob('*')))}")
