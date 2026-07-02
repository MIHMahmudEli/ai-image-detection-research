"""
Download GenImage from HuggingFace (bitmind per-generator datasets).
Usage: python scripts/07_download_hf_genimage.py --generator BigGAN
"""
import argparse, sys, time, json, os, io
from pathlib import Path
import requests
import pyarrow.parquet as pq
from PIL import Image

HF_BASE = "https://huggingface.co/datasets/bitmind/GenImage_{generator}/resolve/main/data"

GENERATORS = {
    "BigGAN":      {"shards": 8, "repo": "bitmind/GenImage_BigGAN"},
    "ADM":         {"shards": 12, "repo": "bitmind/GenImage_ADM"},
    "MidJourney":  {"shards": None, "repo": "bitmind/GenImage_MidJourney"},
    "glide":       {"shards": None, "repo": "bitmind/GenImage_glide"},
    "wukong":      {"shards": None, "repo": "bitmind/GenImage_wukong"},
    "VQDM":        {"shards": None, "repo": "bitmind/GenImage_VQDM"},
    "stable_diffusion_v_1_4": {"shards": None, "repo": None},
    "stable_diffusion_v_1_5": {"shards": None, "repo": None},
}

GENIMAGE_OUT = Path("dataset/images/genimage_ai")

def fmt_bytes(b):
    for u in ["B", "KB", "MB", "GB"]:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"

def discover_shards(repo):
    """Discover shard count from HF API."""
    import requests
    url = f"https://huggingface.co/api/datasets/{repo}"
    r = requests.get(url)
    data = r.json()
    siblings = data.get("siblings", [])
    shards = [s["rfilename"] for s in siblings if s["rfilename"].startswith("data/train-") and s["rfilename"].endswith(".parquet")]
    shards.sort()
    return shards

def download_shard(url, out_path):
    """Download a single parquet shard with progress."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".part"
    if out_path.exists():
        return True, "already exists"
    
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    t0 = time.time()
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
    elapsed = time.time() - t0
    speed = downloaded / elapsed / 1e6 if elapsed > 0 else 0
    os.rename(tmp, out_path)
    return True, f"{fmt_bytes(downloaded)} in {elapsed:.0f}s ({speed:.1f} MB/s)"

def extract_shard(shard_path, out_dir, gen_name):
    """Extract all images from a parquet shard to out_dir."""
    pf = pq.ParquetFile(shard_path)
    total = pf.metadata.num_rows
    extracted = 0
    t0 = time.time()
    
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx)
        struct_arr = table.column("image").combine_chunks()
        
        for i in range(len(struct_arr)):
            row = struct_arr[i].as_py()
            img_bytes = row["bytes"]
            rel_path = row["path"]
            
            # Preserve original filename
            out_path = out_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if already exists
            if out_path.exists():
                continue
            
            try:
                img = Image.open(io.BytesIO(img_bytes))
                img.save(out_path, "PNG")
                extracted += 1
            except Exception as e:
                print(f"    Error extracting {rel_path}: {e}")
        
        if (rg_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = extracted / elapsed if elapsed > 0 else 0
            print(f"    Extracted {extracted}/{total} images ({rate:.0f} img/s)...")
    
    elapsed = time.time() - t0
    print(f"    Extracted {extracted} images from {shard_path.name} in {elapsed:.0f}s")
    return extracted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", choices=list(GENERATORS.keys()), required=True)
    parser.add_argument("--download-dir", type=str, default="genimage_zips/hf_shards")
    parser.add_argument("--extract-only", action="store_true", help="Skip download, extract only")
    args = parser.parse_args()
    
    gen = args.generator
    gen_info = GENERATORS[gen]
    repo = gen_info["repo"]
    
    if repo is None:
        print(f"No HF repo configured for {gen}. Check bitmind/GenImage_{gen}")
        # Try to auto-detect
        repo = f"bitmind/GenImage_{gen}"
    
    dl_dir = Path(args.download_dir) / gen
    dl_dir.mkdir(parents=True, exist_ok=True)
    
    gen_out = GENIMAGE_OUT / gen
    
    print(f"Generator: {gen}")
    print(f"Repo: {repo}")
    print(f"Download to: {dl_dir.resolve()}")
    print(f"Extract to: {gen_out.resolve()}")
    print()
    
    # Discover shards
    print("Discovering shards...")
    shards = discover_shards(repo)
    if not shards:
        print(f"  No shards found for {repo}! Check the repo name.")
        return
    print(f"  Found {len(shards)} shards")
    
    # Download
    if not args.extract_only:
        t_start = time.time()
        for i, shard_name in enumerate(shards):
            url = f"{HF_BASE.format(generator=gen)}/{shard_name.split('/')[-1]}"
            # Actually, the URL is direct
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{shard_name}"
            out_path = dl_dir / shard_name.split("/")[-1]
            
            if out_path.exists():
                print(f"[{i+1}/{len(shards)}] {out_path.name} — already exists ({fmt_bytes(out_path.stat().st_size)})")
                continue
            
            print(f"[{i+1}/{len(shards)}] Downloading {shard_name}...")
            ok, msg = download_shard(url, out_path)
            if ok:
                print(f"  Done: {msg}")
            else:
                print(f"  FAILED: {msg}")
                continue
        
        # Summary
        downloaded = sum(1 for s in shards if (dl_dir / s.split("/")[-1]).exists())
        elapsed = time.time() - t_start
        total_size = sum((dl_dir / s.split("/")[-1]).stat().st_size for s in shards if (dl_dir / s.split("/")[-1]).exists())
        print(f"\nDownloaded {downloaded}/{len(shards)} shards ({fmt_bytes(total_size)}) in {elapsed:.0f}s")
    
    # Extract
    available = sum(1 for s in shards if (dl_dir / s.split("/")[-1]).exists())
    if available == len(shards):
        print(f"\nExtracting {gen} to {gen_out}...")
        gen_out.mkdir(parents=True, exist_ok=True)
        total_extracted = 0
        t_extract = time.time()
        
        for i, shard_name in enumerate(shards):
            shard_path = dl_dir / shard_name.split("/")[-1]
            print(f"\n[{i+1}/{len(shards)}] Extracting {shard_path.name}...")
            n = extract_shard(shard_path, gen_out, gen)
            total_extracted += n
        
        elapsed = time.time() - t_extract
        print(f"\nExtracted {total_extracted} images total in {elapsed:.0f}s ({total_extracted/elapsed:.0f} img/s)")
        print(f"Images in: {gen_out.resolve()}")
    else:
        print(f"\nNot all shards available ({available}/{len(shards)}). Cannot extract.")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
