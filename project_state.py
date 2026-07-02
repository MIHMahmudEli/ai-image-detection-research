import os, sys, shutil, datetime
from pathlib import Path
from PIL import Image

print("=== DATASET SUMMARY ===")
paths = {
    "Real": "dataset/images/real",
    "BigGAN": "dataset/images/genimage_ai/BigGAN",
    "glide": "dataset/images/genimage_ai/glide",
    "ADM": "dataset/images/genimage_ai/ADM",
}
for name, p in paths.items():
    d = Path(p)
    if d.exists():
        imgs = [f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        total_bytes = sum(f.stat().st_size for f in imgs)
        if imgs:
            sample_img = Image.open(imgs[0])
            print(f"  {name:8s}: {len(imgs):>6} images, {total_bytes/1e9:.2f} GB, {sample_img.size}")
            sample_img.close()
        else:
            print(f"  {name:8s}: 0 images")
    else:
        print(f"  {name:8s}: directory not found")

print()
print("=== DISK SPACE ===")
for drive in ["D:", "E:"]:
    try:
        usage = shutil.disk_usage(f"{drive}\\")
        print(f"  {drive}: {usage.free/1e9:.1f} GB free / {usage.total/1e9:.1f} GB total")
    except:
        pass

print()
print("=== PARQUET CACHE ===")
for d in ["genimage_zips/hf_shards/BigGAN", "genimage_zips/hf_shards/glide", "genimage_zips/hf_shards/ADM", "genimage_zips"]:
    p = Path(d)
    if p.exists():
        files = list(p.rglob("*"))
        total = sum(f.stat().st_size for f in files if f.is_file())
        print(f"  {d}: {len(files)} files, {total/1e9:.2f} GB")

print()
print("=== GOOGLE DRIVE QUOTA ===")
z01 = Path("genimage_zips/imagenet_ai_0419_biggan.z01")
if z01.exists():
    created = datetime.datetime.fromtimestamp(z01.stat().st_ctime)
    now = datetime.datetime.now()
    elapsed = now - created
    remaining = datetime.timedelta(hours=24) - elapsed
    print(f"  .z01 downloaded at: {created}")
    print(f"  Elapsed: {elapsed.total_seconds()/3600:.1f}h")
    print(f"  Quota reset in: ~{remaining.total_seconds()/3600:.1f}h" if remaining.total_seconds() > 0 else "  Quota should be reset!")

print()
sys.path.insert(0, "model/src")
try:
    from model import MFFT
    m = MFFT()
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"=== MODEL ===")
    print(f"  MFFT: {total:,} total params, {trainable:,} trainable")
except Exception as e:
    print(f"=== MODEL ERROR === {e}")
