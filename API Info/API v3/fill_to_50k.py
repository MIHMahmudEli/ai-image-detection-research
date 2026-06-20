"""Fast fill to 50K using PIL style transfer (fastest method)."""
import random, io, hashlib
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np

v3 = Path(__file__).parent
real_dir = v3 / "ai_dataset_50k" / "real_images"
altered_dir = v3 / "ai_dataset_50k" / "ai_altered_images"
altered_dir.mkdir(parents=True, exist_ok=True)

real_files = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
current = len(list(altered_dir.glob("*")))
real_count = len(real_files)
ai_count = len(list((v3 / "ai_dataset_50k" / "ai_generated_images").glob("*")))
need = 50000 - (real_count + ai_count + current)
print(f"Real: {real_count}  AI: {ai_count}  Altered: {current}  Need: {need}")

def style_transfer(img):
    styles = [
        lambda x: x.filter(ImageFilter.EDGE_ENHANCE_MORE),
        lambda x: ImageEnhance.Color(x).enhance(0.5),
        lambda x: ImageEnhance.Contrast(x).enhance(1.5),
        lambda x: ImageEnhance.Sharpness(x).enhance(2.0),
        lambda x: x.filter(ImageFilter.MedianFilter(3)),
        lambda x: x.filter(ImageFilter.GaussianBlur(1.5)),
        lambda x: x.filter(ImageFilter.UnsharpMask(2, 150)),
        lambda x: ImageOps.posterize(x, 4),
        lambda x: Image.fromarray(np.uint8(np.array(x)//96*96 + 48)),
        lambda x: x.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.SHARPEN),
        lambda x: ImageEnhance.Brightness(x).enhance(0.7),
        lambda x: ImageEnhance.Brightness(x).enhance(1.3),
        lambda x: x if random.random() < 0.5 else x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: ImageEnhance.Color(x).enhance(1.4),
        lambda x: x.filter(ImageFilter.Kernel((3,3), [0,-1,0,-1,5,-1,0,-1,0], 1, 0)),
    ]
    return random.choice(styles)(img)

batch = 0
while current < 50000 - (real_count + ai_count):
    batch += 1
    # Process 2000 images per batch then print status
    for i in range(2000):
        if current >= 50000 - (real_count + ai_count):
            break
        idx = (batch * 2000 + i) % len(real_files)
        img = Image.open(real_files[idx]).convert("RGB")
        styled = style_transfer(img)
        fid = f"FILL_{batch}_{i}"
        styled.save(altered_dir / f"{fid}.jpg", quality=88)
        current += 1
    
    real_count = len(real_files)
    ai_count = len(list((v3 / "ai_dataset_50k" / "ai_generated_images").glob("*")))
    current = len(list(altered_dir.glob("*")))
    total = real_count + ai_count + current
    print(f"Batch {batch}: Altered={current}  Total={total}/50000")

print(f"DONE. Total: {real_count + ai_count + len(list(altered_dir.glob('*')))}/50000")
