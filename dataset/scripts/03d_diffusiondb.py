"""
Download DiffusionDB large_random_1k subset (1000 images).
Uses HuggingFace datasets library - Method 1.
"""
import time, hashlib
from pathlib import Path
from datasets import load_dataset

OUT = Path("dataset/images/ai_generated")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading DiffusionDB large_random_1k...")
t0 = time.time()

dataset = load_dataset("poloclub/diffusiondb", "large_random_1k", split="train", trust_remote_code=True)

print(f"Loaded {len(dataset)} samples in {time.time()-t0:.0f}s")
print(f"Features: {dataset.features}")

saved = 0
for i, example in enumerate(dataset):
    img = example["image"]
    prompt = example.get("prompt", "")
    
    fname = f"diffusiondb_{i:05d}.jpg"
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(OUT / fname, quality=95)
    saved += 1
    
    if (i + 1) % 200 == 0:
        print(f"  Saved {i+1}/{len(dataset)}")

elapsed = time.time() - t0
print(f"\nDone! Saved {saved} images in {elapsed:.0f}s ({saved/elapsed:.1f} img/s)")
