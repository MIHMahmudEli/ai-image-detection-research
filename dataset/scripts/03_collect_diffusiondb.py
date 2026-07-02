"""
Step 3: Bulk up AI-Generated images from DiffusionDB (HuggingFace).
Downloads from the free DiffusionDB dataset.
Run: python dataset/scripts/03_collect_diffusiondb.py
"""
import json, time
from pathlib import Path
import requests

OUTPUT_DIR = Path("dataset/images/ai_generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_LOG = Path("dataset/metadata/diffusiondb_log.json")

# DiffusionDB has 14M images on HuggingFace.
# We'll download from the "first_2k" subset for speed.
# In production, use the full dataset: "polinaeterna/diffusiondb"

print("Downloading DiffusionDB (first_2k subset) from HuggingFace...")
print("This requires the 'datasets' library. Installing if needed...")
import subprocess
subprocess.run(["pip", "install", "datasets", "-q"], check=True)

from datasets import load_dataset

ds = load_dataset("polinaeterna/diffusiondb", "first_2k", split="train", trust_remote_code=True)
print(f"Loaded {len(ds)} samples from DiffusionDB")

saved = 0
metadata = []
for i, example in enumerate(ds):
    img = example["image"]
    prompt = example.get("prompt", "")
    seed = example.get("seed", "")
    fname = f"diffusiondb_{i:05d}.jpg"
    
    # Convert to RGB and save as JPEG
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(OUTPUT_DIR / fname, quality=95)
    
    metadata.append({
        "filename": f"dataset/images/ai_generated/{fname}",
        "source": "diffusiondb",
        "prompt": prompt,
        "seed": seed,
    })
    saved += 1
    
    if (i + 1) % 200 == 0:
        print(f"  Downloaded {i+1}/{len(ds)}...")

# Save metadata log
Path("dataset/metadata/diffusiondb_metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

print(f"\nDone! Downloaded {saved} images from DiffusionDB.")
print(f"Metadata saved to dataset/metadata/diffusiondb_metadata.json")
