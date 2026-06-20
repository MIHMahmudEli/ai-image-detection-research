"""
Fast dataset augmentation to reach 50K by generating AI-altered images
from existing real photos using PIL (no GPU needed).
"""

import os, sys, io, json, hashlib, random, logging
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

v3 = Path(__file__).parent
real_dir = v3 / "ai_dataset_50k" / "real_images"
altered_dir = v3 / "ai_dataset_50k" / "ai_altered_images"
altered_dir.mkdir(parents=True, exist_ok=True)

real_files = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
existing_altered = len(list(altered_dir.glob("*")))
needed_altered = max(0, 42500 - existing_altered)  # target 42.5K altered total

logger.info(f"Real images available: {len(real_files)}")
logger.info(f"Existing altered: {existing_altered}")
logger.info(f"Generating: {needed_altered} additional altered images")


def img_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    return buf.getvalue()


# --- Face swap (simple blending) ---
def generate_face_swap(base, ai_source, count):
    collected = 0
    meta = []
    pbar = tqdm(total=count, desc="FaceSwap")
    for i in range(count):
        idx = i % len(base)
        real = Image.open(base[idx]).convert("RGB")
        ai = Image.open(random.choice(ai_source)).convert("RGB")
        ai = ai.resize(real.size, Image.LANCZOS)

        w, h = real.size
        face_w, face_h = int(w*0.35), int(h*0.4)
        x0, y0 = (w-face_w)//2, (h-face_h)//3
        x1, y1 = x0+face_w, y0+face_h

        face = ai.crop((x0, y0, x1, y1)).resize((face_w, face_h))
        mask = Image.new('L', (face_w, face_h), 255)
        mask = mask.filter(ImageFilter.GaussianBlur(12))

        result = real.copy()
        result.paste(face, (x0, y0), mask)

        fid = f"FS_{i}"
        data = img_to_bytes(result)
        (altered_dir / f"{fid}.jpg").write_bytes(data)
        meta.append({"image_id": fid, "image_type": "ai_altered", "method": "face_swap"})
        collected += 1
        pbar.update(1)
    pbar.close()
    return meta


# --- Inpainting (simple neighbor average) ---
def generate_inpainting(base, count):
    collected = 0
    meta = []
    pbar = tqdm(total=count, desc="Inpainting")
    for i in range(count):
        idx = i % len(base)
        img = Image.open(base[idx]).convert("RGB")
        w, h = img.size
        arr = np.array(img, dtype=np.float32)

        mask = np.zeros((h, w), dtype=np.float32)
        num_r = random.randint(1, 3)
        for _ in range(num_r):
            x0 = random.randint(0, w//3)
            y0 = random.randint(0, h//3)
            x1 = random.randint(x0+w//6, min(w, x0+w//2))
            y1 = random.randint(y0+h//6, min(h, y0+h//2))
            mask[y0:y1, x0:x1] = 1.0

        for c in range(3):
            channel = arr[:,:,c]
            masked = mask > 0.5
            smoothed = np.zeros_like(channel)
            from scipy.ndimage import uniform_filter
            smoothed = uniform_filter(channel, size=15)
            channel[masked] = smoothed[masked]
            arr[:,:,c] = channel

        result = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))

        iid = f"IN_{i}"
        data = img_to_bytes(result)
        (altered_dir / f"{iid}.jpg").write_bytes(data)
        meta.append({"image_id": iid, "image_type": "ai_altered", "method": "inpainting"})
        collected += 1
        pbar.update(1)
    pbar.close()
    return meta


# --- Style transfer ---
def generate_style(base, count):
    styles = [
        ("oil", lambda img: img.filter(ImageFilter.MedianFilter(5))),
        ("sketch", lambda img: img.filter(ImageFilter.EDGE_ENHANCE)),
        ("cartoon", lambda img: img.filter(ImageFilter.SMOOTH_MORE)),
        ("vintage", lambda img: ImageEnhance.Color(img).enhance(0.6)),
        ("contrast", lambda img: ImageEnhance.Contrast(img).enhance(1.6)),
        ("blur", lambda img: img.filter(ImageFilter.GaussianBlur(3))),
        ("sharpen", lambda img: img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))),
        ("posterize", lambda img: Image.fromarray(np.uint8(np.array(img) // 64 * 64))),
    ]
    collected = 0
    meta = []
    pbar = tqdm(total=count, desc="StyleTransfer")
    for i in range(count):
        idx = i % len(base)
        img = Image.open(base[idx]).convert("RGB")
        sname, sfn = random.choice(styles)
        styled = sfn(img)

        sid = f"ST_{i}_{sname}"
        data = img_to_bytes(styled)
        (altered_dir / f"{sid}.jpg").write_bytes(data)
        meta.append({"image_id": sid, "image_type": "ai_altered", "method": "style_transfer"})
        collected += 1
        pbar.update(1)
    pbar.close()
    return meta


# Main
total_meta = []
ai_source = list((v3 / "ai_generated_dataset" / "images").glob("*.jpg")) + \
            list((v3 / "ai_altered_dataset" / "images").glob("*.jpg")) + \
            list(altered_dir.glob("*.jpg"))

logger.info(f"AI source images for face swap: {len(ai_source)}")

total_meta += generate_face_swap(real_files, ai_source, min(8000, needed_altered))
remaining = needed_altered - len(total_meta)
if remaining > 0:
    total_meta += generate_inpainting(real_files, min(5000, remaining))
remaining = needed_altered - len(total_meta)
if remaining > 0:
    total_meta += generate_style(real_files, min(4000, remaining))

# Final count
final_count = len(list(altered_dir.glob("*")))
logger.info(f"\nAltered images now: {final_count}")

# Re-check totals
real_count = len(real_files)
ai_count = len(list((v3 / "ai_dataset_50k" / "ai_generated_images").glob("*")))
total = real_count + ai_count + final_count
logger.info(f"Real: {real_count}  AI: {ai_count}  Altered: {final_count}  TOTAL: {total} / 50000")
