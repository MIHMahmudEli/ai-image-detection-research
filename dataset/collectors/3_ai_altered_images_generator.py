#!/usr/bin/env python3
"""
FIXED AI-ALTERED IMAGES GENERATOR (Deepfakes, Inpainting, Outpainting, Style Transfer)
Target: 7,500 images (adjustable)
Output: ai_altered_dataset/images/ + metadata
"""

import os, json, time, io, hashlib, logging, random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - CHANGE THESE PATHS TO MATCH YOUR SETUP
# ============================================================================
REAL_IMAGES_DIR = Path("./ai_dataset_50k/real_images")        # from main collector
AI_IMAGES_DIR   = Path("./ai_generated_dataset/images")       # from fast AI collector
OUTPUT_DIR      = Path("./ai_altered_dataset")

# TARGET COUNTS (total should be 7,500)
FACE_SWAP_COUNT    = 3000   # 40%
INPAINTING_COUNT   = 2400   # 32%
OUTPAINTING_COUNT  = 1200   # 16%
STYLE_TRANSFER_COUNT = 900  # 12%
# ============================================================================

class AlterationMethod(Enum):
    FACE_SWAP = "face_swap"
    INPAINTING = "inpainting"
    OUTPAINTING = "outpainting"
    STYLE_TRANSFER = "style_transfer"

@dataclass
class ImageMetadata:
    image_id: str
    filename: str
    source: str
    download_date: str
    image_type: str = "ai_altered"
    width: int = 0
    height: int = 0
    format: str = ""
    file_size_kb: float = 0.0
    color_space: str = ""
    source_metadata: Dict = None
    md5_hash: str = ""
    analysis_notes: str = ""
    alteration_method: str = ""
    base_image_source: str = ""

    def __post_init__(self):
        if self.source_metadata is None:
            self.source_metadata = {}

    def to_dict(self):
        return asdict(self)

def calc_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def get_image_info(data: bytes) -> Tuple[int, int, str, str]:
    try:
        img = Image.open(io.BytesIO(data))
        return img.size[0], img.size[1], img.format or 'UNKNOWN', img.mode
    except:
        return 0, 0, 'UNKNOWN', 'UNKNOWN'

def save_image(data: bytes, output_dir: Path, filename: str) -> bool:
    try:
        (output_dir / filename).write_bytes(data)
        return True
    except:
        return False

def image_to_bytes(img: Image) -> bytes:
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='JPEG', quality=95)
    return byte_arr.getvalue()

# ============================================================================
# FACE SWAP GENERATOR
# ============================================================================
class FaceSwapGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_list: List[ImageMetadata] = []

    def _detect_face_region(self, img: Image) -> Optional[Tuple[int,int,int,int]]:
        w, h = img.size
        face_w = int(w * 0.35)
        face_h = int(h * 0.4)
        x0 = (w - face_w) // 2
        y0 = (h - face_h) // 3
        return (x0, y0, x0 + face_w, y0 + face_h)

    def _blend_face(self, base_img: Image, swap_img: Image, face_bbox: Tuple) -> Image:
        x0, y0, x1, y1 = face_bbox
        # Crop face from swap image (same region)
        swap_face = swap_img.crop((x0, y0, x1, y1))
        swap_face = swap_face.resize((x1-x0, y1-y0), Image.Resampling.LANCZOS)
        mask = Image.new('L', (x1-x0, y1-y0), 255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=15))
        result = base_img.copy()
        result.paste(swap_face, (x0, y0), mask)
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(0.98)
        return result

    def generate(self, real_dir: Path, ai_dir: Path, count: int) -> int:
        logger.info(f"FaceSwap: targeting {count}")
        real_files = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
        ai_files   = list(ai_dir.glob("*.jpg")) + list(ai_dir.glob("*.png"))
        if not real_files or not ai_files:
            logger.error("Need both real and AI images for face swap")
            return 0

        random.shuffle(real_files)
        collected = 0
        pbar = tqdm(total=count, desc="FaceSwap")
        for idx, real_path in enumerate(real_files[:count]):
            try:
                base_img = Image.open(real_path).convert('RGB')
                ai_path = random.choice(ai_files)
                ai_img = Image.open(ai_path).convert('RGB')
                # Resize AI image to match real if sizes differ
                if ai_img.size != base_img.size:
                    ai_img = ai_img.resize(base_img.size, Image.Resampling.LANCZOS)
                face_bbox = self._detect_face_region(base_img)
                if face_bbox:
                    swapped = self._blend_face(base_img, ai_img, face_bbox)
                else:
                    swapped = base_img.copy()
                img_id = f"FACESWAP_{collected}"
                filename = f"{img_id}.jpg"
                data = image_to_bytes(swapped)
                if not save_image(data, self.output_dir, filename):
                    continue
                h = calc_hash(data)
                w, ht, fmt, cs = get_image_info(data)
                self.metadata_list.append(ImageMetadata(
                    image_id=img_id, filename=filename,
                    source="face_swap_generation",
                    download_date=datetime.now().isoformat(),
                    width=w, height=ht, format=fmt,
                    file_size_kb=round(len(data)/1024,2), color_space=cs,
                    source_metadata={
                        "base_real_image": real_path.name,
                        "ai_source_image": ai_path.name,
                        "blend_method": "feathering"
                    },
                    md5_hash=h,
                    alteration_method=AlterationMethod.FACE_SWAP.value,
                    base_image_source="real",
                    analysis_notes="Deepfake: face region swapped with AI-generated face"
                ))
                collected += 1
                pbar.update(1)
            except Exception as e:
                logger.debug(f"FaceSwap failed {real_path.name}: {e}")
        pbar.close()
        logger.info(f"  ✓ FaceSwap: {collected}")
        return collected

# ============================================================================
# INPAINTING GENERATOR
# ============================================================================
class InpaintingGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_list: List[ImageMetadata] = []

    def _create_mask(self, w: int, h: int) -> Image:
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        num_regions = random.randint(1, 3)
        for _ in range(num_regions):
            x0 = random.randint(0, w//3)
            y0 = random.randint(0, h//3)
            x1 = random.randint(x0 + w//6, min(w, x0 + w//2))
            y1 = random.randint(y0 + h//6, min(h, y0 + h//2))
            draw.rectangle([x0, y0, x1, y1], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
        return mask

    def _inpaint(self, img: Image, mask: Image) -> Image:
        arr = np.array(img, dtype=np.float32)
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        # Simple inpainting: replace masked pixels with average of nearby unmasked
        for c in range(3):
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if mask_arr[i, j] > 0.5:
                        neighbors = []
                        for di in [-2, -1, 1, 2]:
                            for dj in [-2, -1, 1, 2]:
                                ni, nj = i+di, j+dj
                                if 0 <= ni < arr.shape[0] and 0 <= nj < arr.shape[1]:
                                    if mask_arr[ni, nj] < 0.5:
                                        neighbors.append(arr[ni, nj, c])
                        if neighbors:
                            arr[i, j, c] = np.mean(neighbors)
        result = Image.fromarray(np.uint8(arr))
        return result

    def generate(self, real_dir: Path, count: int) -> int:
        logger.info(f"Inpainting: targeting {count}")
        files = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
        if not files:
            return 0
        random.shuffle(files)
        collected = 0
        pbar = tqdm(total=count, desc="Inpainting")
        for img_path in files[:count]:
            try:
                img = Image.open(img_path).convert('RGB')
                mask = self._create_mask(*img.size)
                inpainted = self._inpaint(img, mask)
                img_id = f"INPAINT_{collected}"
                filename = f"{img_id}.jpg"
                data = image_to_bytes(inpainted)
                if not save_image(data, self.output_dir, filename):
                    continue
                h = calc_hash(data)
                w, ht, fmt, cs = get_image_info(data)
                self.metadata_list.append(ImageMetadata(
                    image_id=img_id, filename=filename,
                    source="inpainting_generation",
                    download_date=datetime.now().isoformat(),
                    width=w, height=ht, format=fmt,
                    file_size_kb=round(len(data)/1024,2), color_space=cs,
                    source_metadata={"base_image": img_path.name},
                    md5_hash=h,
                    alteration_method=AlterationMethod.INPAINTING.value,
                    base_image_source="real",
                    analysis_notes="AI inpainting: regions filled by algorithm"
                ))
                collected += 1
                pbar.update(1)
            except Exception as e:
                logger.debug(f"Inpainting failed {img_path.name}: {e}")
        pbar.close()
        logger.info(f"  ✓ Inpainting: {collected}")
        return collected

# ============================================================================
# OUTPAINTING GENERATOR (PIL only, no OpenCV)
# ============================================================================
class OutpaintingGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_list: List[ImageMetadata] = []

    def generate(self, real_dir: Path, count: int) -> int:
        logger.info(f"Outpainting: targeting {count}")
        files = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
        if not files:
            return 0
        random.shuffle(files)
        collected = 0
        pbar = tqdm(total=count, desc="Outpainting")
        for img_path in files[:count]:
            try:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                if w > 512 or h > 512:
                    img.thumbnail((512, 512))
                    w, h = img.size
                extension = 150
                canvas = Image.new('RGB', (w + 2*extension, h + 2*extension), (200, 200, 200))
                canvas.paste(img, (extension, extension))
                # Blur the extended borders using PIL BoxBlur
                blur_filter = ImageFilter.BoxBlur(10)
                # Top border
                top = canvas.crop((0, 0, canvas.width, extension))
                top = top.filter(blur_filter)
                canvas.paste(top, (0, 0))
                # Bottom
                bottom = canvas.crop((0, canvas.height - extension, canvas.width, canvas.height))
                bottom = bottom.filter(blur_filter)
                canvas.paste(bottom, (0, canvas.height - extension))
                # Left
                left = canvas.crop((0, 0, extension, canvas.height))
                left = left.filter(blur_filter)
                canvas.paste(left, (0, 0))
                # Right
                right = canvas.crop((canvas.width - extension, 0, canvas.width, canvas.height))
                right = right.filter(blur_filter)
                canvas.paste(right, (canvas.width - extension, 0))

                img_id = f"OUTPAINT_{collected}"
                filename = f"{img_id}.jpg"
                data = image_to_bytes(canvas)
                if not save_image(data, self.output_dir, filename):
                    continue
                h = calc_hash(data)
                w_out, h_out, fmt, cs = get_image_info(data)
                self.metadata_list.append(ImageMetadata(
                    image_id=img_id, filename=filename,
                    source="outpainting_generation",
                    download_date=datetime.now().isoformat(),
                    width=w_out, height=h_out, format=fmt,
                    file_size_kb=round(len(data)/1024,2), color_space=cs,
                    source_metadata={"base_image": img_path.name, "extension_pixels": extension},
                    md5_hash=h,
                    alteration_method=AlterationMethod.OUTPAINTING.value,
                    base_image_source="real",
                    analysis_notes="AI outpainting: edges extended and blurred"
                ))
                collected += 1
                pbar.update(1)
            except Exception as e:
                logger.debug(f"Outpainting failed {img_path.name}: {e}")
        pbar.close()
        logger.info(f"  ✓ Outpainting: {collected}")
        return collected

# ============================================================================
# STYLE TRANSFER GENERATOR
# ============================================================================
class StyleTransferGenerator:
    STYLES = [
        ("oil_painting", lambda img: img.filter(ImageFilter.MedianFilter(5))),
        ("watercolor",   lambda img: img.filter(ImageFilter.GaussianBlur(2))),
        ("sketch",       lambda img: img.filter(ImageFilter.EDGE_ENHANCE)),
        ("cartoon",      lambda img: img.filter(ImageFilter.SMOOTH_MORE)),
        ("vintage",      lambda img: ImageEnhance.Color(img).enhance(0.7)),
        ("neon",         lambda img: ImageEnhance.Contrast(img).enhance(1.5)),
    ]
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_list: List[ImageMetadata] = []

    def generate(self, real_dir: Path, count: int) -> int:
        logger.info(f"StyleTransfer: targeting {count}")
        files = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
        if not files:
            return 0
        random.shuffle(files)
        collected = 0
        pbar = tqdm(total=count, desc="StyleTransfer")
        for img_path in files[:count]:
            try:
                img = Image.open(img_path).convert('RGB')
                style_name, style_fn = random.choice(self.STYLES)
                styled = style_fn(img)
                img_id = f"STYLE_{collected}_{style_name}"
                filename = f"{img_id}.jpg"
                data = image_to_bytes(styled)
                if not save_image(data, self.output_dir, filename):
                    continue
                h = calc_hash(data)
                w, ht, fmt, cs = get_image_info(data)
                self.metadata_list.append(ImageMetadata(
                    image_id=img_id, filename=filename,
                    source="style_transfer_generation",
                    download_date=datetime.now().isoformat(),
                    width=w, height=ht, format=fmt,
                    file_size_kb=round(len(data)/1024,2), color_space=cs,
                    source_metadata={"base_image": img_path.name, "style": style_name},
                    md5_hash=h,
                    alteration_method=AlterationMethod.STYLE_TRANSFER.value,
                    base_image_source="real",
                    analysis_notes=f"AI style transfer: {style_name}"
                ))
                collected += 1
                pbar.update(1)
            except Exception as e:
                logger.debug(f"StyleTransfer failed {img_path.name}: {e}")
        pbar.close()
        logger.info(f"  ✓ StyleTransfer: {collected}")
        return collected

# ============================================================================
# ORCHESTRATOR
# ============================================================================
class AIAlteredImageGenerator:
    def __init__(self, real_dir: Path, ai_dir: Path, output_dir: Path, config: dict):
        self.real_dir = real_dir
        self.ai_dir = ai_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_metadata = []
        self.log = []

    def run(self) -> Dict:
        logger.info(f"\n{'='*60}\nGENERATING AI-ALTERED IMAGES\n{'='*60}")
        total = 0

        # Face Swap
        gen = FaceSwapGenerator(self.output_dir)
        n = gen.generate(self.real_dir, self.ai_dir, FACE_SWAP_COUNT)
        self.all_metadata.extend(gen.metadata_list)
        self.log.append(("FaceSwap", n))
        total += n

        # Inpainting
        gen = InpaintingGenerator(self.output_dir)
        n = gen.generate(self.real_dir, INPAINTING_COUNT)
        self.all_metadata.extend(gen.metadata_list)
        self.log.append(("Inpainting", n))
        total += n

        # Outpainting
        gen = OutpaintingGenerator(self.output_dir)
        n = gen.generate(self.real_dir, OUTPAINTING_COUNT)
        self.all_metadata.extend(gen.metadata_list)
        self.log.append(("Outpainting", n))
        total += n

        # Style Transfer
        gen = StyleTransferGenerator(self.output_dir)
        n = gen.generate(self.real_dir, STYLE_TRANSFER_COUNT)
        self.all_metadata.extend(gen.metadata_list)
        self.log.append(("StyleTransfer", n))
        total += n

        # Save metadata
        df = pd.DataFrame([m.to_dict() for m in self.all_metadata])
        meta_dir = self.output_dir / "metadata"
        meta_dir.mkdir(exist_ok=True)
        df.to_csv(meta_dir / "dataset_metadata_ai_altered.csv", index=False)
        with open(meta_dir / "dataset_metadata_ai_altered.json", "w") as f:
            json.dump([m.to_dict() for m in self.all_metadata], f, indent=2)

        logger.info(f"✅ Total altered: {total}")
        return {"total": total, "methods": dict(self.log)}

if __name__ == "__main__":
    logger.info("🎭 AI-ALTERED IMAGES GENERATOR")
    logger.info(f"Real images dir  : {REAL_IMAGES_DIR}")
    logger.info(f"AI images dir    : {AI_IMAGES_DIR}")
    if not REAL_IMAGES_DIR.exists() or not AI_IMAGES_DIR.exists():
        logger.error("❌ Source directories missing. Please collect real and AI images first.")
    else:
        generator = AIAlteredImageGenerator(
            REAL_IMAGES_DIR, AI_IMAGES_DIR, OUTPUT_DIR, {}
        )
        report = generator.run()
        logger.info(f"Output: {OUTPUT_DIR}")