# =============================================================================
#  SECTION 1: IMPORTS & SETUP
# =============================================================================
import os, json, time, io, random, hashlib, logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from itertools import cycle

import requests
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
#  SECTION 2: DATA MODELS
# =============================================================================
class ImageType(Enum):
    REAL         = "real"
    AI_GENERATED = "ai_generated"
    AI_ALTERED   = "ai_altered"           # new class

class ImageSource(Enum):
    UNSPLASH        = "unsplash"
    PEXELS          = "pexels"
    PIXABAY         = "pixabay"
    CIVITAI         = "civitai"
    HUGGINGFACE     = "huggingface"
    POLLINATIONS    = "pollinations"
    LOCAL           = "local"
    AI_ALTERED_LOCAL = "ai_altered_local"  # new source

@dataclass
class ImageMetadata:
    image_id: str
    filename: str
    source: str
    source_url: str
    download_date: str
    image_type: str
    width: int
    height: int
    format: str
    file_size_kb: float
    color_space: str
    source_metadata: Dict
    md5_hash: str
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    processed: bool = False
    analysis_notes: str = ""

    def to_dict(self):
        return asdict(self)

# =============================================================================
#  SECTION 3: BASE COLLECTOR
# =============================================================================
class BaseCollector:
    def __init__(self, output_dir: Path, subfolder: str):
        self.output_dir = output_dir / subfolder
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_list: List[ImageMetadata] = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _calculate_hash(self, data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    def _get_image_info(self, data: bytes) -> Tuple[int, int, str, str]:
        try:
            img = Image.open(io.BytesIO(data))
            return img.size[0], img.size[1], img.format or 'UNKNOWN', img.mode
        except:
            return 0, 0, 'UNKNOWN', 'UNKNOWN'

    def _save_image(self, data: bytes, filename: str) -> bool:
        try:
            (self.output_dir / filename).write_bytes(data)
            return True
        except:
            return False

    def _check_duplicate(self, h: str) -> Optional[str]:
        for m in self.metadata_list:
            if m.md5_hash == h:
                return m.image_id
        return None

    def _download_and_record(self, url: str, image_id: str, image_type: str,
                              source_val: str, source_meta: dict, note: str,
                              extra_headers: dict = None) -> bool:
        try:
            hdrs = dict(self.session.headers)
            if extra_headers:
                hdrs.update(extra_headers)
            r = requests.get(url, headers=hdrs, timeout=20)
            r.raise_for_status()
            data = r.content
            if len(data) < 5000:
                return False
            content_type = r.headers.get('content-type', '')
            if 'image' not in content_type and not url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                return False
            h = self._calculate_hash(data)
            dup = self._check_duplicate(h)
            filename = f"{image_id}.jpg"
            w, ht, fmt, cs = self._get_image_info(data)
            if not self._save_image(data, filename):
                return False
            self.metadata_list.append(ImageMetadata(
                image_id=image_id, filename=filename,
                source=source_val, source_url=url,
                download_date=datetime.now().isoformat(),
                image_type=image_type, width=w, height=ht, format=fmt,
                file_size_kb=round(len(data) / 1024, 2), color_space=cs,
                source_metadata=source_meta, md5_hash=h,
                is_duplicate=dup is not None, duplicate_of=dup,
                analysis_notes=note
            ))
            return True
        except Exception as e:
            logger.debug(f"Download failed ({image_id}): {e}")
            return False

# =============================================================================
#  SECTION 4: REAL IMAGE COLLECTORS
# =============================================================================
class UnsplashCollector(BaseCollector):
    def __init__(self, api_key: str, output_dir: Path):
        super().__init__(output_dir, "real_images")
        self.api_key = api_key

    def collect(self, queries: List[str], total_count: int) -> int:
        logger.info(f"Unsplash: targeting {total_count} images")
        collected, q_idx, page = 0, 0, 1
        while collected < total_count and q_idx < len(queries):
            query = queries[q_idx]
            try:
                r = self.session.get(
                    "https://api.unsplash.com/search/photos",
                    headers={"Authorization": f"Client-ID {self.api_key}"},
                    params={"query": query, "per_page": 30, "page": page},
                    timeout=15)
                if r.status_code == 429:
                    logger.warning("Unsplash rate-limited, waiting 60s…")
                    time.sleep(60); continue
                r.raise_for_status()
                results = r.json().get('results', [])
                if not results:
                    q_idx += 1; page = 1; continue
                for res in results:
                    if collected >= total_count: break
                    url = res['urls'].get('regular') or res['urls']['full']
                    if self._download_and_record(url, f"UNSPLASH_{res['id']}", ImageType.REAL.value,
                                                ImageSource.UNSPLASH.value,
                                                {"author": res['user']['name']}, f"Query: {query}"):
                        collected += 1
                page += 1
            except Exception as e:
                logger.warning(f"Unsplash error '{query}': {e}")
                q_idx += 1; page = 1
        logger.info(f"  ✓ Unsplash: {collected}")
        return collected

class PexelsCollector(BaseCollector):
    def __init__(self, api_key: str, output_dir: Path):
        super().__init__(output_dir, "real_images")
        self.api_key = api_key

    def collect(self, queries: List[str], total_count: int) -> int:
        logger.info(f"Pexels: targeting {total_count} images")
        collected, q_idx, page = 0, 0, 1
        while collected < total_count and q_idx < len(queries):
            query = queries[q_idx]
            try:
                r = self.session.get(
                    "https://api.pexels.com/v1/search",
                    headers={"Authorization": self.api_key},
                    params={"query": query, "per_page": 80, "page": page},
                    timeout=15)
                if r.status_code == 429:
                    logger.warning("Pexels rate-limited, waiting 30s…")
                    time.sleep(30); continue
                r.raise_for_status()
                photos = r.json().get('photos', [])
                if not photos:
                    q_idx += 1; page = 1; continue
                for p in photos:
                    if collected >= total_count: break
                    url = p['src'].get('large2x') or p['src']['large']
                    if self._download_and_record(url, f"PEXELS_{p['id']}", ImageType.REAL.value,
                                                ImageSource.PEXELS.value,
                                                {"photographer": p['photographer']}, f"Query: {query}"):
                        collected += 1
                page += 1
            except Exception as e:
                logger.warning(f"Pexels error '{query}': {e}")
                q_idx += 1; page = 1
        logger.info(f"  ✓ Pexels: {collected}")
        return collected

class PixabayCollector(BaseCollector):
    def __init__(self, api_key: str, output_dir: Path):
        super().__init__(output_dir, "real_images")
        self.api_key = api_key

    def collect(self, queries: List[str], total_count: int) -> int:
        logger.info(f"Pixabay: targeting {total_count} images")
        collected, q_idx, page = 0, 0, 1
        while collected < total_count and q_idx < len(queries):
            query = queries[q_idx]
            try:
                r = self.session.get("https://pixabay.com/api/",
                                     params={"key": self.api_key, "q": query, "per_page": 200,
                                             "page": page, "image_type": "photo"}, timeout=15)
                r.raise_for_status()
                hits = r.json().get('hits', [])
                if not hits:
                    q_idx += 1; page = 1; continue
                for h in hits:
                    if collected >= total_count: break
                    url = h.get('largeImageURL', '')
                    if not url: continue
                    if self._download_and_record(url, f"PIXABAY_{h['id']}", ImageType.REAL.value,
                                                ImageSource.PIXABAY.value,
                                                {"user": h['user']}, f"Query: {query}"):
                        collected += 1
                page += 1
            except Exception as e:
                logger.warning(f"Pixabay error '{query}': {e}")
                q_idx += 1; page = 1
        logger.info(f"  ✓ Pixabay: {collected}")
        return collected

# =============================================================================
#  SECTION 5: AI‑GENERATED COLLECTORS (CivitAI, HuggingFace, Pollinations)
# =============================================================================
class CivitaiCollector(BaseCollector):
    SORT_COMBOS = [
        {"sort": "Most Reactions", "period": "AllTime"},
        {"sort": "Most Comments",  "period": "AllTime"},
        {"sort": "Newest",         "period": "AllTime"},
        {"sort": "Most Reactions", "period": "Month"},
        {"sort": "Most Reactions", "period": "Week"},
        {"sort": "Newest",         "period": "Month"},
    ]
    def __init__(self, output_dir: Path, api_key: str = None):
        super().__init__(output_dir, "ai_generated_images")
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
            logger.info("  CivitAI: using API token ✓")
        else:
            logger.warning("  CivitAI: no API token — add CIVITAI_API_KEY to .env for best results")

    def collect(self, count: int = 500) -> int:
        logger.info(f"CivitAI: targeting {count} images")
        collected = 0
        for combo in self.SORT_COMBOS:
            if collected >= count: break
            cursor, consecutive_empty = None, 0
            while collected < count:
                try:
                    params = {"limit": 100, "nsfw": "None", "sort": combo["sort"],
                              "period": combo["period"], "type": "image"}
                    if cursor: params["cursor"] = cursor
                    r = self.session.get("https://civitai.com/api/v1/images", params=params, timeout=20)
                    if r.status_code == 401:
                        logger.error("CivitAI 401: invalid or missing API token."); return collected
                    if r.status_code == 429:
                        logger.warning("CivitAI rate-limited, waiting 45s…"); time.sleep(45); continue
                    if r.status_code == 403:
                        logger.error("CivitAI 403: check API token permissions"); return collected
                    r.raise_for_status()
                    result = r.json()
                    items = result.get('items', [])
                    if not items:
                        consecutive_empty += 1
                        if consecutive_empty >= 2: break
                        continue
                    consecutive_empty = 0
                    for item in items:
                        if collected >= count: break
                        url = item.get('url', '')
                        if not url or item.get('type', 'image') != 'image': continue
                        if self._download_and_record(url, f"CIVITAI_{item.get('id', f'c{collected}')}",
                                                    ImageType.AI_GENERATED.value, ImageSource.CIVITAI.value,
                                                    {"model": str(item.get('meta', {}).get('Model', ''))[:100],
                                                     "prompt": str(item.get('meta', {}).get('prompt', ''))[:200]},
                                                    f"CivitAI {combo['sort']} / {combo['period']}"):
                            collected += 1
                    cursor = result.get('metadata', {}).get('nextCursor')
                    if not cursor:
                        logger.info(f"  CivitAI cursor exhausted for combo {combo}, switching…")
                        break
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"CivitAI error: {e}")
                    break
        logger.info(f"  ✓ CivitAI: {collected}")
        return collected

class HuggingFaceCollector(BaseCollector):
    HF_API = "https://datasets-server.huggingface.co/rows"
    CONFIGS = [
        ("2m_random_1k", "train", 1000),
        ("2m_random_5k", "train", 5000),
        ("2m_random_10k", "train", 10000),
    ]
    def __init__(self, output_dir: Path):
        super().__init__(output_dir, "ai_generated_images")

    def collect(self, count: int = 400) -> int:
        logger.info(f"HuggingFace DiffusionDB: targeting {count} images")
        collected = 0
        for config, split, max_rows in self.CONFIGS:
            if collected >= count: break
            offset, page_size = 0, 100
            while collected < count and offset < max_rows:
                try:
                    r = self.session.get(self.HF_API, params={
                        "dataset": "poloclub/diffusiondb", "config": config,
                        "split": split, "offset": offset, "length": page_size
                    }, timeout=30)
                    if r.status_code == 429:
                        logger.warning("HuggingFace rate-limited, waiting 30s…"); time.sleep(30); continue
                    if not r.ok:
                        logger.warning(f"HuggingFace {r.status_code} for config {config}"); break
                    rows = r.json().get('rows', [])
                    if not rows: break
                    for row in rows:
                        if collected >= count: break
                        row_data = row.get('row', {})
                        image_field = row_data.get('image', {})
                        img_url = ""
                        if isinstance(image_field, dict):
                            img_url = image_field.get('src') or image_field.get('url', '')
                        elif isinstance(image_field, str):
                            img_url = image_field
                        if not img_url: continue
                        prompt = str(row_data.get('prompt', ''))[:300]
                        seed = str(row_data.get('seed', ''))
                        img_id = f"HF_{config}_{offset}_{row.get('row_idx', collected)}"
                        if self._download_and_record(img_url, img_id, ImageType.AI_GENERATED.value,
                                                    ImageSource.HUGGINGFACE.value,
                                                    {"prompt": prompt, "seed": seed, "config": config},
                                                    f"DiffusionDB config={config}"):
                            collected += 1
                    offset += page_size
                    time.sleep(0.3)
                except Exception as e:
                    logger.warning(f"HuggingFace error (config={config}): {e}")
                    break
        logger.info(f"  ✓ HuggingFace: {collected}")
        return collected

class PollinationsCollector(BaseCollector):
    PROMPTS = [
        "a beautiful fantasy landscape with mountains and lakes",
        "portrait of a cyberpunk woman with neon lights",
        "futuristic city skyline at night with flying cars",
        "detailed oil painting of a medieval knight",
        "anime girl with blue hair in a magical forest",
        "abstract digital art with geometric patterns",
        "realistic portrait of an elderly man with wrinkles",
        "sci-fi spacecraft in outer space with nebula background",
        "watercolor painting of cherry blossom trees in japan",
        "a majestic dragon breathing fire over a castle",
        "hyperrealistic still life of fruits on a wooden table",
        "surreal melting clock landscape in the style of dali",
        "black and white portrait of a jazz musician playing saxophone",
        "colorful pop art illustration of a cat",
        "impressionist painting of a rainy paris street at night",
        "3d render of a robot sitting in a coffee shop",
        "fantasy map of an imaginary island kingdom",
        "close up macro photo of a butterfly on a flower",
        "dark gothic cathedral interior with stained glass windows",
        "pixel art landscape with 8bit style mountains and sunset",
        "concept art of a post apocalyptic city overgrown with plants",
        "minimalist line art portrait of a woman",
        "realistic wolf in a snowy pine forest",
        "steampunk clockwork mechanical elephant",
        "underwater scene with colorful coral reef and tropical fish",
        "aurora borealis over a snowy landscape with pine trees",
        "digital art of a phoenix rising from flames",
        "art nouveau style illustration of a woman with flowers",
        "cubist portrait in the style of picasso",
        "neon lit tokyo street at night in rain",
    ]
    def __init__(self, output_dir: Path):
        super().__init__(output_dir, "ai_generated_images")

    def collect(self, count: int = 300) -> int:
        logger.info(f"Pollinations.ai: targeting {count} images")
        collected, prompt_idx = 0, 0
        while collected < count:
            prompt = self.PROMPTS[prompt_idx % len(self.PROMPTS)]
            seed = int(time.time() * 1000) % 999999
            try:
                import urllib.parse
                encoded = urllib.parse.quote(prompt)
                url = f"https://image.pollinations.ai/prompt/{encoded}?width=768&height=768&seed={seed}&nologo=true&enhance=true"
                img_id = f"POLLINATIONS_{collected}_{seed}"
                if self._download_and_record(url, img_id, ImageType.AI_GENERATED.value,
                                            ImageSource.POLLINATIONS.value,
                                            {"prompt": prompt, "seed": seed},
                                            f"Pollinations: {prompt[:50]}"):
                    collected += 1
                prompt_idx += 1
                time.sleep(1.5)
            except Exception as e:
                logger.warning(f"Pollinations error: {e}")
                prompt_idx += 1
                time.sleep(2)
        logger.info(f"  ✓ Pollinations: {collected}")
        return collected

# =============================================================================
#  SECTION 6: AI‑ALTERED COLLECTOR (LOCAL GPU INPAINTING)
#  NOTE: Requires GPU + `pip install diffusers transformers accelerate torch`
# =============================================================================
class AIAlteredCollector(BaseCollector):
    """
    Generates AI‑altered images from a folder of real images using
    Stable Diffusion inpainting. Outputs go into ai_altered_images/.
    """
    def __init__(self, output_dir: Path, real_images_dir: Path, hf_token: str = None):
        super().__init__(output_dir, "ai_altered_images")
        self.real_images_dir = Path(real_images_dir)
        self.hf_token = hf_token

    def _load_pipeline(self):
        from diffusers import StableDiffusionInpaintPipeline
        import torch

        model_id = "stabilityai/stable-diffusion-2-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_auth_token=self.hf_token if self.hf_token else True
        )
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        return pipe

    def collect(self, count: int = 7500) -> int:
        logger.info(f"AI‑altered: targeting {count} images")
        pipe = self._load_pipeline()

        real_files = list(self.real_images_dir.glob("*"))
        if not real_files:
            logger.error("No real images found for alteration.")
            return 0

        collected = 0
        random.shuffle(real_files)

        prompts = [
            "replace with a realistic flower",
            "remove the object and fill with natural background",
            "add a cute cat sitting",
            "replace with a modern lamp",
            "swap face with a celebrity",
            "remove text and clean the area",
            "add a window showing a sunset",
            "replace with a vintage clock",
            "remove the background and place in a forest",
            "change the color to neon blue",
        ]

        for img_path in cycle(real_files):
            if collected >= count:
                break
            try:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size

                # Create a random mask (rectangle covering ~30% area)
                mask = Image.new("L", (w, h), 0)
                mask_draw = ImageDraw.Draw(mask)
                x0 = random.randint(0, w//2)
                y0 = random.randint(0, h//2)
                x1 = random.randint(x0 + w//4, w)
                y1 = random.randint(y0 + h//4, h)
                mask_draw.rectangle([x0, y0, x1, y1], fill=255)

                prompt = random.choice(prompts)

                result = pipe(
                    prompt=prompt,
                    image=img,
                    mask_image=mask,
                    num_inference_steps=30,
                    guidance_scale=7.5
                ).images[0]

                img_id = f"ALTERED_{collected}_{img_path.stem}"
                filename = f"{img_id}.jpg"
                byte_arr = io.BytesIO()
                result.save(byte_arr, format='JPEG', quality=95)
                data = byte_arr.getvalue()

                h = self._calculate_hash(data)
                dup = self._check_duplicate(h)
                w_out, h_out, fmt, cs = self._get_image_info(data)

                if not self._save_image(data, filename):
                    continue

                self.metadata_list.append(ImageMetadata(
                    image_id=img_id,
                    filename=filename,
                    source=ImageSource.AI_ALTERED_LOCAL.value,
                    source_url=str(img_path),
                    download_date=datetime.now().isoformat(),
                    image_type=ImageType.AI_ALTERED.value,
                    width=w_out,
                    height=h_out,
                    format=fmt,
                    file_size_kb=round(len(data)/1024, 2),
                    color_space=cs,
                    source_metadata={
                        "original_file": img_path.name,
                        "prompt": prompt,
                        "mask_coords": [x0, y0, x1, y1]
                    },
                    md5_hash=h,
                    is_duplicate=dup is not None,
                    duplicate_of=dup,
                    analysis_notes=f"Inpainted: {prompt}"
                ))
                collected += 1
                if collected % 100 == 0:
                    logger.info(f"  Altered: {collected}/{count}")

            except Exception as e:
                logger.warning(f"Alteration failed for {img_path}: {e}")
                continue

        logger.info(f"  ✓ AI‑altered: {collected}")
        return collected

# =============================================================================
#  SECTION 7: MAIN ORCHESTRATOR (UPDATED FOR 50K + AI‑ALTERED)
# =============================================================================
class DatasetCollectorV2:
    REAL_QUERIES = [
        "portrait professional", "landscape nature", "objects products",
        "face human", "mountain scenery", "food photography",
        "people street", "beach water", "animal wildlife",
        "city urban architecture", "abstract texture", "forest trees",
        "night photography", "sunset sky", "family children",
        "sport action", "flower macro", "rain weather",
        "desert sand", "winter snow", "street art graffiti",
        "modern technology", "vintage car", "kitchen interior",
        "office desk", "macro insect", "airplane sky",
    ]

    def __init__(self, output_dir: Path = Path("./ai_dataset_50k")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.real_metadata: List[ImageMetadata] = []
        self.ai_metadata: List[ImageMetadata] = []
        self.altered_metadata: List[ImageMetadata] = []
        self.collection_log = {
            "start_time": datetime.now().isoformat(),
            "collections": [],
            "config": {}
        }

    @property
    def all_metadata(self) -> List[ImageMetadata]:
        return self.real_metadata + self.ai_metadata + self.altered_metadata

    def load_api_keys(self) -> Dict[str, str]:
        load_dotenv()
        keys = {
            "UNSPLASH": os.getenv("UNSPLASH_API_KEY"),
            "PEXELS": os.getenv("PEXELS_API_KEY"),
            "PIXABAY": os.getenv("PIXABAY_API_KEY"),
            "CIVITAI": os.getenv("CIVITAI_API_KEY"),
            "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN"),
        }
        for k, v in keys.items():
            status = "✓ loaded" if v else "✗ MISSING"
            logger.info(f"  {k:20}: {status}")
        if not keys["CIVITAI"]:
            logger.warning(
                "\n  *** Get a FREE CivitAI API token at: "
                "https://civitai.com/user/account → API Keys\n"
                "  *** Add to .env: CIVITAI_API_KEY=your_token\n"
            )
        return keys

    def collect_real_images(self, api_keys: Dict[str, str], target: int = 25000) -> int:
        logger.info(f"\n{'='*70}\nCOLLECTING REAL IMAGES  (Target: {target})\n{'='*70}")
        active = [k for k in ("UNSPLASH", "PEXELS", "PIXABAY") if api_keys.get(k)]
        if not active:
            logger.error("No real-image API keys found!"); return 0
        per_source = -(-target // len(active))
        total = 0
        if api_keys.get("UNSPLASH"):
            c = UnsplashCollector(api_keys["UNSPLASH"], self.output_dir)
            n = c.collect(self.REAL_QUERIES, per_source)
            self.real_metadata.extend(c.metadata_list)
            self.collection_log["collections"].append({"source": "Unsplash", "count": n})
            total += n
        if api_keys.get("PEXELS"):
            c = PexelsCollector(api_keys["PEXELS"], self.output_dir)
            n = c.collect(self.REAL_QUERIES, per_source)
            self.real_metadata.extend(c.metadata_list)
            self.collection_log["collections"].append({"source": "Pexels", "count": n})
            total += n
        if api_keys.get("PIXABAY"):
            c = PixabayCollector(api_keys["PIXABAY"], self.output_dir)
            n = c.collect(self.REAL_QUERIES, per_source)
            self.real_metadata.extend(c.metadata_list)
            self.collection_log["collections"].append({"source": "Pixabay", "count": n})
            total += n
        logger.info(f"✓ Real images total: {total} / {target}\n")
        return total

    def collect_ai_images(self, api_keys: Dict[str, str], target: int = 17500) -> int:
        logger.info(f"{'='*70}\nCOLLECTING AI IMAGES  (Target: {target})\n{'='*70}")
        civitai_target = int(target * 0.40)
        hf_target = int(target * 0.35)
        poll_target = target - civitai_target - hf_target
        total = 0

        c = CivitaiCollector(self.output_dir, api_key=api_keys.get("CIVITAI"))
        n = c.collect(civitai_target)
        self.ai_metadata.extend(c.metadata_list)
        self.collection_log["collections"].append({"source": "CivitAI", "count": n})
        total += n
        hf_target += max(0, civitai_target - n)

        c = HuggingFaceCollector(self.output_dir)
        n = c.collect(hf_target)
        self.ai_metadata.extend(c.metadata_list)
        self.collection_log["collections"].append({"source": "HuggingFace", "count": n})
        total += n

        remaining = target - total
        if remaining > 0:
            c = PollinationsCollector(self.output_dir)
            n = c.collect(remaining)
            self.ai_metadata.extend(c.metadata_list)
            self.collection_log["collections"].append({"source": "Pollinations", "count": n})
            total += n

        logger.info(f"✓ AI images total: {total} / {target}\n")
        return total

    def collect_ai_altered_images(self, real_images_dir: Path, count: int = 7500) -> int:
        logger.info(f"\n{'='*70}\nCOLLECTING AI‑ALTERED IMAGES (Target: {count})\n{'='*70}")
        # Use the HuggingFace token (for loading the SD model)
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        c = AIAlteredCollector(
            output_dir=self.output_dir,
            real_images_dir=real_images_dir,
            hf_token=hf_token
        )
        n = c.collect(count)
        self.altered_metadata.extend(c.metadata_list)
        self.collection_log["collections"].append({"source": "AI‑altered", "count": n})
        logger.info(f"✓ AI‑altered total: {n}\n")
        return n

    def generate_report(self) -> Dict:
        all_meta = self.all_metadata
        report = {
            "total_images": len(all_meta),
            "real_images": len(self.real_metadata),
            "ai_generated_images": len(self.ai_metadata),
            "ai_altered_images": len(self.altered_metadata),
            "duplicates_found": sum(1 for m in all_meta if m.is_duplicate),
            "by_source": {},
            "timestamp": datetime.now().isoformat()
        }
        for m in all_meta:
            report["by_source"][m.source] = report["by_source"].get(m.source, 0) + 1
        logger.info(f"\n{'='*70}\nDATASET SUMMARY\n{'='*70}")
        logger.info(f"Total:      {report['total_images']}")
        logger.info(f"Real:       {report['real_images']}")
        logger.info(f"AI:         {report['ai_generated_images']}")
        logger.info(f"Altered:    {report['ai_altered_images']}")
        logger.info(f"Duplicates: {report['duplicates_found']}")
        for src, cnt in sorted(report["by_source"].items()):
            pct = cnt / max(report['total_images'], 1) * 100
            logger.info(f"  {src:.<45} {cnt:>5} ({pct:>5.1f}%)")
        return report

    def save_metadata(self):
        logger.info("Saving metadata files…")
        def _write(meta_list, label, stem):
            if not meta_list:
                logger.warning(f"  ✗ No {label} metadata to save"); return
            df = pd.DataFrame([m.to_dict() for m in meta_list])
            df.to_csv(self.output_dir / f"{stem}.csv", index=False)
            with open(self.output_dir / f"{stem}.json", 'w') as f:
                json.dump([m.to_dict() for m in meta_list], f, indent=2)
            logger.info(f"  ✓ {label:6} → {stem}.csv / .json  ({len(df)} records)")

        _write(self.real_metadata, "Real", "dataset_metadata_real")
        _write(self.ai_metadata, "AI", "dataset_metadata_ai")
        _write(self.altered_metadata, "Altered", "dataset_metadata_ai_altered")
        _write(self.all_metadata, "All", "dataset_metadata_all")

        self.collection_log["end_time"] = datetime.now().isoformat()
        with open(self.output_dir / "collection_log.json", 'w') as f:
            json.dump(self.collection_log, f, indent=2)
        logger.info(f"  ✓ Log   → collection_log.json\n")

    def run_full_collection(self,
                            real_count: int = 25000,
                            ai_generated_count: int = 17500,
                            ai_altered_count: int = 7500) -> Dict:
        self.collection_log["config"] = {
            "real_target": real_count,
            "ai_generated_target": ai_generated_count,
            "ai_altered_target": ai_altered_count
        }
        api_keys = self.load_api_keys()

        # 1. Real images
        self.collect_real_images(api_keys, real_count)

        # 2. AI‑generated images
        self.collect_ai_images(api_keys, ai_generated_count)

        # 3. AI‑altered images (requires real images folder)
        real_images_dir = self.output_dir / "real_images"
        if not real_images_dir.exists():
            logger.error("Real images folder missing – cannot generate altered images.")
        else:
            self.collect_ai_altered_images(real_images_dir, ai_altered_count)

        report = self.generate_report()
        self.save_metadata()
        return report

# =============================================================================
#  SECTION 8: EXECUTION (RUN THIS CELL OR BLOCK TO START)
# =============================================================================
if __name__ == "__main__":
    collector = DatasetCollectorV2(output_dir=Path("./ai_dataset_50k"))
    report = collector.run_full_collection(
        real_count=25000,
        ai_generated_count=17500,
        ai_altered_count=7500
    )
    print(json.dumps(report, indent=2))