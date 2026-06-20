"""
================================================================================
AI IMAGE DATASET COLLECTOR - 2026 WORKING VERSION
================================================================================
Real images   : Unsplash, Pexels, Pixabay  (API keys in .env)
AI images     : CivitAI (free token), Hugging Face DiffusionDB, Pollinations.ai

Why the old AI sources broke:
  - Lexica.art   → requires paid auth since late 2025
  - OpenArt.ai   → closed public API
  - CivitAI      → now requires Bearer token (free account, get at civitai.com/user/account)

New sources:
  - CivitAI          : free API token from civitai.com/user/account  → add CIVITAI_API_KEY to .env
  - HuggingFace      : DiffusionDB dataset, no key needed
  - Pollinations.ai  : free prompt-to-image, no key needed

.env file should contain:
    UNSPLASH_API_KEY=...
    PEXELS_API_KEY=...
    PIXABAY_API_KEY=...
    CIVITAI_API_KEY=...        ← new, get free at civitai.com/user/account

Output files:
    dataset_metadata_real.csv / .json
    dataset_metadata_ai.csv   / .json
    dataset_metadata_all.csv  / .json
    collection_log.json
"""

import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from PIL import Image
import io
from tqdm import tqdm
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class ImageType(Enum):
    REAL         = "real"
    AI_GENERATED = "ai_generated"

class ImageSource(Enum):
    UNSPLASH     = "unsplash"
    PEXELS       = "pexels"
    PIXABAY      = "pixabay"
    CIVITAI      = "civitai"
    HUGGINGFACE  = "huggingface"
    POLLINATIONS = "pollinations"
    LOCAL        = "local"

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


# ============================================================================
# BASE COLLECTOR
# ============================================================================

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
        except Exception:
            return 0, 0, 'UNKNOWN', 'UNKNOWN'

    def _save_image(self, data: bytes, filename: str) -> bool:
        try:
            (self.output_dir / filename).write_bytes(data)
            return True
        except Exception:
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

            # reject tiny / non-image responses
            if len(data) < 5000:
                return False
            content_type = r.headers.get('content-type', '')
            if 'image' not in content_type and not url.lower().endswith(
                    ('.jpg', '.jpeg', '.png', '.webp')):
                return False

            h   = self._calculate_hash(data)
            dup = self._check_duplicate(h)
            filename = f"{image_id}.jpg"
            w, ht, fmt, cs = self._get_image_info(data)

            if not self._save_image(data, filename):
                return False

            self.metadata_list.append(ImageMetadata(
                image_id=image_id, filename=filename,
                source=source_val, source_url=url,
                download_date=datetime.now().isoformat(),
                image_type=image_type,
                width=w, height=ht, format=fmt,
                file_size_kb=round(len(data) / 1024, 2),
                color_space=cs, source_metadata=source_meta,
                md5_hash=h,
                is_duplicate=dup is not None, duplicate_of=dup,
                analysis_notes=note
            ))
            return True
        except Exception as e:
            logger.debug(f"Download failed ({image_id}): {e}")
            return False


# ============================================================================
# REAL IMAGE COLLECTORS  (unchanged — these work fine)
# ============================================================================

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
                    if self._download_and_record(
                        url, f"UNSPLASH_{res['id']}", ImageType.REAL.value,
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
                    if self._download_and_record(
                        url, f"PEXELS_{p['id']}", ImageType.REAL.value,
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
                r = self.session.get(
                    "https://pixabay.com/api/",
                    params={"key": self.api_key, "q": query,
                            "per_page": 200, "page": page, "image_type": "photo"},
                    timeout=15)
                r.raise_for_status()
                hits = r.json().get('hits', [])
                if not hits:
                    q_idx += 1; page = 1; continue
                for h in hits:
                    if collected >= total_count: break
                    url = h.get('largeImageURL', '')
                    if not url: continue
                    if self._download_and_record(
                        url, f"PIXABAY_{h['id']}", ImageType.REAL.value,
                        ImageSource.PIXABAY.value,
                        {"user": h['user']}, f"Query: {query}"):
                        collected += 1
                page += 1
            except Exception as e:
                logger.warning(f"Pixabay error '{query}': {e}")
                q_idx += 1; page = 1
        logger.info(f"  ✓ Pixabay: {collected}")
        return collected


# ============================================================================
# AI IMAGE COLLECTORS  — all three replaced/fixed
# ============================================================================

class CivitaiCollector(BaseCollector):
    """
    CivitAI now requires a Bearer token (free).
    Get yours at: https://civitai.com/user/account  → API Keys section
    Add to .env: CIVITAI_API_KEY=your_token_here

    Falls back to no-auth if key missing (may get fewer results).
    Uses multiple sort/period combos to work around cursor exhaustion.
    """

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
            if collected >= count:
                break
            cursor            = None
            consecutive_empty = 0

            while collected < count:
                try:
                    params = {
                        "limit":  100,
                        "nsfw":   "None",
                        "sort":   combo["sort"],
                        "period": combo["period"],
                        "type":   "image",        # images only, not videos
                    }
                    if cursor:
                        params["cursor"] = cursor

                    r = self.session.get(
                        "https://civitai.com/api/v1/images",
                        params=params, timeout=20)

                    if r.status_code == 401:
                        logger.error("CivitAI 401: invalid or missing API token. "
                                     "Get a free token at civitai.com/user/account")
                        return collected
                    if r.status_code == 429:
                        logger.warning("CivitAI rate-limited, waiting 45s…")
                        time.sleep(45); continue
                    if r.status_code == 403:
                        logger.error("CivitAI 403: check your API token permissions")
                        return collected

                    r.raise_for_status()
                    result = r.json()
                    items  = result.get('items', [])

                    if not items:
                        consecutive_empty += 1
                        if consecutive_empty >= 2:
                            break
                        continue

                    consecutive_empty = 0
                    for item in items:
                        if collected >= count: break
                        url = item.get('url', '')
                        if not url: continue
                        # skip videos
                        if item.get('type', 'image') != 'image': continue
                        if self._download_and_record(
                            url,
                            f"CIVITAI_{item.get('id', f'c{collected}')}",
                            ImageType.AI_GENERATED.value,
                            ImageSource.CIVITAI.value,
                            {
                                "model":  str(item.get('meta', {}).get('Model', ''))[:100],
                                "prompt": str(item.get('meta', {}).get('prompt', ''))[:200],
                            },
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
    """
    Downloads AI-generated images from the DiffusionDB dataset hosted on
    Hugging Face Datasets Server. No API key required — fully public.
    Dataset: poloclub/diffusiondb  (14M Stable Diffusion images)
    Docs: https://huggingface.co/datasets/poloclub/diffusiondb
    """

    HF_API = "https://datasets-server.huggingface.co/rows"
    CONFIGS = [
        ("2m_random_1k",  "train", 1000),
        ("2m_random_5k",  "train", 5000),
        ("2m_random_10k", "train", 10000),
    ]

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, "ai_generated_images")

    def collect(self, count: int = 400) -> int:
        logger.info(f"HuggingFace DiffusionDB: targeting {count} images")
        collected = 0

        for config, split, max_rows in self.CONFIGS:
            if collected >= count:
                break

            offset    = 0
            page_size = 100

            while collected < count and offset < max_rows:
                try:
                    r = self.session.get(
                        self.HF_API,
                        params={
                            "dataset": "poloclub/diffusiondb",
                            "config":  config,
                            "split":   split,
                            "offset":  offset,
                            "length":  page_size,
                        },
                        timeout=30)

                    if r.status_code == 429:
                        logger.warning("HuggingFace rate-limited, waiting 30s…")
                        time.sleep(30); continue
                    if not r.ok:
                        logger.warning(f"HuggingFace {r.status_code} for config {config}")
                        break

                    rows = r.json().get('rows', [])
                    if not rows:
                        break

                    for row in rows:
                        if collected >= count: break
                        row_data = row.get('row', {})

                        # DiffusionDB rows contain an 'image' field with a src URL
                        image_field = row_data.get('image', {})
                        if isinstance(image_field, dict):
                            img_url = image_field.get('src') or image_field.get('url', '')
                        elif isinstance(image_field, str):
                            img_url = image_field
                        else:
                            continue

                        if not img_url:
                            continue

                        prompt   = str(row_data.get('prompt', ''))[:300]
                        seed     = str(row_data.get('seed', ''))
                        img_id   = f"HF_{config}_{offset}_{row.get('row_idx', collected)}"

                        if self._download_and_record(
                            img_url, img_id,
                            ImageType.AI_GENERATED.value,
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
    """
    Pollinations.ai — completely free, no API key, no rate limits.
    Generates images on the fly from prompts.
    API: https://image.pollinations.ai/prompt/{prompt}?width=512&height=512&nologo=true

    We use a list of diverse prompts to generate varied AI images.
    Each call generates a unique image (seeded by timestamp).
    """

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
        collected  = 0
        prompt_idx = 0

        while collected < count:
            prompt = self.PROMPTS[prompt_idx % len(self.PROMPTS)]
            # Add seed variation so the same prompt gives different images
            seed   = int(time.time() * 1000) % 999999

            try:
                import urllib.parse
                encoded = urllib.parse.quote(prompt)
                url = (f"https://image.pollinations.ai/prompt/{encoded}"
                       f"?width=768&height=768&seed={seed}&nologo=true&enhance=true")

                img_id = f"POLLINATIONS_{collected}_{seed}"

                if self._download_and_record(
                    url, img_id,
                    ImageType.AI_GENERATED.value,
                    ImageSource.POLLINATIONS.value,
                    {"prompt": prompt, "seed": seed},
                    f"Pollinations: {prompt[:50]}"):
                    collected += 1
                    logger.debug(f"  Pollinations {collected}/{count}: {prompt[:40]}")

                prompt_idx += 1
                time.sleep(1.5)   # Pollinations asks for gentle usage

            except Exception as e:
                logger.warning(f"Pollinations error: {e}")
                prompt_idx += 1
                time.sleep(2)

        logger.info(f"  ✓ Pollinations: {collected}")
        return collected


class LocalFolderCollector(BaseCollector):
    def collect_from_folder(self, folder_path: Path, image_type: str = "ai_generated") -> int:
        logger.info(f"Local folder: {folder_path}")
        if not folder_path.exists():
            logger.error(f"Not found: {folder_path}"); return 0

        sub = "real_images" if image_type == "real" else "ai_generated_images"
        self.output_dir = self.output_dir.parent / sub
        self.output_dir.mkdir(parents=True, exist_ok=True)

        collected = 0
        supported = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        files = [f for f in folder_path.rglob('*') if f.suffix.lower() in supported]
        logger.info(f"Found {len(files)} images")

        for f in tqdm(files, desc="Local"):
            try:
                data = f.read_bytes()
                if len(data) < 1000: continue
                h   = self._calculate_hash(data)
                dup = self._check_duplicate(h)
                iid  = f"LOCAL_{f.stem}_{collected}"
                fname = f"{iid}{f.suffix.lower()}"
                w, ht, fmt, cs = self._get_image_info(data)
                if self._save_image(data, fname):
                    self.metadata_list.append(ImageMetadata(
                        image_id=iid, filename=fname,
                        source=ImageSource.LOCAL.value, source_url=str(f),
                        download_date=datetime.now().isoformat(),
                        image_type=image_type,
                        width=w, height=ht, format=fmt,
                        file_size_kb=round(len(data) / 1024, 2),
                        color_space=cs,
                        source_metadata={"original_filename": f.name},
                        md5_hash=h, is_duplicate=dup is not None, duplicate_of=dup,
                        analysis_notes="Local import"
                    ))
                    collected += 1
            except Exception:
                continue
        return collected


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class DatasetCollectorV2:

    REAL_QUERIES = [
        "portrait professional", "landscape nature", "objects products",
        "face human", "mountain scenery", "food photography",
        "people street", "beach water", "animal wildlife",
        "city urban architecture", "abstract texture", "forest trees",
        "night photography", "sunset sky", "family children",
        "sport action", "flower macro", "rain weather",
        "desert sand", "winter snow",
    ]

    def __init__(self, output_dir: Path = Path("./ai_dataset")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.real_metadata: List[ImageMetadata] = []
        self.ai_metadata:   List[ImageMetadata] = []
        self.collection_log = {
            "start_time":  datetime.now().isoformat(),
            "collections": [],
            "config":      {}
        }

    @property
    def all_metadata(self) -> List[ImageMetadata]:
        return self.real_metadata + self.ai_metadata

    def load_api_keys(self) -> Dict[str, str]:
        from dotenv import load_dotenv
        load_dotenv()
        keys = {
            "UNSPLASH":  os.getenv("UNSPLASH_API_KEY"),
            "PEXELS":    os.getenv("PEXELS_API_KEY"),
            "PIXABAY":   os.getenv("PIXABAY_API_KEY"),
            "CIVITAI":   os.getenv("CIVITAI_API_KEY"),
        }
        for k, v in keys.items():
            status = "✓ loaded" if v else "✗ MISSING"
            logger.info(f"  {k:10}: {status}")
        if not keys["CIVITAI"]:
            logger.warning(
                "\n  *** Get a FREE CivitAI API token at: "
                "https://civitai.com/user/account → API Keys\n"
                "  *** Add to .env: CIVITAI_API_KEY=your_token\n"
                "  *** Without it CivitAI may return 0 images\n"
            )
        return keys

    # ---------------------------------------------------------------------- #
    #  REAL IMAGES                                                            #
    # ---------------------------------------------------------------------- #

    def collect_real_images(self, api_keys: Dict[str, str], target: int = 1000) -> int:
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

    # ---------------------------------------------------------------------- #
    #  AI IMAGES                                                              #
    # ---------------------------------------------------------------------- #

    def collect_ai_images(self, api_keys: Dict[str, str], target: int = 1000) -> int:
        logger.info(f"{'='*70}\nCOLLECTING AI IMAGES  (Target: {target})\n{'='*70}")

        # Budget: CivitAI=40%, HuggingFace=35%, Pollinations=25%
        # Pollinations is slowest (1.5s/image) so keep its share lower
        civitai_target = int(target * 0.40)
        hf_target      = int(target * 0.35)
        poll_target    = target - civitai_target - hf_target

        total = 0

        # CivitAI
        c = CivitaiCollector(self.output_dir, api_key=api_keys.get("CIVITAI"))
        n = c.collect(civitai_target)
        self.ai_metadata.extend(c.metadata_list)
        self.collection_log["collections"].append({"source": "CivitAI", "count": n})
        total += n
        logger.info(f"  Running AI total: {total}")

        # If CivitAI under-delivered, give the slack to HuggingFace
        hf_target += max(0, civitai_target - n)

        # HuggingFace DiffusionDB
        c = HuggingFaceCollector(self.output_dir)
        n = c.collect(hf_target)
        self.ai_metadata.extend(c.metadata_list)
        self.collection_log["collections"].append({"source": "HuggingFace", "count": n})
        total += n
        logger.info(f"  Running AI total: {total}")

        # Pollinations fills any remaining gap
        remaining = target - total
        if remaining > 0:
            c = PollinationsCollector(self.output_dir)
            n = c.collect(remaining)
            self.ai_metadata.extend(c.metadata_list)
            self.collection_log["collections"].append({"source": "Pollinations", "count": n})
            total += n
            logger.info(f"  Running AI total: {total}")

        logger.info(f"✓ AI images total: {total} / {target}\n")
        return total

    # ---------------------------------------------------------------------- #
    #  LOCAL FOLDERS                                                          #
    # ---------------------------------------------------------------------- #

    def collect_from_local_folders(self, folders: Dict[str, Path]) -> int:
        logger.info(f"{'='*70}\nLOCAL FOLDERS\n{'='*70}")
        total = 0
        for image_type, folder_path in folders.items():
            try:
                c = LocalFolderCollector(self.output_dir)
                n = c.collect_from_folder(Path(folder_path), image_type=image_type)
                if image_type == "real":
                    self.real_metadata.extend(c.metadata_list)
                else:
                    self.ai_metadata.extend(c.metadata_list)
                self.collection_log["collections"].append(
                    {"source": f"Local:{Path(folder_path).name}",
                     "type": image_type, "count": n})
                total += n
            except Exception as e:
                logger.error(f"Local folder failed: {e}")
        return total

    # ---------------------------------------------------------------------- #
    #  REPORT + SAVE                                                          #
    # ---------------------------------------------------------------------- #

    def generate_report(self) -> Dict:
        all_meta = self.all_metadata
        report = {
            "total_images":        len(all_meta),
            "real_images":         len(self.real_metadata),
            "ai_generated_images": len(self.ai_metadata),
            "duplicates_found":    sum(1 for m in all_meta if m.is_duplicate),
            "by_source":           {},
            "timestamp":           datetime.now().isoformat()
        }
        for m in all_meta:
            report["by_source"][m.source] = report["by_source"].get(m.source, 0) + 1

        logger.info(f"\n{'='*70}\nDATASET SUMMARY\n{'='*70}")
        logger.info(f"Total:      {report['total_images']}")
        logger.info(f"Real:       {report['real_images']}")
        logger.info(f"AI:         {report['ai_generated_images']}")
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

        _write(self.real_metadata,  "Real",     "dataset_metadata_real")
        _write(self.ai_metadata,    "AI",       "dataset_metadata_ai")
        _write(self.all_metadata,   "All",      "dataset_metadata_all")

        self.collection_log["end_time"] = datetime.now().isoformat()
        with open(self.output_dir / "collection_log.json", 'w') as f:
            json.dump(self.collection_log, f, indent=2)
        logger.info(f"  ✓ Log   → collection_log.json\n")

    # ---------------------------------------------------------------------- #
    #  FULL RUN                                                               #
    # ---------------------------------------------------------------------- #

    def run_full_collection(self, real_count: int = 1000, ai_count: int = 1000,
                             local_folders: Dict[str, Path] = None) -> Dict:
        self.collection_log["config"] = {
            "real_target": real_count,
            "ai_target":   ai_count,
            "has_local_folders": local_folders is not None
        }
        api_keys = self.load_api_keys()
        self.collect_real_images(api_keys, real_count)
        self.collect_ai_images(api_keys, ai_count)   # ← now passes api_keys for CivitAI token
        if local_folders:
            self.collect_from_local_folders(local_folders)
        report = self.generate_report()
        self.save_metadata()
        return report