#!/usr/bin/env python3
"""
FAST AI-GENERATED IMAGES COLLECTOR – FIXED
Uses: CivitAI (cursor) + HuggingFace DiffusionDB (Parquet) + Pollinations
Target: 17,500 images for 50k pipeline
Time: ~1–2 hours
"""

import os, io, time, json, logging, random, hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# Optional but recommended for HuggingFace
try:
    from huggingface_hub import list_repo_files, hf_hub_download
except ImportError:
    list_repo_files = None
    hf_hub_download = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
class ImageSource(Enum):
    CIVITAI      = "civitai"
    HUGGINGFACE  = "huggingface"
    POLLINATIONS = "pollinations"

@dataclass
class ImageMetadata:
    image_id: str
    filename: str
    source: str
    source_url: str
    download_date: str
    image_type: str = "ai_generated"
    width: int = 0
    height: int = 0
    format: str = ""
    file_size_kb: float = 0.0
    color_space: str = ""
    source_metadata: Dict = None
    md5_hash: str = ""
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    analysis_notes: str = ""

    def __post_init__(self):
        if self.source_metadata is None:
            self.source_metadata = {}

    def to_dict(self):
        return asdict(self)

# ---------------------------------------------------------------------------
# Thread-safe base collector
# ---------------------------------------------------------------------------
class BaseCollector:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metadata_list: List[ImageMetadata] = []
        self.lock = Lock()
        self._hash_set = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _calc_hash(self, data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    def _image_info(self, data: bytes):
        try:
            img = Image.open(io.BytesIO(data))
            return img.size[0], img.size[1], img.format or 'UNKNOWN', img.mode
        except:
            return 0, 0, 'UNKNOWN', 'UNKNOWN'

    def _save(self, data: bytes, filename: str) -> bool:
        try:
            (self.output_dir / filename).write_bytes(data)
            return True
        except:
            return False

    def _download_one(self, url: str, img_id: str, source: str,
                      meta: dict, note: str) -> bool:
        """Download a single image, thread-safe."""
        try:
            r = self.session.get(url, timeout=15)
            if r.status_code != 200: return False
            data = r.content
            if len(data) < 2000: return False

            h = self._calc_hash(data)
            with self.lock:
                if h in self._hash_set:
                    return False
                self._hash_set.add(h)

            w, ht, fmt, cs = self._image_info(data)
            filename = f"{img_id}.jpg"
            if not self._save(data, filename): return False

            with self.lock:
                self.metadata_list.append(ImageMetadata(
                    image_id=img_id, filename=filename, source=source,
                    source_url=url, download_date=datetime.now().isoformat(),
                    width=w, height=ht, format=fmt,
                    file_size_kb=round(len(data)/1024, 2), color_space=cs,
                    source_metadata=meta, md5_hash=h, analysis_notes=note
                ))
            return True
        except:
            return False

# ---------------------------------------------------------------------------
# 1. CIVITAI COLLECTOR (sequential, reliable)
# ---------------------------------------------------------------------------
class CivitaiCollector(BaseCollector):
    def __init__(self, output_dir: Path, api_key: str):
        super().__init__(output_dir)
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    def collect(self, count: int) -> int:
        logger.info(f"CivitAI: targeting {count}")
        collected = 0
        pbar = tqdm(total=count, desc="CivitAI")
        cursor = None
        sort_combos = [
            ("Most Reactions", "AllTime"),
            ("Most Reactions", "Month"),
            ("Newest", "Month"),
            ("Most Comments", "AllTime"),
        ]
        combo_idx = 0

        while collected < count:
            sort, period = sort_combos[combo_idx % len(sort_combos)]
            try:
                params = {
                    "limit": 100, "nsfw": "None", "type": "image",
                    "sort": sort, "period": period
                }
                if cursor: params["cursor"] = cursor

                r = self.session.get("https://civitai.com/api/v1/images",
                                     params=params, timeout=20)
                if r.status_code == 429:
                    logger.warning("Rate limited – waiting 60s…"); time.sleep(60); continue
                if r.status_code in (401,403):
                    logger.error("CivitAI auth failed"); break
                if not r.ok:
                    combo_idx += 1; cursor = None; continue

                data = r.json()
                items = data.get("items", [])
                if not items:
                    combo_idx += 1; cursor = None; continue

                for item in items:
                    if collected >= count: break
                    url = item.get("url")
                    if not url or item.get("type") != "image": continue

                    if self._download_one(
                        url,
                        f"CIVITAI_{item.get('id')}",
                        ImageSource.CIVITAI.value,
                        {"prompt": str(item.get("meta", {}).get("prompt", ""))[:200]},
                        f"CivitAI {sort}"
                    ):
                        collected += 1
                        pbar.update(1)

                cursor = data.get("metadata", {}).get("nextCursor")
                if not cursor:
                    combo_idx += 1
                    cursor = None

                time.sleep(0.3)  # polite delay

            except Exception as e:
                logger.debug(f"CivitAI error: {e}")
                combo_idx += 1; cursor = None

        pbar.close()
        logger.info(f"  ✓ CivitAI: {collected}")
        return collected

# ---------------------------------------------------------------------------
# 2. HUGGINGFACE DIFFUSIONDB (via Parquet files – reliable)
# ---------------------------------------------------------------------------
class DiffusionDBParquetCollector(BaseCollector):
    """
    Downloads the actual Parquet files from DiffusionDB, extracts image URLs,
    and downloads them in parallel.
    Requires: pip install huggingface_hub pandas pyarrow
    """
    def __init__(self, output_dir: Path, hf_token: str = None):
        super().__init__(output_dir)
        self.hf_token = hf_token

    def collect(self, count: int) -> int:
        logger.info(f"DiffusionDB Parquet: targeting {count}")
        if not list_repo_files:
            logger.error("huggingface_hub not installed. Run: pip install huggingface_hub pandas pyarrow")
            return 0

        collected = 0
        pbar = tqdm(total=count, desc="DiffusionDB")

        # Get list of parquet files from the dataset
        try:
            files = list_repo_files("poloclub/diffusiondb", repo_type="dataset", token=self.hf_token)
            parquet_files = [f for f in files if f.endswith(".parquet") and "random" in f]
            logger.info(f"Found {len(parquet_files)} Parquet files")
        except Exception as e:
            logger.error(f"Failed to list DiffusionDB files: {e}")
            return 0

        # Shuffle to get diversity
        random.shuffle(parquet_files)

        with ThreadPoolExecutor(max_workers=4) as executor:
            for file in parquet_files:
                if collected >= count:
                    break
                try:
                    logger.info(f"Processing {file} …")
                    local_path = hf_hub_download(
                        "poloclub/diffusiondb", file, repo_type="dataset", token=self.hf_token
                    )
                    import pandas as pd
                    df = pd.read_parquet(local_path)
                    # The column containing image URL is 'image'
                    if 'image' not in df.columns:
                        continue
                    urls = df['image'].dropna().tolist()
                    random.shuffle(urls)

                    # Download in parallel (limit to 10 concurrent per file)
                    futures = []
                    for url in urls:
                        if collected >= count: break
                        futures.append(executor.submit(
                            self._download_one,
                            url,
                            f"HFDB_{collected}",
                            ImageSource.HUGGINGFACE.value,
                            {"source": "diffusiondb"},
                            "DiffusionDB Parquet"
                        ))
                        collected += 1  # we'll track attempted, update on success later

                    # Wait for all futures and count actual successes
                    actual_collected = 0
                    for f in as_completed(futures):
                        if f.result():
                            actual_collected += 1
                            pbar.update(1)
                    # Update collected count to reflect successful downloads
                    with self.lock:
                        # Adjust collected downwards if many failed
                        pass  # metadata_list already contains all successful ones

                except Exception as e:
                    logger.warning(f"Error processing {file}: {e}")
                    continue

        pbar.close()
        logger.info(f"  ✓ DiffusionDB: {len(self.metadata_list)}")
        return len(self.metadata_list)

# ---------------------------------------------------------------------------
# 3. POLLINATIONS (fill gap, slow)
# ---------------------------------------------------------------------------
class PollinationsCollector(BaseCollector):
    PROMPTS = [
        "fantasy landscape mountains lake digital painting",
        "cyberpunk woman neon lights portrait",
        "futuristic city skyline flying cars night rain",
        "medieval knight oil painting dramatic lighting",
        "anime girl blue hair magical forest sakura",
        "abstract geometry vibrant colors digital art",
        "old man portrait wrinkles realistic black white",
        "sci-fi spaceship nebula stars",
        "watercolor cherry blossom Japanese temple",
        "dragon breathing fire castle mountain",
        "hyperrealistic still life fruit wooden table",
        "surreal melting clocks Dali landscape",
        "jazz musician saxophone moody lighting",
        "pop art cat colorful Warhol",
        "Paris street rain impressionist oil painting",
        "3D robot coffee shop cozy",
        "fantasy island map parchment cartography",
        "macro butterfly flower morning dew",
        "gothic cathedral interior stained glass",
        "pixel art sunset 8-bit mountains retro",
        "post-apocalyptic city overgrown plants concept art",
        "minimalist line art woman portrait",
        "wolf snowy forest winter realistic",
        "steampunk mechanical elephant clockwork",
        "coral reef underwater colorful fish",
        "aurora borealis snowy landscape pine trees",
        "phoenix rising flames digital art",
        "Art Nouveau woman flowers golden frame",
        "Cubist portrait Picasso style",
        "neon Tokyo street rain cyberpunk",
    ]

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)

    def collect(self, count: int) -> int:
        logger.info(f"Pollinations: targeting {count}")
        collected = 0
        pbar = tqdm(total=count, desc="Pollinations")
        idx = 0
        while collected < count:
            prompt = self.PROMPTS[idx % len(self.PROMPTS)]
            seed = random.randint(1000, 999999)
            try:
                import urllib.parse
                encoded = urllib.parse.quote(prompt)
                url = (f"https://image.pollinations.ai/prompt/{encoded}"
                       f"?width=768&height=768&seed={seed}&nologo=true&enhance=true")
                if self._download_one(
                    url,
                    f"POLL_{collected}_{seed}",
                    ImageSource.POLLINATIONS.value,
                    {"prompt": prompt, "seed": seed},
                    "Pollinations"
                ):
                    collected += 1
                    pbar.update(1)
                time.sleep(1.0)  # required rate limit
            except Exception:
                time.sleep(1.5)
            idx += 1
        pbar.close()
        logger.info(f"  ✓ Pollinations: {collected}")
        return collected

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class FastAICollector:
    def __init__(self, output_dir: Path = Path("./ai_generated_dataset")):
        self.output_dir = output_dir
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        self.all_metadata = []
        self.log = []

    def run(self, target: int = 17500) -> Dict:
        load_dotenv()
        civitai_key = os.getenv("CIVITAI_API_KEY")
        hf_token   = os.getenv("HUGGINGFACE_TOKEN")

        logger.info("="*60)
        logger.info(f"FAST AI COLLECTOR – Target: {target}")
        logger.info(f"CivitAI key : {'✓' if civitai_key else '✗'}")
        logger.info(f"HF token   : {'✓' if hf_token else '✗'}")
        logger.info("="*60)

        total = 0

        # Phase 1: CivitAI (40%)
        if civitai_key:
            c = CivitaiCollector(self.output_dir / "images", civitai_key)
            n = c.collect(int(target * 0.4))
            self.all_metadata.extend(c.metadata_list)
            self.log.append(("CivitAI", n))
            total += n

        # Phase 2: DiffusionDB (40%)
        if hf_token:
            c = DiffusionDBParquetCollector(self.output_dir / "images", hf_token)
            n = c.collect(int(target * 0.4))
            self.all_metadata.extend(c.metadata_list)
            self.log.append(("DiffusionDB", n))
            total += n

        # Phase 3: Pollinations fill the rest
        remaining = target - total
        if remaining > 0:
            c = PollinationsCollector(self.output_dir / "images")
            n = c.collect(remaining)
            self.all_metadata.extend(c.metadata_list)
            self.log.append(("Pollinations", n))
            total += n

        # Final report
        logger.info(f"✅ Total collected: {len(self.all_metadata)}")
        self._save()
        return {"total": len(self.all_metadata), "sources": dict(self.log)}

    def _save(self):
        df = pd.DataFrame([m.to_dict() for m in self.all_metadata])
        df.to_csv(self.output_dir / "metadata_ai.csv", index=False)
        with open(self.output_dir / "metadata_ai.json", "w") as f:
            json.dump([m.to_dict() for m in self.all_metadata], f, indent=2)
        logger.info(f"Metadata saved to {self.output_dir}")

if __name__ == "__main__":
    collector = FastAICollector(Path("./ai_generated_dataset"))
    report = collector.run(target=17500)
    print("Total collected:", report["total"])