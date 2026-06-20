"""
================================================================================
1️⃣  REAL IMAGES COLLECTOR
================================================================================
Focus: Collect diverse real images from Unsplash, Pexels, Pixabay

Target: 15,000 real images

Output:
    real_dataset/
    ├── images/ (15,000 images)
    ├── metadata/
    │   ├── dataset_metadata_real.csv
    │   └── dataset_metadata_real.json
    └── collection_log.json
"""

import os, json, time, io, hashlib, logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class ImageSource(Enum):
    UNSPLASH = "unsplash"
    PEXELS   = "pexels"
    PIXABAY  = "pixabay"

@dataclass
class ImageMetadata:
    image_id: str
    filename: str
    source: str
    source_url: str
    download_date: str
    image_type: str = "real"
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


# ============================================================================
# BASE COLLECTOR
# ============================================================================

class BaseCollector:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "images"
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

    def _download_and_record(self, url: str, image_id: str, source_val: str,
                              source_meta: dict, note: str) -> bool:
        try:
            r = requests.get(url, headers=self.session.headers, timeout=25)
            r.raise_for_status()
            data = r.content

            if len(data) < 5000:
                return False

            h = self._calculate_hash(data)
            dup = self._check_duplicate(h)
            filename = f"{image_id}.jpg"
            w, ht, fmt, cs = self._get_image_info(data)

            if not self._save_image(data, filename):
                return False

            self.metadata_list.append(ImageMetadata(
                image_id=image_id,
                filename=filename,
                source=source_val,
                source_url=url,
                download_date=datetime.now().isoformat(),
                image_type="real",
                width=w,
                height=ht,
                format=fmt,
                file_size_kb=round(len(data) / 1024, 2),
                color_space=cs,
                source_metadata=source_meta,
                md5_hash=h,
                is_duplicate=dup is not None,
                duplicate_of=dup,
                analysis_notes=note
            ))
            return True
        except Exception as e:
            logger.debug(f"Download failed ({image_id}): {e}")
            return False


# ============================================================================
# REAL IMAGE COLLECTORS
# ============================================================================

class UnsplashCollector(BaseCollector):
    def __init__(self, api_key: str, output_dir: Path):
        super().__init__(output_dir)
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
                    time.sleep(60)
                    continue

                r.raise_for_status()
                results = r.json().get('results', [])

                if not results:
                    q_idx += 1
                    page = 1
                    continue

                for res in results:
                    if collected >= total_count:
                        break
                    url = res['urls'].get('regular') or res['urls']['full']
                    if self._download_and_record(
                        url,
                        f"UNSPLASH_{collected}_{res['id']}",
                        ImageSource.UNSPLASH.value,
                        {"author": res['user']['name']},
                        f"Query: {query}"):
                        collected += 1

                page += 1
            except Exception as e:
                logger.warning(f"Unsplash error '{query}': {e}")
                q_idx += 1
                page = 1

        logger.info(f"  ✓ Unsplash: {collected} images")
        return collected


class PexelsCollector(BaseCollector):
    def __init__(self, api_key: str, output_dir: Path):
        super().__init__(output_dir)
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
                    time.sleep(30)
                    continue

                r.raise_for_status()
                photos = r.json().get('photos', [])

                if not photos:
                    q_idx += 1
                    page = 1
                    continue

                for p in photos:
                    if collected >= total_count:
                        break
                    url = p['src'].get('large2x') or p['src']['large']
                    if self._download_and_record(
                        url,
                        f"PEXELS_{collected}_{p['id']}",
                        ImageSource.PEXELS.value,
                        {"photographer": p['photographer']},
                        f"Query: {query}"):
                        collected += 1

                page += 1
            except Exception as e:
                logger.warning(f"Pexels error '{query}': {e}")
                q_idx += 1
                page = 1

        logger.info(f"  ✓ Pexels: {collected} images")
        return collected


class PixabayCollector(BaseCollector):
    def __init__(self, api_key: str, output_dir: Path):
        super().__init__(output_dir)
        self.api_key = api_key

    def collect(self, queries: List[str], total_count: int) -> int:
        logger.info(f"Pixabay: targeting {total_count} images")
        collected, q_idx, page = 0, 0, 1

        while collected < total_count and q_idx < len(queries):
            query = queries[q_idx]
            try:
                r = self.session.get(
                    "https://pixabay.com/api/",
                    params={
                        "key": self.api_key,
                        "q": query,
                        "per_page": 200,
                        "page": page,
                        "image_type": "photo"
                    },
                    timeout=15)

                r.raise_for_status()
                hits = r.json().get('hits', [])

                if not hits:
                    q_idx += 1
                    page = 1
                    continue

                for h in hits:
                    if collected >= total_count:
                        break
                    url = h.get('largeImageURL', '')
                    if not url:
                        continue
                    if self._download_and_record(
                        url,
                        f"PIXABAY_{collected}_{h['id']}",
                        ImageSource.PIXABAY.value,
                        {"user": h['user']},
                        f"Query: {query}"):
                        collected += 1

                page += 1
            except Exception as e:
                logger.warning(f"Pixabay error '{query}': {e}")
                q_idx += 1
                page = 1

        logger.info(f"  ✓ Pixabay: {collected} images")
        return collected


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class RealImageCollector:
    QUERIES = [
        "portrait face", "landscape nature", "person people",
        "mountain scenery", "food photography", "street urban",
        "beach water", "animal wildlife", "city architecture",
        "forest trees", "sunset sky", "family group",
        "sport action", "flower plant", "rain weather",
        "desert landscape", "winter snow", "vehicle car",
        "indoor room", "office workspace", "sports game",
        "beach ocean", "mountain peak", "city lights",
        "nature park", "animal dog", "animal cat",
        "bird flying", "tree forest", "river water",
        "sunset sunrise", "night city", "day light",
    ]

    def __init__(self, output_dir: Path = Path("./real_dataset")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_metadata: List[ImageMetadata] = []
        self.collection_log = {
            "start_time": datetime.now().isoformat(),
            "collections": [],
            "config": {}
        }

    def load_api_keys(self) -> Dict[str, str]:
        load_dotenv()
        keys = {
            "UNSPLASH": os.getenv("UNSPLASH_API_KEY"),
            "PEXELS":   os.getenv("PEXELS_API_KEY"),
            "PIXABAY":  os.getenv("PIXABAY_API_KEY"),
        }
        for k, v in keys.items():
            status = "✓ loaded" if v else "✗ MISSING"
            logger.info(f"  {k:15}: {status}")
        return keys

    def collect(self, api_keys: Dict[str, str], target: int = 15000) -> int:
        logger.info(f"\n{'='*70}\nCOLLECTING REAL IMAGES (Target: {target})\n{'='*70}\n")

        active = [k for k in ("UNSPLASH", "PEXELS", "PIXABAY") if api_keys.get(k)]
        if not active:
            logger.error("No real-image API keys found!")
            return 0

        per_source = -(-target // len(active))
        total = 0

        # Unsplash
        if api_keys.get("UNSPLASH"):
            c = UnsplashCollector(api_keys["UNSPLASH"], self.output_dir)
            n = c.collect(self.QUERIES, per_source)
            self.all_metadata.extend(c.metadata_list)
            self.collection_log["collections"].append({"source": "Unsplash", "count": n})
            total += n

        # Pexels
        if api_keys.get("PEXELS"):
            c = PexelsCollector(api_keys["PEXELS"], self.output_dir)
            n = c.collect(self.QUERIES, per_source)
            self.all_metadata.extend(c.metadata_list)
            self.collection_log["collections"].append({"source": "Pexels", "count": n})
            total += n

        # Pixabay
        if api_keys.get("PIXABAY"):
            c = PixabayCollector(api_keys["PIXABAY"], self.output_dir)
            n = c.collect(self.QUERIES, per_source)
            self.all_metadata.extend(c.metadata_list)
            self.collection_log["collections"].append({"source": "Pixabay", "count": n})
            total += n

        logger.info(f"\n✓ Real images collected: {total} / {target}\n")
        return total

    def generate_report(self) -> Dict:
        report = {
            "total_images": len(self.all_metadata),
            "by_source": {},
            "timestamp": datetime.now().isoformat()
        }
        for m in self.all_metadata:
            report["by_source"][m.source] = report["by_source"].get(m.source, 0) + 1

        logger.info(f"\n{'='*70}\nDATASET SUMMARY\n{'='*70}")
        logger.info(f"Total images: {report['total_images']:,}")
        for src, cnt in sorted(report["by_source"].items()):
            pct = cnt / max(report['total_images'], 1) * 100
            logger.info(f"  {src:.<45} {cnt:>6,} ({pct:>5.1f}%)")
        return report

    def save_metadata(self):
        logger.info("\nSaving metadata…")
        (self.output_dir / "metadata").mkdir(exist_ok=True)

        if self.all_metadata:
            df = pd.DataFrame([m.to_dict() for m in self.all_metadata])
            csv_path = self.output_dir / "metadata" / "dataset_metadata_real.csv"
            json_path = self.output_dir / "metadata" / "dataset_metadata_real.json"

            df.to_csv(csv_path, index=False)
            with open(json_path, 'w') as f:
                json.dump([m.to_dict() for m in self.all_metadata], f, indent=2)

            logger.info(f"  ✓ CSV  → {csv_path}")
            logger.info(f"  ✓ JSON → {json_path}")

        self.collection_log["end_time"] = datetime.now().isoformat()
        with open(self.output_dir / "metadata" / "collection_log.json", 'w') as f:
            json.dump(self.collection_log, f, indent=2)
        logger.info(f"  ✓ Log  → collection_log.json\n")

    def run(self) -> Dict:
        self.collection_log["config"] = {"target": 15000}
        api_keys = self.load_api_keys()
        self.collect(api_keys, 15000)
        report = self.generate_report()
        self.save_metadata()
        return report


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("🖼️  REAL IMAGES COLLECTOR - 15,000 Images")
    logger.info("="*70 + "\n")

    collector = RealImageCollector(output_dir=Path("./real_dataset"))
    report = collector.run()

    logger.info("="*70)
    logger.info("✓ REAL IMAGES COLLECTION COMPLETE!")
    logger.info("="*70)
    logger.info(f"Total: {report['total_images']:,} images")
    logger.info(f"Output: ./real_dataset/\n")
