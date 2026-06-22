"""
Collect more real images to reach ~25K total, then merge into dataset/.
Target: 7,100 new real images (to go from 17,903 → 25,000)
"""
import sys, json, csv, shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

from tqdm import tqdm
import pandas as pd
from PIL import Image
import io, hashlib, logging, os, time
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent.parent
REAL_DIR = PROJECT / 'dataset' / 'images' / 'real'
META_CSV = PROJECT / 'dataset' / 'metadata' / 'all.csv'


@dataclass
class ImageMeta:
    image_id: str; filename: str; source: str; source_url: str
    download_date: str; image_type: str = "real"
    width: int = 0; height: int = 0; format: str = ""
    file_size_kb: float = 0.0; color_space: str = ""
    md5_hash: str = ""; is_duplicate: bool = False; duplicate_of: Optional[str] = None

    def to_dict(self): return asdict(self)


class Collector:
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

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: List[ImageMeta] = []
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

    def download(self, url: str, image_id: str, source: str, query: str) -> bool:
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                data = r.content
                if len(data) < 5000:
                    return False
                h = hashlib.md5(data).hexdigest()
                filename = f"{image_id}.jpg"
                img = Image.open(io.BytesIO(data))
                w, h_dim = img.size
                fmt = img.format or ''
                cs = img.mode
                (self.output_dir / filename).write_bytes(data)
                self.metadata.append(ImageMeta(
                    image_id=image_id, filename=filename, source=source,
                    source_url=url, download_date=datetime.now().isoformat(),
                    image_type="real", width=w, height=h_dim, format=fmt,
                    file_size_kb=round(len(data)/1024, 2), color_space=cs,
                    md5_hash=h
                ))
                return True
            except Exception:
                if attempt == 2:
                    return False
                time.sleep(2)
        return False

    def collect_unsplash(self, api_key: str, target: int) -> int:
        n = 0; page = 1
        for query in self.QUERIES:
            while n < target:
                r = self.session.get(
                    "https://api.unsplash.com/search/photos",
                    headers={"Authorization": f"Client-ID {api_key}"},
                    params={"query": query, "per_page": 30, "page": page}, timeout=15
                )
                if r.status_code == 429:
                    time.sleep(60); continue
                if r.status_code != 200:
                    break
                results = r.json().get('results', [])
                if not results:
                    break
                for res in results:
                    if n >= target: break
                    url = res['urls'].get('regular') or res['urls']['full']
                    if self.download(url, f"UNSPLASH_{n}_{res['id']}", "unsplash", f"Query: {query}"):
                        n += 1
                page += 1
            page = 1
        return n

    def collect_pexels(self, api_key: str, target: int) -> int:
        n = 0; page = 1
        for query in self.QUERIES:
            while n < target:
                r = self.session.get(
                    "https://api.pexels.com/v1/search",
                    headers={"Authorization": api_key},
                    params={"query": query, "per_page": 80, "page": page}, timeout=15
                )
                if r.status_code == 429:
                    time.sleep(30); continue
                if r.status_code != 200:
                    break
                photos = r.json().get('photos', [])
                if not photos:
                    break
                for p in photos:
                    if n >= target: break
                    url = p['src'].get('large2x') or p['src']['large']
                    if self.download(url, f"PEXELS_{n}_{p['id']}", "pexels", f"Query: {query}"):
                        n += 1
                page += 1
            page = 1
        return n

    def collect_pixabay(self, api_key: str, target: int) -> int:
        n = 0; page = 1
        for query in self.QUERIES:
            while n < target:
                r = self.session.get(
                    "https://pixabay.com/api/",
                    params={"key": api_key, "q": query, "per_page": 200, "page": page, "image_type": "photo"},
                    timeout=15
                )
                if r.status_code != 200:
                    break
                hits = r.json().get('hits', [])
                if not hits:
                    break
                for h in hits:
                    if n >= target: break
                    url = h.get('largeImageURL', '')
                    if url and self.download(url, f"PIXABAY_{n}_{h['id']}", "pixabay", f"Query: {query}"):
                        n += 1
                page += 1
            page = 1
        return n


def main():
    import requests as _req
    import requests
    globals()['requests'] = _req

    existing = len(list(REAL_DIR.glob('*'))) if REAL_DIR.exists() else 0
    target_real = 25000
    needed = target_real - existing
    if needed <= 0:
        print(f"Already have {existing} real images (target {target_real}). Nothing to do.")
        return

    print(f"Current real images: {existing}")
    print(f"Target: {target_real}")
    print(f"Need to collect: {needed}\n")

    api_keys = {
        "UNSPLASH": os.getenv("UNSPLASH_API_KEY"),
        "PEXELS": os.getenv("PEXELS_API_KEY"),
        "PIXABAY": os.getenv("PIXABAY_API_KEY"),
    }

    active = [k for k, v in api_keys.items() if v]
    if not active:
        print("No API keys found! Check dataset/.env")
        return

    print(f"Using APIs: {', '.join(active)}")
    per_source = -(-needed // len(active))

    collector = Collector(REAL_DIR)
    total = 0
    if api_keys["UNSPLASH"]:
        try:
            n = collector.collect_unsplash(api_keys["UNSPLASH"], per_source)
        except Exception as e:
            n = 0; print(f"  Unsplash failed: {e}")
        print(f"  Unsplash: {n} images")
        total += n
    if api_keys["PEXELS"]:
        try:
            n = collector.collect_pexels(api_keys["PEXELS"], per_source)
        except Exception as e:
            n = 0; print(f"  Pexels failed: {e}")
        print(f"  Pexels:    {n} images")
        total += n
    if api_keys["PIXABAY"]:
        try:
            n = collector.collect_pixabay(api_keys["PIXABAY"], per_source)
        except Exception as e:
            n = 0; print(f"  Pixabay failed: {e}")
        print(f"  Pixabay:   {n} images")
        total += n

    print(f"\nTotal collected: {total}")

    if collector.metadata:
        df_new = pd.DataFrame([m.to_dict() for m in collector.metadata])
        if META_CSV.exists():
            df_old = pd.read_csv(META_CSV)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(META_CSV, index=False)
        print(f"Updated metadata: {META_CSV}")

    final_count = len(list(REAL_DIR.glob('*')))
    print(f"\nFinal real image count: {final_count}")


if __name__ == '__main__':
    main()
