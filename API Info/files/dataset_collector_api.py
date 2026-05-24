"""
AI Image Dataset Collector API
================================
Automated tool to collect real and AI-generated images with comprehensive metadata tracking
for research on AI-generated image detection.

Installation:
pip install requests pandas pillow python-dotenv aiohttp tqdm beautifulsoup4 huggingface_hub

API Keys Required:
- UNSPLASH_API_KEY (get from: https://unsplash.com/oauth/applications)
- PEXELS_API_KEY (get from: https://www.pexels.com/api/)
- PIXABAY_API_KEY (get from: https://pixabay.com/api/docs/)
- HUGGINGFACE_TOKEN (get from: https://huggingface.co/settings/tokens)
"""

import os
import json
import csv
import asyncio
import aiohttp
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
from urllib.parse import urljoin
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

class ImageType(Enum):
    """Image classification type"""
    REAL = "real"
    AI_GENERATED = "ai_generated"
    UNKNOWN = "unknown"

class ImageSource(Enum):
    """Source of the image"""
    UNSPLASH = "unsplash"
    PEXELS = "pexels"
    PIXABAY = "pixabay"
    HUGGINGFACE = "huggingface"
    MANUAL = "manual"
    DALLE = "dalle"
    MIDJOURNEY = "midjourney"
    STABLE_DIFFUSION = "stable_diffusion"

@dataclass
class ImageMetadata:
    """Comprehensive metadata for each image"""
    image_id: str  # Unique identifier
    filename: str  # Saved filename
    source: str  # Source platform
    source_url: str  # Original URL
    download_date: str  # When downloaded (ISO format)
    image_type: str  # real or ai_generated
    
    # Image characteristics
    width: int
    height: int
    format: str  # jpg, png, etc
    file_size_kb: float
    color_space: str  # RGB, RGBA, etc
    
    # Source metadata
    source_metadata: Dict  # Original source metadata
    
    # Verification info
    md5_hash: str  # For duplicate detection
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    
    # Processing info
    processed: bool = False
    analysis_notes: str = ""
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)


# ============================================================================
# REAL IMAGE COLLECTORS
# ============================================================================

class RealImageCollector:
    """Base class for collecting real images"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "real_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_list: List[ImageMetadata] = []
    
    def _calculate_hash(self, image_data: bytes) -> str:
        """Calculate MD5 hash of image data"""
        return hashlib.md5(image_data).hexdigest()
    
    def _get_image_info(self, image_data: bytes) -> Tuple[int, int, str, str]:
        """Extract image dimensions and format"""
        try:
            img = Image.open(io.BytesIO(image_data))
            width, height = img.size
            image_format = img.format or 'UNKNOWN'
            color_space = img.mode
            return width, height, image_format, color_space
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return 0, 0, "UNKNOWN", "UNKNOWN"
    
    def _save_image(self, image_data: bytes, filename: str) -> bool:
        """Save image to disk"""
        try:
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                f.write(image_data)
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
    
    def _check_duplicate(self, image_hash: str) -> Optional[str]:
        """Check if image already exists (by hash)"""
        for metadata in self.metadata_list:
            if metadata.md5_hash == image_hash:
                return metadata.image_id
        return None
    
    def save_metadata(self, output_file: Path):
        """Save metadata to CSV"""
        if not self.metadata_list:
            logger.warning("No metadata to save")
            return
        
        df = pd.DataFrame([m.to_dict() for m in self.metadata_list])
        df.to_csv(output_file, index=False)
        logger.info(f"Metadata saved to {output_file}")


class UnsplashCollector(RealImageCollector):
    """Collect real images from Unsplash"""
    
    def __init__(self, api_key: str, output_dir: Path):
        super().__init__(output_dir)
        self.api_key = api_key
        self.base_url = "https://api.unsplash.com"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Client-ID {api_key}"})
    
    def collect(self, query: str, count: int = 30) -> int:
        """
        Collect images from Unsplash
        
        Args:
            query: Search query (e.g., "portrait", "landscape", "nature")
            count: Number of images to collect
        
        Returns:
            Number of images successfully collected
        """
        logger.info(f"Collecting {count} images from Unsplash with query: '{query}'")
        
        collected = 0
        page = 1
        per_page = 30
        
        while collected < count:
            try:
                # Request images
                response = self.session.get(
                    f"{self.base_url}/search/photos",
                    params={
                        "query": query,
                        "per_page": per_page,
                        "page": page,
                        "orientation": "landscape"
                    }
                )
                response.raise_for_status()
                results = response.json()
                
                if not results.get('results'):
                    logger.info(f"No more results available for query: '{query}'")
                    break
                
                # Download each image
                for result in tqdm(results['results'], desc=f"Unsplash - Page {page}"):
                    if collected >= count:
                        break
                    
                    try:
                        # Download image
                        image_url = result['urls']['full']
                        image_response = self.session.get(image_url, timeout=10)
                        image_response.raise_for_status()
                        image_data = image_response.content
                        
                        # Calculate hash and check for duplicates
                        image_hash = self._calculate_hash(image_data)
                        duplicate_of = self._check_duplicate(image_hash)
                        
                        # Create filename
                        image_id = f"UNSPLASH_{result['id']}"
                        filename = f"{image_id}.jpg"
                        
                        # Get image info
                        width, height, fmt, color_space = self._get_image_info(image_data)
                        
                        # Save image
                        if self._save_image(image_data, filename):
                            # Create metadata
                            metadata = ImageMetadata(
                                image_id=image_id,
                                filename=filename,
                                source=ImageSource.UNSPLASH.value,
                                source_url=result['links']['html'],
                                download_date=datetime.now().isoformat(),
                                image_type=ImageType.REAL.value,
                                width=width,
                                height=height,
                                format=fmt,
                                file_size_kb=len(image_data) / 1024,
                                color_space=color_space,
                                source_metadata={
                                    "author": result['user']['name'],
                                    "description": result.get('description', ''),
                                    "likes": result['likes'],
                                },
                                md5_hash=image_hash,
                                is_duplicate=duplicate_of is not None,
                                duplicate_of=duplicate_of,
                                analysis_notes=f"Query: {query}"
                            )
                            self.metadata_list.append(metadata)
                            collected += 1
                            logger.info(f"✓ Collected {collected}/{count}: {image_id}")
                    
                    except Exception as e:
                        logger.error(f"Error processing Unsplash image: {e}")
                        continue
                
                page += 1
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching from Unsplash: {e}")
                break
        
        return collected


class PexelsCollector(RealImageCollector):
    """Collect real images from Pexels"""
    
    def __init__(self, api_key: str, output_dir: Path):
        super().__init__(output_dir)
        self.api_key = api_key
        self.base_url = "https://api.pexels.com/v1"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})
    
    def collect(self, query: str, count: int = 30) -> int:
        """Collect images from Pexels"""
        logger.info(f"Collecting {count} images from Pexels with query: '{query}'")
        
        collected = 0
        page = 1
        per_page = 80  # Pexels allows up to 80 per page
        
        while collected < count:
            try:
                response = self.session.get(
                    f"{self.base_url}/search",
                    params={
                        "query": query,
                        "per_page": per_page,
                        "page": page,
                        "orientation": "landscape"
                    },
                    timeout=10
                )
                response.raise_for_status()
                results = response.json()
                
                if not results.get('photos'):
                    logger.info(f"No more results available for query: '{query}'")
                    break
                
                for photo in tqdm(results['photos'], desc=f"Pexels - Page {page}"):
                    if collected >= count:
                        break
                    
                    try:
                        # Download image
                        image_url = photo['src']['large']
                        image_response = self.session.get(image_url, timeout=10)
                        image_response.raise_for_status()
                        image_data = image_response.content
                        
                        # Calculate hash
                        image_hash = self._calculate_hash(image_data)
                        duplicate_of = self._check_duplicate(image_hash)
                        
                        # Create filename
                        image_id = f"PEXELS_{photo['id']}"
                        filename = f"{image_id}.jpg"
                        
                        # Get image info
                        width, height, fmt, color_space = self._get_image_info(image_data)
                        
                        # Save image
                        if self._save_image(image_data, filename):
                            metadata = ImageMetadata(
                                image_id=image_id,
                                filename=filename,
                                source=ImageSource.PEXELS.value,
                                source_url=photo['url'],
                                download_date=datetime.now().isoformat(),
                                image_type=ImageType.REAL.value,
                                width=width,
                                height=height,
                                format=fmt,
                                file_size_kb=len(image_data) / 1024,
                                color_space=color_space,
                                source_metadata={
                                    "photographer": photo['photographer'],
                                    "photographer_url": photo['photographer_url'],
                                },
                                md5_hash=image_hash,
                                is_duplicate=duplicate_of is not None,
                                duplicate_of=duplicate_of,
                                analysis_notes=f"Query: {query}"
                            )
                            self.metadata_list.append(metadata)
                            collected += 1
                            logger.info(f"✓ Collected {collected}/{count}: {image_id}")
                    
                    except Exception as e:
                        logger.error(f"Error processing Pexels image: {e}")
                        continue
                
                page += 1
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching from Pexels: {e}")
                break
        
        return collected


class PixabayCollector(RealImageCollector):
    """Collect real images from Pixabay"""
    
    def __init__(self, api_key: str, output_dir: Path):
        super().__init__(output_dir)
        self.api_key = api_key
        self.base_url = "https://pixabay.com/api"
    
    def collect(self, query: str, count: int = 30) -> int:
        """Collect images from Pixabay"""
        logger.info(f"Collecting {count} images from Pixabay with query: '{query}'")
        
        collected = 0
        page = 1
        per_page = 200  # Pixabay allows up to 200 per page
        
        while collected < count:
            try:
                response = requests.get(
                    self.base_url,
                    params={
                        "key": self.api_key,
                        "q": query,
                        "per_page": per_page,
                        "page": page,
                        "image_type": "photo"
                    },
                    timeout=10
                )
                response.raise_for_status()
                results = response.json()
                
                if not results.get('hits'):
                    logger.info(f"No more results available for query: '{query}'")
                    break
                
                for hit in tqdm(results['hits'], desc=f"Pixabay - Page {page}"):
                    if collected >= count:
                        break
                    
                    try:
                        # Download image
                        image_url = hit['largeImageURL']
                        image_response = requests.get(image_url, timeout=10)
                        image_response.raise_for_status()
                        image_data = image_response.content
                        
                        # Calculate hash
                        image_hash = self._calculate_hash(image_data)
                        duplicate_of = self._check_duplicate(image_hash)
                        
                        # Create filename
                        image_id = f"PIXABAY_{hit['id']}"
                        filename = f"{image_id}.jpg"
                        
                        # Get image info
                        width, height, fmt, color_space = self._get_image_info(image_data)
                        
                        # Save image
                        if self._save_image(image_data, filename):
                            metadata = ImageMetadata(
                                image_id=image_id,
                                filename=filename,
                                source=ImageSource.PIXABAY.value,
                                source_url=hit['pageURL'],
                                download_date=datetime.now().isoformat(),
                                image_type=ImageType.REAL.value,
                                width=width,
                                height=height,
                                format=fmt,
                                file_size_kb=len(image_data) / 1024,
                                color_space=color_space,
                                source_metadata={
                                    "user": hit['user'],
                                    "likes": hit['likes'],
                                    "downloads": hit['downloads'],
                                },
                                md5_hash=image_hash,
                                is_duplicate=duplicate_of is not None,
                                duplicate_of=duplicate_of,
                                analysis_notes=f"Query: {query}"
                            )
                            self.metadata_list.append(metadata)
                            collected += 1
                            logger.info(f"✓ Collected {collected}/{count}: {image_id}")
                    
                    except Exception as e:
                        logger.error(f"Error processing Pixabay image: {e}")
                        continue
                
                page += 1
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching from Pixabay: {e}")
                break
        
        return collected


# ============================================================================
# AI-GENERATED IMAGE COLLECTORS
# ============================================================================

class AIGeneratedImageCollector(RealImageCollector):
    """Base class for collecting AI-generated images"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "ai_generated_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_list: List[ImageMetadata] = []


class StableDiffusionCollector(AIGeneratedImageCollector):
    """Collect Stable Diffusion generated images from HuggingFace"""
    
    def __init__(self, output_dir: Path, hf_token: Optional[str] = None):
        super().__init__(output_dir)
        self.hf_token = hf_token
        self.datasets = [
            "jbilcke-jw/real-photos",  # Actually real, but good for comparison
            "multimodalart/facesyntheticssimple",  # AI-generated faces
        ]
    
    def collect_from_dataset(self, dataset_name: str, count: int = 50) -> int:
        """
        Collect images from HuggingFace dataset
        
        Args:
            dataset_name: HuggingFace dataset name
            count: Number of images to collect
        
        Returns:
            Number of images collected
        """
        logger.info(f"Collecting {count} images from HuggingFace dataset: {dataset_name}")
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            collected = 0
            
            for idx, example in enumerate(dataset):
                if collected >= count:
                    break
                
                try:
                    # Extract image
                    if 'image' in example:
                        pil_image = example['image']
                    else:
                        logger.warning(f"No 'image' field in dataset example {idx}")
                        continue
                    
                    # Convert to bytes
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    image_data = img_byte_arr.getvalue()
                    
                    # Calculate hash
                    image_hash = self._calculate_hash(image_data)
                    duplicate_of = self._check_duplicate(image_hash)
                    
                    # Create filename
                    image_id = f"STABLE_DIFFUSION_HF_{dataset_name}_{idx}"
                    filename = f"{image_id}.png"
                    
                    # Get image info
                    width, height, fmt, color_space = self._get_image_info(image_data)
                    
                    # Save image
                    if self._save_image(image_data, filename):
                        metadata = ImageMetadata(
                            image_id=image_id,
                            filename=filename,
                            source=ImageSource.STABLE_DIFFUSION.value,
                            source_url=f"huggingface.co/datasets/{dataset_name}",
                            download_date=datetime.now().isoformat(),
                            image_type=ImageType.AI_GENERATED.value,
                            width=width,
                            height=height,
                            format=fmt,
                            file_size_kb=len(image_data) / 1024,
                            color_space=color_space,
                            source_metadata={
                                "dataset": dataset_name,
                                "index": idx,
                            },
                            md5_hash=image_hash,
                            is_duplicate=duplicate_of is not None,
                            duplicate_of=duplicate_of,
                            analysis_notes=f"HuggingFace Dataset: {dataset_name}"
                        )
                        self.metadata_list.append(metadata)
                        collected += 1
                        logger.info(f"✓ Collected {collected}/{count}: {image_id}")
                
                except Exception as e:
                    logger.error(f"Error processing dataset example {idx}: {e}")
                    continue
            
            return collected
        
        except ImportError:
            logger.error("Please install 'datasets' package: pip install datasets")
            return 0
        except Exception as e:
            logger.error(f"Error collecting from HuggingFace: {e}")
            return 0


class LocalFolderCollector(RealImageCollector):
    """Collect images from local folders"""
    
    def collect_from_folder(self, folder_path: Path, image_type: str = "real") -> int:
        """
        Collect images from a local folder
        
        Args:
            folder_path: Path to folder containing images
            image_type: "real" or "ai_generated"
        
        Returns:
            Number of images collected
        """
        logger.info(f"Collecting images from folder: {folder_path}")
        
        if not folder_path.exists():
            logger.error(f"Folder not found: {folder_path}")
            return 0
        
        collected = 0
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        
        image_files = [
            f for f in folder_path.iterdir()
            if f.suffix.lower() in supported_formats
        ]
        
        for image_file in tqdm(image_files, desc=f"Local folder: {folder_path.name}"):
            try:
                # Read image
                with open(image_file, 'rb') as f:
                    image_data = f.read()
                
                # Calculate hash
                image_hash = self._calculate_hash(image_data)
                duplicate_of = self._check_duplicate(image_hash)
                
                # Create filename
                image_id = f"LOCAL_{image_file.stem}_{collected}"
                filename = f"{image_id}{image_file.suffix.lower()}"
                
                # Get image info
                width, height, fmt, color_space = self._get_image_info(image_data)
                
                # Copy to output directory
                if self._save_image(image_data, filename):
                    metadata = ImageMetadata(
                        image_id=image_id,
                        filename=filename,
                        source=ImageSource.MANUAL.value,
                        source_url=str(image_file),
                        download_date=datetime.now().isoformat(),
                        image_type=image_type,
                        width=width,
                        height=height,
                        format=fmt,
                        file_size_kb=len(image_data) / 1024,
                        color_space=color_space,
                        source_metadata={
                            "original_filename": image_file.name,
                            "original_path": str(image_file),
                        },
                        md5_hash=image_hash,
                        is_duplicate=duplicate_of is not None,
                        duplicate_of=duplicate_of,
                        analysis_notes=f"Imported from local folder"
                    )
                    self.metadata_list.append(metadata)
                    collected += 1
                    logger.info(f"✓ Collected {collected}: {image_id}")
            
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                continue
        
        return collected


# ============================================================================
# MAIN DATASET ORCHESTRATOR
# ============================================================================

class DatasetCollector:
    """Main orchestrator for collecting full dataset"""
    
    def __init__(self, output_dir: Path = Path("./dataset")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_metadata: List[ImageMetadata] = []
        self.collection_log = {
            "start_time": datetime.now().isoformat(),
            "collections": []
        }
    
    def load_api_keys(self, env_file: Path = Path(".env")) -> Dict[str, str]:
        """Load API keys from .env file"""
        from dotenv import load_dotenv
        
        load_dotenv(env_file)
        
        keys = {
            "UNSPLASH": os.getenv("UNSPLASH_API_KEY"),
            "PEXELS": os.getenv("PEXELS_API_KEY"),
            "PIXABAY": os.getenv("PIXABAY_API_KEY"),
            "HUGGINGFACE": os.getenv("HUGGINGFACE_TOKEN"),
        }
        
        for key_name, key_value in keys.items():
            if key_value:
                logger.info(f"✓ Loaded {key_name} API key")
            else:
                logger.warning(f"⚠ Missing {key_name} API key")
        
        return keys
    
    def collect_real_images(self, api_keys: Dict[str, str], total_count: int = 250):
        """Collect real images from multiple sources"""
        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTING REAL IMAGES (Target: {total_count})")
        logger.info(f"{'='*60}\n")
        
        per_source = total_count // 3  # Distribute equally
        queries = [
            "portrait",
            "landscape",
            "objects",
            "urban",
            "nature",
            "people",
            "animals",
            "food",
            "architecture",
            "abstract"
        ]
        
        total_collected = 0
        
        # Unsplash
        if api_keys.get("UNSPLASH"):
            try:
                unsplash = UnsplashCollector(api_keys["UNSPLASH"], self.output_dir)
                for query in queries[:3]:
                    collected = unsplash.collect(query, count=per_source // 3)
                    total_collected += collected
                self.all_metadata.extend(unsplash.metadata_list)
                self.collection_log["collections"].append({
                    "source": "Unsplash",
                    "count": len(unsplash.metadata_list)
                })
            except Exception as e:
                logger.error(f"Unsplash collection failed: {e}")
        
        # Pexels
        if api_keys.get("PEXELS"):
            try:
                pexels = PexelsCollector(api_keys["PEXELS"], self.output_dir)
                for query in queries[3:6]:
                    collected = pexels.collect(query, count=per_source // 3)
                    total_collected += collected
                self.all_metadata.extend(pexels.metadata_list)
                self.collection_log["collections"].append({
                    "source": "Pexels",
                    "count": len(pexels.metadata_list)
                })
            except Exception as e:
                logger.error(f"Pexels collection failed: {e}")
        
        # Pixabay
        if api_keys.get("PIXABAY"):
            try:
                pixabay = PixabayCollector(api_keys["PIXABAY"], self.output_dir)
                for query in queries[6:]:
                    collected = pixabay.collect(query, count=per_source // 3)
                    total_collected += collected
                self.all_metadata.extend(pixabay.metadata_list)
                self.collection_log["collections"].append({
                    "source": "Pixabay",
                    "count": len(pixabay.metadata_list)
                })
            except Exception as e:
                logger.error(f"Pixabay collection failed: {e}")
        
        logger.info(f"\n✓ Total real images collected: {total_collected}")
        return total_collected
    
    def collect_ai_images(self, api_keys: Dict[str, str], total_count: int = 250):
        """Collect AI-generated images"""
        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTING AI-GENERATED IMAGES (Target: {total_count})")
        logger.info(f"{'='*60}\n")
        
        total_collected = 0
        
        # Stable Diffusion from HuggingFace
        try:
            sd_collector = StableDiffusionCollector(
                self.output_dir,
                api_keys.get("HUGGINGFACE")
            )
            
            datasets = [
                "multimodalart/facesyntheticssimple",
                "jbilcke-jw/ai-generated-anime",
            ]
            
            for dataset in datasets:
                try:
                    collected = sd_collector.collect_from_dataset(dataset, count=total_count // 2)
                    total_collected += collected
                except Exception as e:
                    logger.error(f"Error collecting from {dataset}: {e}")
            
            self.all_metadata.extend(sd_collector.metadata_list)
            self.collection_log["collections"].append({
                "source": "Stable Diffusion (HuggingFace)",
                "count": len(sd_collector.metadata_list)
            })
        
        except Exception as e:
            logger.error(f"Stable Diffusion collection failed: {e}")
        
        logger.info(f"\n✓ Total AI images collected: {total_collected}")
        return total_collected
    
    def collect_from_local_folders(self, folders: Dict[str, Path]):
        """
        Collect images from local folders
        
        Args:
            folders: Dict mapping image_type to folder path
                     e.g., {"real": Path("./my_real_images"), "ai_generated": Path("./my_ai_images")}
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTING FROM LOCAL FOLDERS")
        logger.info(f"{'='*60}\n")
        
        total_collected = 0
        
        for image_type, folder_path in folders.items():
            try:
                collector = LocalFolderCollector(self.output_dir)
                collected = collector.collect_from_folder(
                    Path(folder_path),
                    image_type=image_type
                )
                total_collected += collected
                self.all_metadata.extend(collector.metadata_list)
                self.collection_log["collections"].append({
                    "source": f"Local Folder: {folder_path.name}",
                    "type": image_type,
                    "count": len(collector.metadata_list)
                })
            except Exception as e:
                logger.error(f"Error collecting from {folder_path}: {e}")
        
        logger.info(f"\n✓ Total local images collected: {total_collected}")
        return total_collected
    
    def generate_dataset_report(self):
        """Generate comprehensive dataset report"""
        logger.info(f"\n{'='*60}")
        logger.info(f"DATASET COLLECTION COMPLETE")
        logger.info(f"{'='*60}\n")
        
        report = {
            "total_images": len(self.all_metadata),
            "real_images": len([m for m in self.all_metadata if m.image_type == ImageType.REAL.value]),
            "ai_generated_images": len([m for m in self.all_metadata if m.image_type == ImageType.AI_GENERATED.value]),
            "duplicates_found": len([m for m in self.all_metadata if m.is_duplicate]),
            "average_file_size_kb": sum(m.file_size_kb for m in self.all_metadata) / len(self.all_metadata) if self.all_metadata else 0,
            "by_source": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Count by source
        for metadata in self.all_metadata:
            source = metadata.source
            if source not in report["by_source"]:
                report["by_source"][source] = 0
            report["by_source"][source] += 1
        
        logger.info(f"Total Images Collected: {report['total_images']}")
        logger.info(f"  - Real: {report['real_images']}")
        logger.info(f"  - AI-Generated: {report['ai_generated_images']}")
        logger.info(f"  - Duplicates Found: {report['duplicates_found']}")
        logger.info(f"  - Average File Size: {report['average_file_size_kb']:.2f} KB")
        logger.info(f"\nBy Source:")
        for source, count in report["by_source"].items():
            logger.info(f"  - {source}: {count}")
        
        return report
    
    def save_all_metadata(self):
        """Save all metadata to CSV and JSON"""
        logger.info(f"\nSaving metadata...")
        
        # Save to CSV
        csv_file = self.output_dir / "dataset_metadata.csv"
        df = pd.DataFrame([m.to_dict() for m in self.all_metadata])
        df.to_csv(csv_file, index=False)
        logger.info(f"✓ Metadata saved to {csv_file}")
        
        # Save to JSON
        json_file = self.output_dir / "dataset_metadata.json"
        with open(json_file, 'w') as f:
            json.dump([m.to_dict() for m in self.all_metadata], f, indent=2)
        logger.info(f"✓ Metadata saved to {json_file}")
        
        # Save collection log
        self.collection_log["end_time"] = datetime.now().isoformat()
        log_file = self.output_dir / "collection_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.collection_log, f, indent=2)
        logger.info(f"✓ Collection log saved to {log_file}")
    
    def run_full_collection(self, config: Dict):
        """
        Run complete dataset collection
        
        Args:
            config: Configuration dict with:
                - real_image_count: Number of real images to collect
                - ai_image_count: Number of AI images to collect
                - local_folders: Dict of local folders to import
                - api_keys: Dict of API keys (or load from .env)
        """
        # Load API keys
        api_keys = config.get("api_keys") or self.load_api_keys()
        
        # Collect real images
        self.collect_real_images(api_keys, config.get("real_image_count", 250))
        
        # Collect AI images
        self.collect_ai_images(api_keys, config.get("ai_image_count", 250))
        
        # Collect from local folders if provided
        if config.get("local_folders"):
            self.collect_from_local_folders(config["local_folders"])
        
        # Generate report
        report = self.generate_dataset_report()
        
        # Save metadata
        self.save_all_metadata()
        
        return report


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the DatasetCollector
    
    Before running, create a .env file with:
    UNSPLASH_API_KEY=your_key
    PEXELS_API_KEY=your_key
    PIXABAY_API_KEY=your_key
    HUGGINGFACE_TOKEN=your_token
    """
    
    # Configuration
    config = {
        "real_image_count": 250,
        "ai_image_count": 250,
        "local_folders": {
            # "real": Path("./my_real_images"),
            # "ai_generated": Path("./my_ai_images"),
        }
    }
    
    # Create collector
    collector = DatasetCollector(output_dir=Path("./ai_detection_dataset"))
    
    # Run collection
    report = collector.run_full_collection(config)
    
    print("\n" + "="*60)
    print("COLLECTION COMPLETE!")
    print("="*60)
    print(f"Dataset saved to: ./ai_detection_dataset")
    print(f"Total images: {report['total_images']}")
