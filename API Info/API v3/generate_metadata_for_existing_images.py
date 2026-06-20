"""
================================================================================
GENERATE METADATA FOR EXISTING IMAGES
================================================================================
Use this if you already have real images downloaded but no metadata.

This script:
1. Scans your existing image folder
2. Extracts image info (width, height, size, format)
3. Calculates MD5 hash for deduplication
4. Creates CSV + JSON metadata files

Usage:
    python 0_generate_metadata_for_existing_images.py \
        --input-dir ./your_existing_images/ \
        --output-dir ./real_dataset_with_metadata/
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import io

import pandas as pd
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

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
    duplicate_of: str = None
    analysis_notes: str = ""

    def __post_init__(self):
        if self.source_metadata is None:
            self.source_metadata = {}

    def to_dict(self):
        return asdict(self)


# ============================================================================
# METADATA GENERATOR
# ============================================================================

class ExistingImagesMetadataGenerator:
    """Generate metadata for existing images"""

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_list: List[ImageMetadata] = []

    def _calculate_hash(self, data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    def _get_image_info(self, data: bytes) -> Tuple[int, int, str, str]:
        try:
            img = Image.open(io.BytesIO(data))
            return img.size[0], img.size[1], img.format or 'UNKNOWN', img.mode
        except Exception:
            return 0, 0, 'UNKNOWN', 'UNKNOWN'

    def _check_duplicate(self, h: str) -> str:
        """Check if hash already exists"""
        for m in self.metadata_list:
            if m.md5_hash == h:
                return m.image_id
        return None

    def process_images(self) -> int:
        """Process all images in input directory"""
        logger.info(f"\nScanning directory: {self.input_dir}")
        
        # Find all image files
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        image_files = [
            f for f in self.input_dir.rglob('*')
            if f.suffix.lower() in supported_formats
        ]

        if not image_files:
            logger.error(f"❌ No images found in {self.input_dir}")
            return 0

        logger.info(f"Found {len(image_files)} images")
        logger.info(f"Processing images and generating metadata...\n")

        processed = 0
        duplicates = 0

        for idx, image_path in enumerate(tqdm(image_files, desc="Processing")):
            try:
                # Read image
                with open(image_path, 'rb') as f:
                    image_data = f.read()

                if len(image_data) < 1000:
                    logger.debug(f"Skipped {image_path.name}: too small")
                    continue

                # Calculate hash
                h = self._calculate_hash(image_data)
                
                # Check for duplicates
                dup_of = self._check_duplicate(h)
                if dup_of:
                    duplicates += 1
                    logger.debug(f"Duplicate found: {image_path.name}")
                    continue

                # Get image info
                w, ht, fmt, cs = self._get_image_info(image_data)

                # Generate image ID
                image_id = f"LOCAL_{idx}_{image_path.stem}"

                # Create metadata
                metadata = ImageMetadata(
                    image_id=image_id,
                    filename=image_path.name,
                    source="local_existing",
                    source_url=str(image_path.absolute()),
                    download_date=datetime.fromtimestamp(image_path.stat().st_mtime).isoformat(),
                    image_type="real",
                    width=w,
                    height=ht,
                    format=fmt,
                    file_size_kb=round(len(image_data) / 1024, 2),
                    color_space=cs,
                    source_metadata={
                        "original_filename": image_path.name,
                        "file_path": str(image_path),
                        "file_size_bytes": len(image_data)
                    },
                    md5_hash=h,
                    is_duplicate=False,
                    duplicate_of=None,
                    analysis_notes="Existing image - metadata generated"
                )

                self.metadata_list.append(metadata)
                processed += 1

            except Exception as e:
                logger.warning(f"Failed to process {image_path.name}: {e}")
                continue

        logger.info(f"\n✓ Processed: {processed} images")
        logger.info(f"⚠️  Duplicates skipped: {duplicates} images")
        return processed

    def save_metadata(self):
        """Save metadata to CSV and JSON"""
        logger.info("\nSaving metadata files…")
        (self.output_dir / "metadata").mkdir(exist_ok=True)

        if not self.metadata_list:
            logger.error("❌ No metadata to save")
            return

        # CSV
        df = pd.DataFrame([m.to_dict() for m in self.metadata_list])
        csv_path = self.output_dir / "metadata" / "dataset_metadata_real.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  ✓ CSV  → {csv_path} ({len(df):,} records)")

        # JSON
        json_path = self.output_dir / "metadata" / "dataset_metadata_real.json"
        with open(json_path, 'w') as f:
            json.dump([m.to_dict() for m in self.metadata_list], f, indent=2)
        logger.info(f"  ✓ JSON → {json_path}")

        # Collection log
        log_data = {
            "created_date": datetime.now().isoformat(),
            "source": "local_existing_images",
            "total_images": len(self.metadata_list),
            "source_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "metadata_files": {
                "csv": str(csv_path),
                "json": str(json_path)
            }
        }
        log_path = self.output_dir / "metadata" / "generation_log.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        logger.info(f"  ✓ Log  → {log_path}\n")

    def generate_report(self) -> Dict:
        """Generate summary report"""
        report = {
            "total_images": len(self.metadata_list),
            "average_size_mb": sum(m.file_size_kb for m in self.metadata_list) / max(len(self.metadata_list), 1) / 1024,
            "resolution_stats": {
                "avg_width": sum(m.width for m in self.metadata_list) / max(len(self.metadata_list), 1),
                "avg_height": sum(m.height for m in self.metadata_list) / max(len(self.metadata_list), 1),
                "formats": {},
                "color_spaces": {}
            }
        }

        for m in self.metadata_list:
            report["resolution_stats"]["formats"][m.format] = \
                report["resolution_stats"]["formats"].get(m.format, 0) + 1
            report["resolution_stats"]["color_spaces"][m.color_space] = \
                report["resolution_stats"]["color_spaces"].get(m.color_space, 0) + 1

        logger.info("="*70)
        logger.info("DATASET SUMMARY")
        logger.info("="*70)
        logger.info(f"Total images: {report['total_images']:,}")
        logger.info(f"Avg image size: {report['average_size_mb']:.2f} MB")
        logger.info(f"Avg resolution: {report['resolution_stats']['avg_width']:.0f}×{report['resolution_stats']['avg_height']:.0f}")
        logger.info(f"\nFormats:")
        for fmt, count in report["resolution_stats"]["formats"].items():
            logger.info(f"  {fmt}: {count}")
        logger.info(f"\nColor spaces:")
        for cs, count in report["resolution_stats"]["color_spaces"].items():
            logger.info(f"  {cs}: {count}")
        logger.info("")

        return report

    def run(self) -> Dict:
        """Run complete metadata generation"""
        self.process_images()
        self.save_metadata()
        report = self.generate_report()
        return report


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Auto-detect existing dataset directories to allow running without arguments
    default_input = "./ai_dataset_50k/real_images" if os.path.exists("./ai_dataset_50k/real_images") else None
    default_output = "./ai_dataset_50k" if os.path.exists("./ai_dataset_50k") else "./real_dataset_with_metadata"

    parser = argparse.ArgumentParser(description="Generate metadata for existing images")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=default_input,
        required=default_input is None,
        help="Directory containing existing images (defaults to ./ai_dataset_50k/real_images if found)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output,
        help="Output directory for metadata (defaults to ./ai_dataset_50k if found)"
    )

    args = parser.parse_args()

    logger.info("\n" + "="*70)
    logger.info("🔍 METADATA GENERATOR FOR EXISTING IMAGES")
    logger.info("="*70)

    generator = ExistingImagesMetadataGenerator(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir)
    )
    report = generator.run()

    logger.info("="*70)
    logger.info("✓ METADATA GENERATION COMPLETE!")
    logger.info("="*70)
    logger.info(f"Total images processed: {report['total_images']:,}")
    logger.info(f"Output directory: {args.output_dir}\n")