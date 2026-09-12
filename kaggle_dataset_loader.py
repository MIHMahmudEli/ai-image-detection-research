"""
Multi-shard Kaggle Dataset Loader
==================================
Auto-discovers all mounted dataset directories under /kaggle/input/,
validates each shard against an expected manifest, and merges them
into a single unified index for PyTorch DataLoader consumption.

Usage:
    from kaggle_dataset_loader import KaggleDatasetLoader

    loader = KaggleDatasetLoader(
        expected_shards=["mfft-shard-0", "mfft-shard-1"],
        min_images_per_shard=1000,
    )
    df = loader.load()            # merged DataFrame
    samples = loader.to_samples() # list of (path, label, class_name)
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
LABEL_MAP = {
    "real": 0, "real_image": 0, "0": 0,
    "ai_generated": 1, "ai": 1, "ai-generated": 1, "ai_generation": 1,
    "1": 1, "fake": 1,
    "ai_altered": 2, "altered": 2, "deepfake": 2,
}
CLASS_NAMES = ["real", "ai_generated", "deepfake"]


@dataclass
class ShardManifest:
    """Expected manifest for a single dataset shard."""
    slug: str
    min_images: int = 500
    expected_labels: Optional[List[str]] = None  # e.g. ["real", "ai_generated"]
    checksum_sample: Optional[str] = None        # first 1000 chars of metadata CSV hash
    metadata_filename: str = "metadata.csv"


@dataclass
class ShardInfo:
    """Validated info about a discovered shard."""
    slug: str
    path: Path
    metadata_path: Path
    images_dir: Path
    total_images: int
    label_distribution: Dict[str, int]
    checksum_ok: bool
    validation_errors: List[str] = field(default_factory=list)


class KaggleDatasetLoader:
    """
    Discovers, validates, and merges multi-account Kaggle dataset shards.

    Each shard is a Kaggle Dataset mounted under /kaggle/input/<slug>/.
    A valid shard contains:
        <slug>/
            metadata.csv   (columns: filename, label, [source, generator])
            images/        (directory of .jpg/.png files)
    """

    def __init__(
        self,
        expected_shards: Optional[List[str]] = None,
        min_images_per_shard: int = 500,
        metadata_filename: str = "metadata.csv",
        input_root: str = None,
        seed: int = 42,
    ):
        self.expected_shards = expected_shards or []
        self.min_images_per_shard = min_images_per_shard
        self.metadata_filename = metadata_filename
        self.input_root = Path(input_root) if input_root else KAGGLE_INPUT_ROOT
        self.seed = seed
        self._discovered: Dict[str, ShardInfo] = {}

    def discover(self) -> Dict[str, ShardInfo]:
        """Auto-discover all mounted dataset directories with metadata.csv."""
        discovered = {}

        if not self.input_root.exists():
            raise FileNotFoundError(
                f"Kaggle input root not found: {self.input_root}\n"
                "Ensure datasets are attached via the Kaggle Input panel."
            )

        for entry in sorted(self.input_root.iterdir()):
            if not entry.is_dir():
                continue

            metadata_path = entry / self.metadata_filename
            if not metadata_path.exists():
                # Try common alternatives
                for alt in ["train_metadata.csv", "clean_metadata.csv", "labels.csv"]:
                    candidate = entry / alt
                    if candidate.exists():
                        metadata_path = candidate
                        break
                else:
                    logger.debug(f"Skipping {entry.name}: no metadata CSV found")
                    continue

            # Find images directory
            images_dir = self._find_images_dir(entry)
            if images_dir is None:
                logger.warning(f"Skipping {entry.name}: no images/ directory found")
                continue

            discovered[entry.name] = ShardInfo(
                slug=entry.name,
                path=entry,
                metadata_path=metadata_path,
                images_dir=images_dir,
                total_images=0,
                label_distribution={},
                checksum_ok=False,
            )

        logger.info(f"Discovered {len(discovered)} shards with metadata: "
                     f"{list(discovered.keys())}")
        self._discovered = discovered
        return discovered

    def _find_images_dir(self, shard_path: Path) -> Optional[Path]:
        """Find the images directory within a shard."""
        candidates = [
            shard_path / "images",
            shard_path / "data",
            shard_path,
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                image_count = sum(
                    1 for f in c.rglob("*")
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                )
                if image_count > 0:
                    return c
        return None

    def validate(self, shard: ShardInfo) -> ShardInfo:
        """Validate a single shard against expected constraints."""
        errors = []

        # Load metadata
        try:
            df = pd.read_csv(shard.metadata_path, low_memory=False)
        except Exception as e:
            errors.append(f"Failed to read metadata: {e}")
            shard.validation_errors = errors
            shard.total_images = 0
            return shard

        # Validate required columns
        if "filename" not in df.columns:
            errors.append("Missing 'filename' column in metadata")
        if "label" not in df.columns:
            errors.append("Missing 'label' column in metadata")

        if errors:
            shard.validation_errors = errors
            shard.total_images = 0
            return shard

        # Resolve image paths and count valid images
        valid_rows = []
        missing_count = 0
        zero_byte_count = 0

        for _, row in df.iterrows():
            img_path = self._resolve_image_path(row, shard.images_dir)
            if img_path is None:
                missing_count += 1
                continue

            if img_path.stat().st_size == 0:
                zero_byte_count += 1
                continue

            label_str = str(row.get("label", "")).lower()
            label_int = LABEL_MAP.get(label_str)
            if label_int is None:
                continue

            valid_rows.append({
                "path": str(img_path),
                "label": label_int,
                "label_name": label_str if label_int != 2 else "deepfake",
                "filename": row.get("filename", ""),
                "source": row.get("source", shard.slug),
                "generator": row.get("generator", ""),
                "shard": shard.slug,
            })

        shard.total_images = len(valid_rows)
        shard.label_distribution = dict(Counter(r["label_name"] for r in valid_rows))

        # Validation checks
        if shard.total_images < self.min_images_per_shard:
            errors.append(
                f"Too few images: {shard.total_images} < {self.min_images_per_shard}"
            )

        if missing_count > 0:
            errors.append(f"{missing_count} images referenced in metadata but missing on disk")

        if zero_byte_count > 0:
            errors.append(f"{zero_byte_count} zero-byte images skipped")

        # Checksum validation
        if shard.checksum_sample:
            actual_hash = self._compute_metadata_hash(shard.metadata_path)
            if actual_hash != shard.checksum_sample:
                errors.append(
                    f"Checksum mismatch: expected {shard.checksum_sample[:16]}..., "
                    f"got {actual_hash[:16]}..."
                )
                shard.checksum_ok = False
            else:
                shard.checksum_ok = True
        else:
            shard.checksum_ok = True

        shard.validation_errors = errors
        return shard

    def _resolve_image_path(self, row: pd.Series, images_dir: Path) -> Optional[Path]:
        """Resolve image file path from a metadata row."""
        filename = str(row.get("filename", ""))
        if not filename:
            return None

        filename = filename.replace("\\", "/")

        # Direct lookup
        candidate = images_dir / filename
        if candidate.exists() and candidate.is_file():
            return candidate

        # Subdirectory lookup
        for subdir in ["images", "real", "ai_generated", "ai_altered"]:
            candidate = images_dir / subdir / filename
            if candidate.exists() and candidate.is_file():
                return candidate

        # Flat name (filename without directory)
        flat_name = Path(filename).name
        for subdir in ["", "images", "real", "ai_generated", "ai_altered"]:
            candidate = images_dir / subdir / flat_name
            if candidate.exists() and candidate.is_file():
                return candidate

        return None

    def _compute_metadata_hash(self, metadata_path: Path) -> str:
        """Compute SHA-256 hash of metadata CSV (first 1000 chars for speed)."""
        with open(metadata_path, "r", encoding="utf-8") as f:
            content = f.read(1000)
        return hashlib.sha256(content.encode()).hexdigest()

    def validate_all(self) -> Dict[str, ShardInfo]:
        """Discover and validate all shards."""
        if not self._discovered:
            self.discover()

        for slug, shard in self._discovered.items():
            self.validate(shard)
            status = "OK" if not shard.validation_errors else "ISSUES"
            logger.info(
                f"  [{status}] {slug}: {shard.total_images} images, "
                f"distribution={shard.label_distribution}"
            )
            for err in shard.validation_errors:
                logger.warning(f"    - {err}")

        return self._discovered

    def check_expected_shards(self) -> List[str]:
        """Verify all expected shards are present and valid. Raises on failure."""
        if not self._discovered:
            self.validate_all()

        missing = []
        for slug in self.expected_shards:
            if slug not in self._discovered:
                missing.append(slug)
            elif self._discovered[slug].total_images == 0:
                missing.append(f"{slug} (found but 0 valid images)")
            elif self._discovered[slug].validation_errors:
                # Log warnings but don't fail on validation warnings
                pass

        if missing:
            raise FileNotFoundError(
                f"Missing/empty expected shards: {missing}\n"
                f"Available shards: {list(self._discovered.keys())}\n"
                "Attach all required datasets via the Kaggle Input panel."
            )

        return missing

    def merge(self) -> pd.DataFrame:
        """Merge all valid shards into a single DataFrame."""
        if not self._discovered:
            self.validate_all()

        all_rows = []
        for slug, shard in self._discovered.items():
            if shard.total_images == 0:
                logger.warning(f"Skipping empty shard: {slug}")
                continue

            df = pd.read_csv(shard.metadata_path, low_memory=False)
            for _, row in df.iterrows():
                img_path = self._resolve_image_path(row, shard.images_dir)
                if img_path is None or img_path.stat().st_size == 0:
                    continue

                label_str = str(row.get("label", "")).lower()
                label_int = LABEL_MAP.get(label_str)
                if label_int is None:
                    continue

                all_rows.append({
                    "path": str(img_path),
                    "label": label_int,
                    "label_name": "real" if label_int == 0
                        else ("ai_generated" if label_int == 1 else "deepfake"),
                    "filename": row.get("filename", ""),
                    "source": row.get("source", shard.slug),
                    "generator": row.get("generator", ""),
                    "shard": shard.slug,
                })

        if not all_rows:
            raise RuntimeError("No valid samples found across any shard")

        df = pd.DataFrame(all_rows)

        # Summary
        total = len(df)
        dist = df["label_name"].value_counts().to_dict()
        shard_counts = df["shard"].value_counts().to_dict()
        logger.info(f"\nMerged dataset: {total} samples")
        logger.info(f"  Distribution: {dist}")
        logger.info(f"  Per-shard: {shard_counts}")

        return df

    def to_samples(self, df: Optional[pd.DataFrame] = None) -> List[Tuple[str, int, str]]:
        """Convert merged DataFrame to list of (path, label, class_name)."""
        if df is None:
            df = self.merge()
        return list(zip(df["path"], df["label"], df["label_name"]))

    def load(self, df_only: bool = False):
        """
        Full pipeline: discover, validate, merge.
        Returns DataFrame. If df_only=False, also returns samples list.
        """
        self.validate_all()
        df = self.merge()

        if df_only:
            return df

        samples = self.to_samples(df)
        return df, samples

    def summary(self) -> str:
        """Human-readable summary of discovered shards."""
        lines = ["Kaggle Dataset Loader Summary", "=" * 40]
        for slug, shard in sorted(self._discovered.items()):
            status = "OK" if not shard.validation_errors else "ISSUES"
            lines.append(f"  [{status}] {slug}")
            lines.append(f"    Images: {shard.total_images}")
            lines.append(f"    Labels: {shard.label_distribution}")
            for err in shard.validation_errors:
                lines.append(f"    WARNING: {err}")
        lines.append(f"  Total shards: {len(self._discovered)}")
        return "\n".join(lines)


def create_pytorch_datasets(
    df: pd.DataFrame,
    image_size: int = 224,
    val_split: float = 0.15,
    seed: int = 42,
):
    """
    Create PyTorch train/val datasets + dataloaders from merged DataFrame.
    Uses stratified split. Returns (train_loader, val_loader, train_dataset, val_dataset).
    """
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import transforms
    from sklearn.model_selection import train_test_split
    from PIL import Image

    class KaggleImageDataset(Dataset):
        def __init__(self, samples, transform=None):
            self.samples = samples  # list of (path, label)
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
            except Exception:
                # Retry with next sample
                for offset in range(1, min(10, len(self.samples))):
                    retry_idx = (idx + offset) % len(self.samples)
                    try:
                        img = Image.open(self.samples[retry_idx][0]).convert("RGB")
                        if self.transform:
                            img = self.transform(img)
                        return img, self.samples[retry_idx][1]
                    except Exception:
                        continue
                blank = torch.zeros(3, image_size, image_size)
                return blank, label

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(image_size + 16),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, fill=128),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size + 16),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Stratified split
    paths = df["path"].tolist()
    labels = df["label"].tolist()

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=val_split, stratify=labels, random_state=seed,
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))

    train_dataset = KaggleImageDataset(train_samples, transform=train_transform)
    val_dataset = KaggleImageDataset(val_samples, transform=val_transform)

    # Weighted sampler for class balance
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    num_workers = 2 if torch.cuda.is_available() else 0

    train_loader = DataLoader(
        train_dataset, batch_size=32, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Class counts (train): real={class_counts[0]}, ai={class_counts[1]}", end="")
    if len(class_counts) > 2:
        print(f", deepfake={class_counts[2]}", end="")
    print()

    return train_loader, val_loader, train_dataset, val_dataset
