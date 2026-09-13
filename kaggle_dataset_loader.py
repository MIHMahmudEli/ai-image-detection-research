"""
Kaggle Dataset Loader (Manifest-Based)
========================================
Downloads the master manifest from HuggingFace Hub and resolves image
paths from mounted Kaggle datasets. Every training session uses the
exact same split indices, ensuring fair evaluation across all models.

Usage:
    from kaggle_dataset_loader import KaggleDatasetLoader

    loader = KaggleDatasetLoader(hf_token=os.environ["HF_TOKEN"])
    train_df, val_df, test_df = loader.load()
    train_loader, val_loader = loader.to_dataloaders(image_size=224)
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
HF_REPO_ID = "studyhub991/mfft-master-manifest"
CLASS_NAMES = ["real", "ai_generated", "deepfake"]


class KaggleDatasetLoader:
    """
    Loads the master manifest from HuggingFace and resolves image paths
    from mounted Kaggle datasets.

    Every session gets the exact same train/val/test split because:
    1. The manifest defines deterministic split indices
    2. Each session resolves paths from its mounted datasets
    3. Only images present in BOTH the manifest AND the mount are used
    """

    def __init__(
        self,
        hf_token: str = None,
        hf_repo: str = HF_REPO_ID,
        input_root: str = None,
        image_size: int = 224,
    ):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.hf_repo = hf_repo
        self.input_root = Path(input_root) if input_root else KAGGLE_INPUT_ROOT
        self.image_size = image_size
        self._manifest = None
        self._mount_map: Dict[str, Path] = {}

    def _get_hf_token(self) -> str:
        """Get HF token from various sources."""
        if self.hf_token:
            return self.hf_token

        # Try Kaggle Secrets
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass

        # Try .env file
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("hf="):
                        return line.strip().split("=", 1)[1]

        raise ValueError(
            "HF_TOKEN not found. Set via:\n"
            "  1. Kaggle Secret: Name=HF_TOKEN\n"
            "  2. Environment: export HF_TOKEN=...\n"
            "  3. .env file: hf=hf_..."
        )

    def download_manifest(self) -> dict:
        """Download master manifest from HuggingFace Hub."""
        if self._manifest is not None:
            return self._manifest

        print("Downloading master manifest from HuggingFace Hub...")

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            os.system("pip install huggingface_hub -q")
            from huggingface_hub import hf_hub_download

        token = self._get_hf_token()

        try:
            manifest_path = hf_hub_download(
                repo_id=self.hf_repo,
                filename="master_manifest.json",
                repo_type="model",
                token=token,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download manifest from {self.hf_repo}: {e}\n"
                "Ensure the HF repo exists and contains master_manifest.json.\n"
                "Run: python build_master_manifest.py --upload"
            )

        with open(manifest_path, "r") as f:
            self._manifest = json.load(f)

        print(f"  Manifest loaded: {self._manifest['sample_count'] if 'sample_count' in self._manifest else len(self._manifest.get('samples', []))} samples")
        print(f"  Version: {self._manifest.get('version', 'unknown')}")
        print(f"  Hash: {self._manifest.get('manifest_hash', 'N/A')}")

        return self._manifest

    def discover_mounts(self) -> Dict[str, Path]:
        """Auto-discover all mounted Kaggle datasets."""
        if self._mount_map:
            return self._mount_map

        if not self.input_root.exists():
            logger.warning(f"Kaggle input root not found: {self.input_root}")
            return {}

        for entry in sorted(self.input_root.iterdir()):
            if entry.is_dir():
                self._mount_map[entry.name] = entry
                logger.debug(f"  Mounted: {entry.name}")

        print(f"  Discovered {len(self._mount_map)} mounted datasets: {list(self._mount_map.keys())}")
        return self._mount_map

    def resolve_path(self, relative_path: str, kaggle_mount_slug: str) -> Optional[Path]:
        """
        Resolve an image path from the manifest against mounted Kaggle datasets.
        
        The manifest stores paths like "Stable Diffusion/images/0/custom_0_0.png"
        which need to be resolved against the Kaggle mount path.
        """
        mounts = self.discover_mounts()

        # Try the specific mount slug first
        if kaggle_mount_slug and kaggle_mount_slug in mounts:
            mount_dir = mounts[kaggle_mount_slug]
            # The mount may have an extra subdirectory
            # e.g., /kaggle/input/stable-diffusion/Stable Diffusion/images/...
            candidate = mount_dir / relative_path
            if candidate.exists() and candidate.is_file():
                return candidate

            # Try without the first directory component (some datasets have nested dirs)
            parts = Path(relative_path).parts
            if len(parts) > 1:
                candidate = mount_dir / Path(*parts[1:])
                if candidate.exists() and candidate.is_file():
                    return candidate

            # Try flat search - just the filename
            filename = Path(relative_path).name
            for img in mount_dir.rglob(filename):
                if img.is_file():
                    return img

        # Try all mounts as fallback
        filename = Path(relative_path).name
        for mount_name, mount_dir in mounts.items():
            # Direct path
            candidate = mount_dir / relative_path
            if candidate.exists() and candidate.is_file():
                return candidate

            # Flat search
            for img in mount_dir.rglob(filename):
                if img.is_file():
                    return img

        return None

    def load(self, split: str = "all") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load manifest and resolve paths. Returns (train_df, val_df, test_df).

        split: "train", "val", "test", or "all"
        """
        manifest = self.download_manifest()
        samples = manifest.get("samples", [])
        split_info = manifest.get("split", {})

        if not samples:
            raise RuntimeError("Manifest has no samples. Re-run build_master_manifest.py")

        # Get split indices
        train_indices = split_info.get("train_indices", [])
        val_indices = split_info.get("val_indices", [])
        test_indices = split_info.get("test_indices", [])

        if not train_indices:
            raise RuntimeError("Manifest has no split indices. Re-run build_master_manifest.py")

        print(f"\nManifest split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        # Resolve paths for each split
        def resolve_split(indices, split_name):
            resolved = []
            missing = 0
            for idx in indices:
                if idx >= len(samples):
                    continue
                sample = samples[idx]
                path = self.resolve_path(
                    sample["relative_path"],
                    sample.get("kaggle_mount_slug", ""),
                )
                if path is None:
                    missing += 1
                    continue

                resolved.append({
                    "path": str(path),
                    "label": sample["label"],
                    "label_name": CLASS_NAMES[sample["label"]],
                    "image_id": sample["image_id"],
                    "source": sample.get("source", ""),
                    "split": split_name,
                })

            if missing > 0:
                print(f"  {split_name}: {missing}/{len(indices)} images not found on this mount")

            return pd.DataFrame(resolved)

        train_df = resolve_split(train_indices, "train")
        val_df = resolve_split(val_indices, "val")
        test_df = resolve_split(test_indices, "test")

        # Print summary
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if len(df) > 0:
                dist = df["label_name"].value_counts().to_dict()
                print(f"  {name}: {len(df)} samples | {dist}")
            else:
                print(f"  {name}: 0 samples")

        return train_df, val_df, test_df

    def to_dataloaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        batch_size: int = 64,
        image_size: int = None,
        num_workers: int = 2,
    ):
        """Create PyTorch dataloaders with shared transforms."""
        import torch
        from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
        from torchvision import transforms
        from PIL import Image

        image_size = image_size or self.image_size

        class ManifestDataset(Dataset):
            def __init__(self, df, transform=None):
                self.samples = list(zip(df["path"].tolist(), df["label"].tolist()))
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
                    for offset in range(1, min(10, len(self.samples))):
                        retry_idx = (idx + offset) % len(self.samples)
                        try:
                            img = Image.open(self.samples[retry_idx][0]).convert("RGB")
                            if self.transform:
                                img = self.transform(img)
                            return img, self.samples[retry_idx][1]
                        except Exception:
                            continue
                    return torch.zeros(3, image_size, image_size), label

        # Shared transforms (deterministic across all sessions)
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

        train_dataset = ManifestDataset(train_df, transform=train_transform)
        val_dataset = ManifestDataset(val_df, transform=val_transform)

        # Weighted sampler for class balance
        train_labels = [s[1] for s in train_dataset.samples]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[l] for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        print(f"\nDataloaders: train={len(train_dataset)}, val={len(val_dataset)}")
        print(f"Class counts (train): {dict(Counter(train_labels))}")

        return train_loader, val_loader

    def get_split_info(self) -> dict:
        """Return split metadata for logging/verification."""
        manifest = self.download_manifest()
        split = manifest.get("split", {})
        return {
            "seed": split.get("seed"),
            "train_count": split.get("train_count"),
            "val_count": split.get("val_count"),
            "test_count": split.get("test_count"),
            "train_class_dist": split.get("train_class_dist"),
            "val_class_dist": split.get("val_class_dist"),
            "test_class_dist": split.get("test_class_dist"),
            "manifest_hash": manifest.get("manifest_hash"),
        }
