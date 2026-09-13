"""
Kaggle Dataset Loader
======================
Manifest-based split assignment for reproducible, cross-session data loading.

Replaces the old train_test_split approach with a frozen manifest that
guarantees identical val/test images across all parallel Kaggle sessions.

Usage (in Kaggle notebooks):
    from kaggle_dataset_loader import KaggleDatasetLoader

    loader = KaggleDatasetLoader(manifest=manifest, manifest_sha256=sha256)
    train_loader, val_loader, test_loader = loader.create_dataloaders(batch_size=32)
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from pipeline_config import (
    IMG_EXTENSIONS, LABEL_MAP, CLASS_NAMES, KAGGLE_DATASETS, KAGGLE_INPUT_ROOT,
)
from split_manifest_manager import stable_image_id, build_mount_index

logger = logging.getLogger(__name__)

# Default transforms (shared across all splits for compatibility)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ManifestDataset(Dataset):
    """
    Dataset that loads images based on manifest entries and resolved paths.
    Handles missing files gracefully with retry logic.
    """

    def __init__(self, entries: List[Dict], transform=None):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = Path(entry["path"])
        label = entry["label_int"]

        for attempt in range(3):
            try:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"Failed to load {img_path}: {e}")
                    dummy = torch.zeros(3, 224, 224)
                    return dummy, label


class KaggleDatasetLoader:
    """
    Manifest-based dataset loader. Uses a frozen manifest to assign
    train/val/test splits, guaranteeing cross-session reproducibility.

    Args:
        manifest: The manifest dict (from SplitManifestManager)
        manifest_sha256: SHA-256 of the manifest for verification
        input_root: Root of Kaggle input mounts (default /kaggle/input)
        image_size: Target image size (default 224)
    """

    def __init__(
        self,
        manifest: dict,
        manifest_sha256: str = "",
        input_root: str = None,
        image_size: int = 224,
    ):
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self.input_root = Path(input_root) if input_root else KAGGLE_INPUT_ROOT
        self.image_size = image_size

        # Verify hash if provided
        if manifest_sha256:
            from split_manifest_manager import compute_manifest_sha256
            computed = compute_manifest_sha256(manifest)
            if computed != manifest_sha256:
                raise ValueError(
                    f"Manifest hash mismatch! expected={manifest_sha256[:12]}, "
                    f"computed={computed[:12]}"
                )

        # Build mount indices ONCE (BUG 1 fix — O(total_files) not O(N*M))
        self._mount_indices: Dict[str, Dict[str, Path]] = {}
        self._build_all_indices()

        # Apply split manifest
        self.train_data, self.val_data, self.test_data = self._apply_split_manifest()

    def _build_all_indices(self):
        """Build per-shard lookup indices by walking each mount once."""
        from split_manifest_manager import match_mount, _find_mount_path, _load_mount_overrides

        shards_needed = set(
            info["shard"] for info in self.manifest["images"].values()
        )

        # Discover actual mount names
        all_mount_names = []
        if self.input_root.exists():
            all_mount_names = [d.name for d in self.input_root.iterdir() if d.is_dir()]
            datasets_dir = self.input_root / "datasets"
            if datasets_dir.exists():
                all_mount_names += [d.name for d in datasets_dir.iterdir() if d.is_dir()]
            all_mount_names = list(set(all_mount_names))

        overrides = _load_mount_overrides()

        for shard in shards_needed:
            if shard not in KAGGLE_DATASETS:
                continue
            label, subdir = KAGGLE_DATASETS[shard]

            # Check overrides first
            if shard in overrides:
                mount_dir = _find_mount_path(self.input_root, overrides[shard])
            else:
                mount_dir, _, _ = match_mount(shard, all_mount_names, self.input_root)

            if mount_dir is None:
                continue

            image_root = mount_dir / subdir
            if not image_root.exists():
                image_root = mount_dir

            idx = build_mount_index(image_root, shard, label)
            self._mount_indices[shard] = idx

        print(f"[DatasetLoader] Built indices for {len(self._mount_indices)} shards")

    def _resolve_path(self, image_id: str, info: dict) -> Optional[Path]:
        """O(1) lookup via pre-built mount index."""
        shard = info["shard"]
        idx = self._mount_indices.get(shard)
        if idx is not None:
            return idx.get(image_id)
        return None

    def _apply_split_manifest(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Join manifest entries with mounted dataset paths."""
        train_entries = []
        val_entries = []
        test_entries = []
        unresolved = []

        for image_id, info in self.manifest["images"].items():
            split = info["split"]
            path = self._resolve_path(image_id, info)

            if path is None:
                unresolved.append((image_id, info["shard"], info["label"]))
                continue

            entry = {
                "image_id": image_id,
                "path": str(path),
                "shard": info["shard"],
                "label": info["label"],
                "label_int": info.get("label_int", 0),
            }

            if split == "train":
                train_entries.append(entry)
            elif split == "val":
                val_entries.append(entry)
            elif split == "test":
                test_entries.append(entry)

        # Report
        total = len(train_entries) + len(val_entries) + len(test_entries)
        print(f"[DatasetLoader] Resolved {total} images:")
        print(f"  train: {len(train_entries)}")
        print(f"  val:   {len(val_entries)}")
        print(f"  test:  {len(test_entries)}")

        if unresolved:
            print(f"\n  WARNING: {len(unresolved)} images could not be resolved:")
            for img_id, shard, label in unresolved[:10]:
                print(f"    {img_id} ({shard}/{label})")
            if len(unresolved) > 10:
                print(f"    ... and {len(unresolved) - 10} more")

            # BUG 2 fix: use .get() with fallback, never bare dict access
            manifest_total = self.manifest.get("total_images", len(self.manifest.get("images", {})))
            unresolved_pct = len(unresolved) / manifest_total if manifest_total > 0 else 0
            if unresolved_pct > 0.02:
                raise RuntimeError(
                    f"Too many unresolved images: {len(unresolved)}/{manifest_total} "
                    f"({unresolved_pct:.1%}). Check that all datasets are mounted."
                )

        # Sanity check: distribution tolerance +/-2%
        self._verify_distribution(train_entries, val_entries, test_entries)

        return train_entries, val_entries, test_entries

    def _verify_distribution(
        self, train_entries, val_entries, test_entries
    ):
        """Verify class distribution matches manifest within +/-2% tolerance."""
        manifest_dist = self.manifest.get("class_distribution", {})
        if not manifest_dist:
            return

        # BUG 2 fix: use .get() with fallback
        total = self.manifest.get("total_images", len(self.manifest.get("images", {})))
        if total == 0:
            return

        # Count actual distribution across all splits
        all_entries = train_entries + val_entries + test_entries
        actual_dist = Counter(e["label"] for e in all_entries)

        tolerance = 0.02
        for label_name, expected_ratio in manifest_dist.items():
            actual_ratio = actual_dist.get(label_name, 0) / len(all_entries) if all_entries else 0
            if abs(actual_ratio - expected_ratio) > tolerance:
                logger.warning(
                    f"Distribution mismatch for {label_name}: "
                    f"expected ~{expected_ratio:.3f}, got {actual_ratio:.3f} "
                    f"(delta={abs(actual_ratio - expected_ratio):.3f} > {tolerance})"
                )

    def create_dataloaders(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders with WeightedRandomSampler for balanced training.

        Returns:
            (train_loader, val_loader, test_loader)
        """
        train_ds = ManifestDataset(self.train_data, transform=TRAIN_TRANSFORM)
        val_ds = ManifestDataset(self.val_data, transform=VAL_TRANSFORM)
        test_ds = ManifestDataset(self.test_data, transform=VAL_TRANSFORM)

        # WeightedRandomSampler for training (handle class imbalance)
        if len(train_ds) > 0:
            labels = [e["label_int"] for e in self.train_data]
            class_counts = Counter(labels)
            class_weights = {c: 1.0 / count for c, count in class_counts.items()}
            sample_weights = [class_weights[l] for l in labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
            )
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader

    def get_class_names(self) -> List[str]:
        return CLASS_NAMES

    def get_split_summary(self) -> Dict:
        """Return a summary of the current data splits."""
        return {
            "manifest_sha256": self.manifest_sha256[:16] if self.manifest_sha256 else "",
            "manifest_version": self.manifest.get("version", "?"),
            # BUG 2 fix: use .get() with fallback
            "total_images": self.manifest.get("total_images", len(self.manifest.get("images", {}))),
            "split_sizes": self.manifest.get("split_sizes", {}),
            "class_distribution": self.manifest.get("class_distribution", {}),
            "resolved": {
                "train": len(self.train_data),
                "val": len(self.val_data),
                "test": len(self.test_data),
            },
        }
