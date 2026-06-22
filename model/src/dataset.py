import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import pandas as pd
from PIL import Image
import random
import numpy as np
from typing import Optional, Callable, Dict, List, Tuple
from torchvision import transforms


class ImageTransform:
    def __init__(self, size: int = 384, augment: bool = True):
        self.size = size
        self.augment = augment

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5, fill=128),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                self.normalize,
            ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img.convert("RGB"))


class AIDetectionDataset(Dataset):
    """
    Dataset for AI-generated image detection.
    Supports multiple metadata formats (v1, v2, v3).

    Labels:
        0 = real
        1 = ai_generated
    """
    def __init__(
        self,
        root_dir: str,
        metadata_paths: List[str],
        transform: Optional[Callable] = None,
        is_train: bool = True,
        size: int = 384,
        undersample: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or ImageTransform(size=size, augment=is_train)
        self.samples = []
        self.metadata_paths = metadata_paths

        self._load_all_metadata()

        if undersample and is_train:
            self._undersample()

        print(f"Dataset loaded: {len(self.samples)} samples")
        self._print_stats()

    def _load_all_metadata(self):
        for mp in self.metadata_paths:
            p = Path(mp)
            if not p.exists():
                print(f"  Warning: {p} not found, skipping")
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                try:
                    df = pd.read_json(p)
                except Exception:
                    print(f"  Warning: cannot read {p}")
                    continue

            image_dirs = self._resolve_image_dirs(p, df)
            if not image_dirs:
                continue

            for _, row in df.iterrows():
                img_path = self._resolve_image_path(row, image_dirs)
                if img_path and img_path.exists():
                    label = self._get_label(row)
                    if label is not None:
                        self.samples.append((str(img_path), label))

    def _resolve_image_dirs(self, metadata_path: Path, df: pd.DataFrame) -> List[Path]:
        base = metadata_path.parent.parent / "images"
        candidates = [
            base / "real",
            base / "ai_generated",
            base / "ai_altered",
        ]
        existing = [c for c in candidates if c.exists()]
        if existing:
            return existing

        if 'filename' in df.columns:
            first_file = str(df['filename'].iloc[0])
            valid = [c for c in candidates if (c / first_file).exists()]
            if valid:
                return valid

        print(f"  Warning: cannot resolve image dir for {metadata_path}")
        return []

    def _resolve_image_path(self, row: pd.Series, image_dirs: List[Path]) -> Optional[Path]:
        for image_dir in image_dirs:
            if 'filename' in row and pd.notna(row['filename']):
                p = image_dir / row['filename']
                if p.exists():
                    return p
                p = Path(str(row['filename']))
                if p.exists():
                    return p

            if 'image_id' in row and pd.notna(row['image_id']):
                for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    p = image_dir / f"{row['image_id']}{ext}"
                    if p.exists():
                        return p

        return None

    def _get_label(self, row: pd.Series) -> Optional[int]:
        if 'image_type' in row:
            val = str(row['image_type']).lower()
            if val in ('real', 'real_image'):
                return 0
            if val in ('ai_generated', 'ai', 'ai-generated', 'ai_generation'):
                return 1
            if val in ('ai_altered', 'altered', 'deepfake'):
                return 1

        if 'label' in row:
            val = str(row['label']).lower()
            if val in ('0', 'real'):
                return 0
            if val in ('1', 'ai', 'fake'):
                return 1

        return None

    def _undersample(self):
        labels = np.array([s[1] for s in self.samples])
        n_real = (labels == 0).sum()
        n_ai = (labels == 1).sum()
        target = min(n_real, n_ai)

        real_indices = [i for i, l in enumerate(labels) if l == 0]
        ai_indices = [i for i, l in enumerate(labels) if l == 1]

        if len(real_indices) > target:
            real_indices = random.sample(real_indices, target)
        if len(ai_indices) > target:
            ai_indices = random.sample(ai_indices, target)

        kept = set(real_indices + ai_indices)
        self.samples = [s for i, s in enumerate(self.samples) if i in kept]

    def _print_stats(self):
        labels = [s[1] for s in self.samples]
        n_real = labels.count(0)
        n_ai = labels.count(1)
        print(f"  Real: {n_real}, AI: {n_ai}, Total: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, label


def create_dataloaders(
    root_dir: str,
    metadata_paths: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    size: int = 384,
    val_split: float = 0.15,
    undersample: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates stratified train/val dataloaders from metadata files.
    """
    full_dataset = AIDetectionDataset(
        root_dir=root_dir,
        metadata_paths=metadata_paths,
        transform=None,
        is_train=True,
        size=size,
        undersample=False,
    )

    labels = [s[1] for s in full_dataset.samples]

    indices = list(range(len(full_dataset)))
    from sklearn.model_selection import train_test_split

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels,
        random_state=42,
    )

    train_dataset = AIDetectionDataset(
        root_dir=root_dir,
        metadata_paths=[],  # don't reload
        transform=ImageTransform(size=size, augment=True),
        is_train=True,
        size=size,
        undersample=undersample,
    )
    train_dataset.samples = [full_dataset.samples[i] for i in train_idx]

    val_dataset = AIDetectionDataset(
        root_dir=root_dir,
        metadata_paths=[],
        transform=ImageTransform(size=size, augment=False),
        is_train=False,
        size=size,
        undersample=False,
    )
    val_dataset.samples = [full_dataset.samples[i] for i in val_idx]

    print(f"\nTrain: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_weighted_sampler(dataset: AIDetectionDataset) -> WeightedRandomSampler:
    labels = [s[1] for s in dataset.samples]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = [weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
