"""Regenerate both split index files for the updated dataset."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from src.dataset import create_split_dataloaders

METADATA = ROOT / "dataset" / "metadata"

# 1. QUICK_5K: based on test_train_manifest.csv (5000 samples)
print("=== QUICK_5K (5000 samples from test_train_manifest.csv) ===")
f = METADATA / "split_indices_quick5k.json"
if f.exists():
    f.unlink()
train_loader, val_loader, test_loader = create_split_dataloaders(
    root_dir=str(ROOT),
    metadata_paths=[str(METADATA / "test_train_manifest.csv")],
    batch_size=32,
    split_index_path=str(f),
    max_samples=5000,
    seed=42,
)
print(f"  Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")

# 2. SMOKE: based on train_manifest.csv (600 samples)
print("\n=== SMOKE (600 samples from train_manifest.csv) ===")
f = METADATA / "split_indices_smoke.json"
if f.exists():
    f.unlink()
train_loader, val_loader, test_loader = create_split_dataloaders(
    root_dir=str(ROOT),
    metadata_paths=[str(METADATA / "train_manifest.csv")],
    batch_size=32,
    split_index_path=str(f),
    max_samples=600,
    seed=42,
)
print(f"  Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")

print("\nDone — both split files regenerated.")
