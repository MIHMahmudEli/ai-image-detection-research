"""Build dataset/metadata/test_train_manifest.csv — the exact 5,000-image
balanced subset (2,500 real / 2,500 AI, seed 42) used by the notebooks'
QUICK_5K mode, written as a standalone manifest with all original metadata
columns (generator, source, md5, ...).

The selection replicates create_split_dataloaders(max_samples=5000, seed=42)
on clean_metadata.csv row-for-row, so the shared split file
split_indices_quick5k.json built against that subset remains valid when the
notebooks load this manifest directly (which skips the 2.7M-row metadata scan).

Usage:  python dataset/scripts/prepare_test_manifest.py
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from src.dataset import AIDetectionDataset  # noqa: E402

SEED = 42
MAX_SAMPLES = 5000
SRC = ROOT / "dataset" / "metadata" / "clean_metadata.csv"
OUT = ROOT / "dataset" / "metadata" / "test_train_manifest.csv"
SPLIT_FILE = ROOT / "dataset" / "metadata" / "split_indices_quick5k.json"


def main() -> None:
    # empty-manifest instance: reuses the dataset's own path-resolution and
    # label logic without loading anything
    probe = AIDetectionDataset(str(ROOT), metadata_paths=[], is_train=False)
    df = pd.read_csv(SRC, low_memory=False, dtype={"generator": str, "md5": str})
    dirs = probe._resolve_image_dirs(SRC, df)
    print(f"{SRC.name}: {len(df)} rows; scanning files (takes a few minutes)...")

    # mirror _load_all_metadata exactly, but keep the originating row index
    samples = []  # (path, label, row_idx)
    for idx, row in df.iterrows():
        p = probe._resolve_image_path(row, dirs)
        if p and p.exists():
            if p.stat().st_size == 0:
                continue
            label = probe._get_label(row)
            if label is not None:
                samples.append((str(p), label, idx))
    print(f"Resolved {len(samples)} loadable samples")

    # identical subset draw to create_split_dataloaders(max_samples, seed)
    rng = random.Random(SEED)
    by_class = {0: [], 1: []}
    for s in samples:
        by_class[s[1]].append(s)
    per_class = MAX_SAMPLES // 2
    subset = []
    for _lbl, items in by_class.items():
        rng.shuffle(items)
        subset.extend(items[:per_class])
    rng.shuffle(subset)

    out_df = df.iloc[[s[2] for s in subset]]
    out_df.to_csv(OUT, index=False)
    print(f"Wrote {len(out_df)} rows -> {OUT}")

    # verify: loading the new manifest must reproduce the subset
    # sample-for-sample, or the shared split indices would be misaligned
    check = AIDetectionDataset(str(ROOT), metadata_paths=[str(OUT)], is_train=False)
    expected = [(s[0], s[1]) for s in subset]
    if check.samples != expected:
        raise SystemExit("FAILED: manifest does not reproduce the QUICK_5K subset")

    if SPLIT_FILE.exists():
        n = json.loads(SPLIT_FILE.read_text()).get("n_samples")
        if n != len(expected):
            raise SystemExit(
                f"FAILED: {SPLIT_FILE.name} expects {n} samples, manifest has "
                f"{len(expected)} — delete the split file to regenerate")
        print(f"Verified against {SPLIT_FILE.name} (n_samples={n})")
    print("OK: manifest reproduces the QUICK_5K subset sample-for-sample.")


if __name__ == "__main__":
    main()
