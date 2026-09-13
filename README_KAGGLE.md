# Kaggle Multi-Session Training Pipeline

Resumable, multi-session, multi-account training for MFFT on Kaggle with HuggingFace Hub persistence and frozen split manifests.

## Overview

This pipeline allows you to train MFFT across multiple Kaggle sessions (each limited to 9-12 hours) and **multiple Kaggle accounts** by persisting all checkpoints, metrics, and training state to a HuggingFace Hub repo. All parallel runs evaluate on the **exact same data** because a frozen split manifest is shared across sessions.

```
Session 1 (account A, mfft-base):  Epochs 1-12  ──► bootstrap manifest → HF Hub
Session 2 (account B, mfft-tiny):  Epochs 1-12  ──► download manifest → HF Hub
Session 3 (account A, mfft-base):  Epochs 13-24 ──► download manifest → HF Hub
...
```

**Key guarantees:**
- Same `split_manifest.json` used by ALL sessions (frozen after first run)
- Same `manifest_sha256` verified on every resume
- Per-run namespacing prevents checkpoint collisions between parallel sessions

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                      Kaggle Notebook                          │
│                                                               │
│  split_manifest_manager.py                                    │
│    ├── Downloads existing manifest from HF (all runs)         │
│    ├── OR bootstraps from mounted datasets (first run only)   │
│    ├── Computes stable_image_id (sha1 of filename+shard+label)│
│    └── Uploads frozen manifest to HF (shared across accounts) │
│                                                               │
│  kaggle_dataset_loader.py                                     │
│    ├── Joins manifest entries with mounted dataset paths      │
│    ├── Assigns train/val/test splits from manifest            │
│    ├── Sanity check: distribution ±2% tolerance               │
│    └── Creates DataLoaders with WeightedRandomSampler         │
│                                                               │
│  hf_checkpoint_manager.py                                     │
│    ├── Per-run namespacing: runs/<run_id>/...                  │
│    ├── Resume: download last.pt + training_state.json         │
│    ├── Verifies manifest_sha256 matches on resume             │
│    └── Per-epoch: save local → upload HF (with retry)         │
│                                                               │
│  kaggle_train_resumable.py                                    │
│    ├── Generates run_id from model + account + date           │
│    ├── Calls SplitManifestManager at startup                  │
│    ├── Wires dataset loader + checkpoint manager              │
│    └── Uploads final artifacts + ONNX export                  │
│                                                               │
│                            │                                  │
│                            ▼                                  │
│               ┌─────────────────────────────┐                 │
│               │    HuggingFace Hub           │                 │
│               │                              │                 │
│               │  studyhub991/mfft-checkpoints│                 │
│               │    ├── manifest/split_manifest.json (SHARED)  │
│               │    ├── runs/<run_id>/...     (PER-RUN)        │
│               │    │   ├── checkpoints/      │                │
│               │    │   ├── logs/             │                │
│               │    │   └── README.md         │                │
│               └─────────────────────────────┘                 │
└───────────────────────────────────────────────────────────────┘
```

## Setup (One-Time)

### 1. Create HuggingFace Repos

```bash
# On huggingface.co, create TWO repos:
#   1. studyhub991/mfft-checkpoints (Model, private)
#   2. studyhub991/mfft-master-manifest (Model, private)
```

### 2. Set Kaggle Secrets

On each Kaggle account, add these secrets:
- `HF_TOKEN`: Your HuggingFace write token
- (Optional) `KAGGLE_USERNAME`: Your Kaggle username (for run_id labeling)

### 3. Create Kaggle Notebook

1. Create new Kaggle Notebook
2. **Settings → Accelerator**: GPU T4 x2 (or P100)
3. **Settings → Internet**: On
4. **Add Input**: Attach all 11 dataset shards:
   - `stable-diffusion`, `places365`, `open-images-v7-dataset`
   - `ntire2026`, `midjourney`, `mfft-real`, `genimage-ai`
   - `faceforensics`, `dfdc-faces-of-the-train-sample`
   - `dall-e3`, `celebdf-v2image-dataset`

### 4. Copy Files to Kaggle

Copy to your Kaggle notebook:
- `split_manifest_manager.py`
- `kaggle_dataset_loader.py`
- `hf_checkpoint_manager.py`
- `kaggle_train_resumable.py`

### 5. Run

```python
!python kaggle_train_resumable.py
```

## How the Split Manifest Works

### First Run (Bootstrap)

The first time `kaggle_train_resumable.py` runs (with any account, any model variant):

1. `SplitManifestManager` tries to download `manifest/split_manifest.json` from HF
2. It does NOT exist → triggers bootstrap
3. Scans all mounted datasets under `/kaggle/input/`
4. For each image, computes a **stable ID**: `sha1(filename + shard + label)`
5. Creates a **stratified** train/val/test split (seed=42):
   - 70% train, 15% val, 15% test
   - Class balance preserved in each split
6. Uploads `manifest/split_manifest.json` to HF (frozen, never modified)
7. Logs `created_by_run` and `manifest_sha256`

### Later Runs (Download)

Every subsequent run:

1. `SplitManifestManager` downloads existing manifest from HF
2. Verifies `manifest_sha256` matches stored hash
3. `KaggleDatasetLoader` resolves image paths from mounted datasets
4. Assigns each image to train/val/test using the manifest
5. Raises error if any image cannot be found (not silently dropped)

### Adding New Datasets

To add new Kaggle datasets after the manifest is frozen:

```bash
# On Kaggle (with new datasets attached):
!python extend_manifest.py
```

This:
1. Downloads current manifest
2. Discovers NEW shards not already in `shards_used`
3. Splits new images using same ratios
4. Appends to manifest, bumps version
5. Archives old version as `manifest/split_manifest_vN.json`
6. Uploads new version to `manifest/split_manifest.json`

**Old runs are NOT invalidated** — they still have their `manifest_sha256` recorded and can be resumed with their original data.

## Run ID Convention

Each training session gets a unique `run_id`:

```
<model_variant>_<account>_<date>

Examples:
  base_studyhub_20250601
  tiny_mihmahm_20250601
  large_benjami_20250602
```

All HF paths are prefixed with `runs/<run_id>/`:
```
runs/base_studyhub_20250601/checkpoints/last.pt
runs/base_studyhub_20250601/checkpoints/best.pt
runs/base_studyhub_20250601/logs/training_state.json
runs/base_studyhub_20250601/logs/metrics.csv
runs/base_studyhub_20250601/README.md
```

The manifest stays at the top level (shared, un-namespaced):
```
manifest/split_manifest.json
```

## HuggingFace Repo Structure

```
studyhub991/mfft-checkpoints/
├── manifest/
│   ├── split_manifest.json              # Frozen split (SHARED)
│   ├── split_manifest_summary.json      # Human-readable summary
│   └── split_manifest_v1.json           # Archived versions
├── runs/
│   ├── base_studyhub_20250601/
│   │   ├── README.md                    # Per-run model card
│   │   ├── checkpoints/
│   │   │   ├── last.pt                  # Most recent epoch
│   │   │   ├── best.pt                  # Best val_macro_f1
│   │   │   └── epoch_*.pt               # Snapshots
│   │   ├── logs/
│   │   │   ├── training_state.json      # Resume source of truth
│   │   │   ├── metrics.csv              # Per-epoch metrics
│   │   │   └── session_log.csv          # Session timestamps
│   │   └── results/
│   │       └── model.onnx               # ONNX export
│   └── tiny_mihmahm_20250601/
│       └── ...
```

## Metrics Tracked

| Metric | Purpose |
|---|---|
| `val_macro_f1` | **Primary** — best checkpoint + early stopping decision |
| `val_auc` | Tiebreaker when macro_f1 is equal |
| `val_loss` | Logged for manuscript figures |
| `val_accuracy` | Logged for manuscript figures |

**Why macro-F1?** The dataset is imbalanced (47.4% real / 43.8% AI / 9% deepfake). Macro-F1 weights all three classes equally, preventing the majority classes from dominating the metric.

## Monitoring

### Check Run Status (Local)

```bash
python list_active_runs.py
```

Output:
```
run_id                              epoch  best_f1 status          manifest_sha256  last_updated
──────────────────────────────────────────────────────────────────────────────────────────────────
base_studyhub_20250601                 12   0.8234 training        a1b2c3d4e5f6g7h8  2025-06-01
tiny_mihmahm_20250601                   8   0.7891 early_stopped   a1b2c3d4e5f6g7h8  2025-06-01
```

The `manifest_sha256` column confirms all runs used the same data split.

### List Available Datasets (Local)

```bash
python explore_kaggle_datasets.py
```

Lists all datasets across all configured Kaggle accounts.

## Resume After Disconnect

```
Session starts
    │
    ├── SplitManifestManager.get_or_bootstrap()
    │   ├── Manifest exists on HF → download, verify hash
    │   └── No manifest → bootstrap from mounted datasets
    │
    ├── KaggleDatasetLoader(manifest, manifest_sha256)
    │   ├── Resolve paths from mounted datasets
    │   ├── Join with manifest entries
    │   └── Create DataLoaders with WeightedRandomSampler
    │
    ├── HFCheckpointManager(run_id, manifest_sha256)
    │   ├── Download runs/<run_id>/logs/training_state.json
    │   ├── Verify manifest_sha256 matches current
    │   ├── Download runs/<run_id>/checkpoints/last.pt
    │   └── Resume from last_epoch + 1
    │
    └── Training loop (40 epochs max)
        │
        ├── Per epoch:
        │   ├── Train + validate
        │   ├── Save local checkpoint
        │   ├── Upload to runs/<run_id>/checkpoints/
        │   ├── Append runs/<run_id>/logs/metrics.csv
        │   └── Update runs/<run_id>/logs/training_state.json
        │
        └── Early stopping (patience=8, metric=val_macro_f1)
```

## File Listing

| File | Purpose |
|---|---|
| `split_manifest_manager.py` | Download-or-bootstrap manifest, path resolution |
| `extend_manifest.py` | Manually add new dataset shards to manifest |
| `kaggle_dataset_loader.py` | Manifest-based split assignment, DataLoaders |
| `hf_checkpoint_manager.py` | Per-run namespaced checkpoint management |
| `kaggle_train_resumable.py` | Main training orchestrator |
| `explore_kaggle_datasets.py` | Local: list datasets across all accounts |
| `list_active_runs.py` | Local: cross-run status dashboard |
| `README_KAGGLE.md` | This file |

## Troubleshooting

### "Manifest hash mismatch on resume"
- The manifest was extended/modified since this run started
- The run was trained on different data — this is expected behavior
- Resume will abort to prevent comparing incomparable results

### "Too many unresolved images"
- Not all 11 datasets are attached to this Kaggle session
- Attach all required shards in the Input panel

### "No training images resolved"
- Check that dataset mount paths match `KAGGLE_DATASETS` config
- Run `explore_kaggle_datasets.py` locally to verify dataset availability

### Session timeout mid-epoch
- The checkpoint for that epoch is NOT saved (only completed epochs)
- Next session resumes from the last completed epoch

### Parallel runs showing different results
- Check `list_active_runs.py` for manifest SHA-256 mismatch
- If mismatched, the runs used different data splits — not comparable
- Ensure all runs download the same manifest (first run bootstraps, rest download)
