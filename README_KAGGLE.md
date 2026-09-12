# Kaggle Multi-Session Training Pipeline

Resumable, multi-session training for MFFT on Kaggle with HuggingFace Hub persistence.

## Overview

This pipeline allows you to train MFFT across multiple Kaggle sessions (each limited to 9-12 hours) by persisting all checkpoints, metrics, and training state to a HuggingFace Hub repo. If a session disconnects, the next session automatically resumes from exactly where it left off.

```
Session 1: Epochs 1-12  ──► HF Hub ──► disconnect
Session 2: Epochs 13-24 ──► HF Hub ──► timeout
Session 3: Epochs 25-40 ──► training complete
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Kaggle Notebook                       │
│                                                         │
│  kaggle_dataset_loader.py                               │
│    ├── Auto-discovers mounted shards                    │
│    ├── Validates manifests (count, checksums)           │
│    └── Merges into unified DataFrame                    │
│                                                         │
│  hf_checkpoint_manager.py                               │
│    ├── Resume: download last.pt + training_state.json   │
│    ├── Per-epoch: save local → upload HF (with retry)   │
│    ├── Early stopping on val_macro_f1                   │
│    └── Session logging (GPU-hours)                      │
│                                                         │
│  kaggle_train_resumable.py                              │
│    ├── Clones MFFT repo (pinned commit)                │
│    ├── Wires dataset loader + checkpoint manager        │
│    ├── 40-epoch schedule, 8-epoch patience              │
│    └── Uploads final artifacts + ONNX export            │
│                                                         │
│                          │                              │
│                          ▼                              │
│              HuggingFace Hub Repo                       │
│         <username>/mfft-checkpoints                     │
└─────────────────────────────────────────────────────────┘
```

## Setup (One-Time)

### 1. Create HuggingFace Repo

```bash
# On huggingface.co, create a new Model repo:
#   Name: mfft-checkpoints
#   Type: Model (private recommended)
#   Repo ID: <username>/mfft-checkpoints
```

### 2. Create Kaggle Dataset Shards

Split your dataset into multiple shards (for multi-account storage limits):

```bash
# Package dataset (run locally)
python dataset/kaggle_package.py --max-images 50000 --output ./kaggle_dataset_shard_0
python dataset/kaggle_package.py --max-images 50000 --output ./kaggle_dataset_shard_1

# Upload each shard as a separate Kaggle Dataset
# (can be under different accounts)
kaggle datasets create -p ./kaggle_dataset_shard_0 --dir-mode zip
kaggle datasets create -p ./kaggle_dataset_shard_1 --dir-mode zip
```

### 3. Set HF Token as Kaggle Secret

1. Go to Kaggle → Account → Secrets
2. Click "Add a new secret"
3. Name: `HF_TOKEN`
4. Value: Your HuggingFace write token (get from https://huggingface.co/settings/tokens)
5. Click "Save"

### 4. Create Kaggle Notebook

1. Create new Kaggle Notebook
2. **Settings → Accelerator**: GPU T4 x2 (or P100)
3. **Settings → Internet**: On
4. **Add Input**: Attach all dataset shards (from step 2)

### 5. Configure Script

Edit the CONFIG section in `kaggle_train_resumable.py`:

```python
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/ai-image-detection-research.git"
GITHUB_COMMIT_SHA = "main"  # or specific tag/commit
HF_REPO_ID = "YOUR_USERNAME/mfft-checkpoints"
EXPECTED_SHARDS = ["mfft-dataset-shard-0", "mfft-dataset-shard-1"]
MAX_EPOCHS = 40
PATIENCE = 8
```

### 6. Run

Copy all 3 files to your Kaggle notebook:
- `kaggle_dataset_loader.py`
- `hf_checkpoint_manager.py`
- `kaggle_train_resumable.py`

Run `kaggle_train_resumable.py`.

## HuggingFace Repo Structure

```
<username>/mfft-checkpoints/
├── README.md                          # Model card (auto-updated)
├── checkpoints/
│   ├── last.pt                        # Most recent epoch (overwritten)
│   ├── best.pt                        # Best val_macro_f1 (overwritten)
│   ├── epoch_10.pt                    # Periodic snapshot
│   ├── epoch_20.pt                    # Periodic snapshot
│   ├── epoch_30.pt                    # Periodic snapshot
│   └── epoch_40.pt                    # Periodic snapshot
├── logs/
│   ├── training_state.json            # Resume source of truth
│   ├── metrics.csv                    # Per-epoch metrics history
│   └── session_log.csv               # Session start/end + GPU-hours
└── results/
    ├── final_model.pt                 # Final weights
    ├── model.onnx                     # ONNX export
    ├── final_metrics.json             # All metrics
    ├── manuscript_ready_metrics.json  # Numbers for paper [TBD] fields
    └── figures/                       # Training curves, ROC, etc.
```

## Resume After Disconnect

The resume process is **idempotent** — re-running the notebook always picks up exactly where it left off:

```
Session starts
    │
    ├── Download logs/training_state.json from HF Hub
    │   ├── Found & status=running?
    │   │   ├── YES → Download checkpoints/last.pt
    │   │   │   ├── Restore model, optimizer, scheduler state
    │   │   │   ├── Resume from epoch = last_epoch + 1
    │   │   │   └── Continue training
    │   │   └── NO (status=completed/early_stopped?)
    │   │       └── Start fresh run
    │   └── Not found?
    │       └── Start fresh run
    │
    └── Training loop (40 epochs max)
        │
        ├── Per epoch:
        │   ├── Train + validate
        │   ├── Save local checkpoint
        │   ├── Upload checkpoints/last.pt (overwrite)
        │   ├── If epoch % 10 == 0 → upload snapshot
        │   ├── If improved → upload best.pt
        │   ├── Append metrics.csv
        │   └── Update + upload training_state.json
        │
        └── Early stopping (patience=8, metric=val_macro_f1)
```

## Metrics Tracked

| Metric | Purpose |
|---|---|
| `val_macro_f1` | **Primary** — best checkpoint + early stopping decision |
| `val_auc` | Tiebreaker when macro_f1 is equal |
| `val_loss` | Logged for manuscript figures |
| `val_accuracy` | Logged for manuscript figures |

**Why macro-F1?** The dataset is imbalanced (47.4% real / 43.8% AI / 9% deepfake). Macro-F1 weights all three classes equally, preventing the majority classes from dominating the metric.

## Per-Epoch Upload Sequence

After every epoch:

1. **Save local** → `checkpoints/last.pt`
2. **Upload** → `checkpoints/last.pt` (overwrite)
3. **Every 10th epoch** → also upload `checkpoints/epoch_<N>.pt` (permanent)
4. **If val_macro_f1 improved** → also upload `checkpoints/best.pt`
5. **Append** → `logs/metrics.csv` (all 4 metrics)
6. **Update** → `logs/training_state.json` (epoch, best_metric, counters)

## Error Handling

- All HF uploads use exponential backoff retry (max 5 attempts)
- `training_state.json` is written atomically (temp file → verify → rename)
- Corrupted checkpoints are detected and training starts fresh
- Missing shards fail loudly with a clear error message
- Session start/end timestamps + GPU-hours logged to `session_log.csv`

## Multi-Shard Dataset Loading

The `kaggle_dataset_loader.py` module:

1. **Discovers** all mounted datasets under `/kaggle/input/`
2. **Validates** each shard:
   - Required columns: `filename`, `label`
   - Minimum image count
   - Checksum verification (optional)
   - Missing/zero-byte image detection
3. **Merges** into a single DataFrame with columns:
   - `path` — absolute image path
   - `label` — 0 (real), 1 (ai_generated), 2 (deepfake)
   - `label_name` — human-readable
   - `source` — generator/source dataset
   - `shard` — which shard it came from

## Kaggle Session Limits

| Limit | Value |
|---|---|
| Max runtime (GPU) | 9 hours |
| Max runtime (CPU) | 12 hours |
| Working disk | ~20 GB |
| VRAM (T4) | 16 GB |
| VRAM (P100) | 16 GB |

With 50K images, IMAGE_SIZE=224, BATCH_SIZE=64:
- ~500 batches/epoch
- ~3-5 min/epoch on T4
- ~10-15 epochs per session
- ~3-4 sessions to complete 40 epochs

## File Listing

| File | Purpose |
|---|---|
| `kaggle_dataset_loader.py` | Multi-shard discovery, validation, merge |
| `hf_checkpoint_manager.py` | Save/load/upload/resume with retry logic |
| `kaggle_train_resumable.py` | Main training orchestrator |
| `README_KAGGLE.md` | This file |

## Troubleshooting

### "HF_TOKEN not found"
- Ensure the Kaggle Secret is named exactly `HF_TOKEN`
- Ensure "Access internet" is enabled in notebook settings
- Try: `from kaggle_secrets import UserSecretsClient; UserSecretsClient().get_secret("HF_TOKEN")`

### "Missing/empty expected shards"
- Check the Kaggle Input panel has all shards attached
- Check shard slug names match `EXPECTED_SHARDS` config exactly
- Shards appear as `/kaggle/input/<slug>/`

### "No improvement for 8 consecutive epochs"
- This is expected early stopping behavior
- The best model is saved at `checkpoints/best.pt`
- Increase `PATIENCE` if you want longer training

### Session timeout mid-epoch
- The checkpoint for that epoch is NOT saved (only completed epochs are saved)
- Next session resumes from the last completed epoch
- Partially-uploaded files are safe (HF Hub handles partial writes)

### "Checksum mismatch"
- A shard was modified after upload
- Re-upload the shard or remove the checksum from the expected manifest
