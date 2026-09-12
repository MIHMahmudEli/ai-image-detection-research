# DGX Full-Scale Run Guide

> **Last validated:** 2026-09-04 — all images present, 1,424,326 manifest rows, 0 missing/zero-byte files in 20K sample.

---

## Table of Contents

1. [Quick Summary](#1-quick-summary)
2. [Dataset Status](#2-dataset-status)
3. [Step 1: Transfer to DGX](#3-step-1-transfer-to-dgx)
4. [Step 2: Install Dependencies](#4-step-2-install-dependencies)
5. [Step 3: Build Training Manifest](#5-step-3-build-training-manifest)
6. [Step 4: Verify on DGX Before Training](#6-step-4-verify-on-dgx-before-training)
7. [Step 5: Run Notebooks (in order)](#7-step-5-run-notebooks-in-order)
8. [Step 6: Monitor Training](#8-step-6-monitor-training)
9. [Step 7: Retrieve Results](#9-step-7-retrieve-results)
10. [Run Modes Reference](#10-run-modes-reference)
11. [LOGO Protocol](#11-logo-protocol)
12. [Troubleshooting](#12-troubleshooting)
13. [Output Locations](#13-output-locations)

---

## 1. Quick Summary

```
Transfer repo + dataset (~166GB) → DGX
  ↓
Install deps (pip install -r model/requirements.txt)
  ↓
Build manifest (python dataset/scripts/prepare_training_manifest.py)
  ↓
Verify (python dataset/validate_for_dgx.py)
  ↓
Run 8 notebooks in order:
  1. train_mfft_base.ipynb      ← creates shared split
  2. train_mfft_tiny.ipynb
  3. train_mfft_large.ipynb
  4. train_baselines.ipynb      ← 10 baselines
  5. train_ablation_study.ipynb ← 9+ ablations
  6. paper_evals.ipynb          ← CIs, McNemar, calibration
  7. train_logo.ipynb           ← cross-generator eval
  8. generate_paper_outputs.ipynb ← all figures + tables
  ↓
Download checkpoints + paper/results/
```

**Total GPU time: ~35-50 hours** (can split across sessions)

---

## 2. Dataset Status

**Validated locally on 2026-09-04:**

| Metric | Value | Status |
|--------|-------|--------|
| `train_manifest.csv` rows | 1,424,326 | ✅ |
| `clean_metadata.csv` rows | 2,812,408 | ✅ |
| Images sampled (20K) | 20,000/20,000 found | ✅ |
| Missing files in sample | 0 | ✅ |
| Zero-byte files in sample | 0 | ✅ |
| `split_indices.json` | Will be created by notebook 1 | ✅ expected |
| `split_indices_smoke.json` | 600 samples (CPU mode) | ✅ |
| `split_indices_quick5k.json` | 5,000 samples (QUICK_5K mode) | ✅ |

### Label Distribution (train_manifest.csv)

| Label | Count | % |
|-------|-------|---|
| real | 675,479 | 47.4% |
| ai_generated | 623,720 | 43.8% |
| deepfake | 125,127 | 8.8% |

### Source Distribution

| Source | Count | Type |
|--------|-------|------|
| places365 | 400,000 | Real (capped) |
| ntire2026 | 277,643 | Mixed |
| genimage_biggan | 162,000 | AI |
| glide | 162,000 | AI |
| celebdf | 101,031 | Deepfake |
| stable_diffusion | 100,000 | AI |
| imagenet | 98,001 | Real |
| celebdf_v2_real | 50,360 | Real |
| pexels_unsplash | 25,000 | Real |
| faceforensics | 20,351 | Deepfake |
| dalle3 | 18,998 | AI |
| dfdc | 3,745 | Deepfake |
| midjourney | 3,079 | AI |
| dfdc_real | 1,518 | Real |
| open_images_v7 | 600 | Real |

### Known Gap

No REAL frames found for **FaceForensics++** — its 20,351 fake frames have no real counterparts. This creates a "face => fake" shortcut. To fix: extract frames from FF++ `original_sequences` into `dataset/images/FaceForensics/` and re-run `prepare_training_manifest.py`.

---

## 3. Step 1: Transfer to DGX

### What to Transfer

```
ai-image-detection-research/
├── model/                 ← Python code + notebooks
│   ├── src/               ← All .py modules
│   ├── *.ipynb            ← 8 training notebooks
│   ├── test_model_verify/ ← Local verification notebooks
│   └── requirements.txt
├── dataset/
│   ├── images/            ← ~166GB, ALL subdirectories
│   │   ├── BigGAN/
│   │   ├── CelebDF_V2/
│   │   ├── DALL-E3/
│   │   ├── DFDC/
│   │   ├── FaceForensics/
│   │   ├── genimage_ai/
│   │   ├── Midjourney/
│   │   ├── NTIRE2026/
│   │   ├── Open-Images-V7-Dataset/
│   │   ├── Places365/
│   │   ├── real/
│   │   └── Stable Diffusion/
│   ├── metadata/          ← CSV files
│   │   ├── clean_metadata.csv
│   │   ├── train_manifest.csv
│   │   └── *.json split files
│   └── scripts/           ← Collection/processing scripts
├── paper/                 ← Manuscript + figures
├── api/                   ← FastAPI server
└── DGX_RUN_GUIDE.md       ← This file
```

### Transfer Methods

**Option A: rsync (recommended)**
```bash
# From your local machine:
rsync -avzP --progress \
  /path/to/ai-image-detection-research/ \
  dgx-user@dgx-host:/workspace/ai-image-detection-research/

# Or just the dataset if repo is already there:
rsync -avzP --progress \
  /path/to/ai-image-detection-research/dataset/ \
  dgx-user@dgx-host:/workspace/ai-image-detection-research/dataset/
```

**Option B: scp**
```bash
scp -r /path/to/ai-image-detection-research/ dgx-user@dgx-host:/workspace/
```

**Option C: External drive**
```bash
# Mount external drive on DGX, then symlink:
ln -s /mnt/external/ai-image-detection-research /workspace/ai-image-detection-research
```

### Verify Transfer

```bash
# On the DGX, check critical paths exist:
ls -la /workspace/ai-image-detection-research/dataset/images/
# Should show 12 directories: BigGAN CelebDF_V2 DALL-E3 DFDC FaceForensics
#   genimage_ai Midjourney NTIRE2026 Open-Images-V7-Dataset Places365
#   real Stable Diffusion

ls -la /workspace/ai-image-detection-research/model/src/
# Should show: __init__.py baselines.py calibration.py config.py dataset.py
#   evaluate.py logo_eval.py model.py robustness.py stats.py train.py
#   train_baselines.py train_variants.py visualize.py
```

---

## 4. Step 2: Install Dependencies

```bash
cd /workspace/ai-image-detection-research

# Core dependencies
pip install -r model/requirements.txt

# Additional dependencies (for baselines + ablation)
pip install open_clip_torch    # CLIP baseline
pip install datasets           # HuggingFace datasets (for some baselines)
pip install scipy              # McNemar test
```

### Verify Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
import torchvision
print(f'torchvision: {torchvision.__version__}')
import sklearn
print(f'scikit-learn: {sklearn.__version__}')
import pandas
print(f'pandas: {pandas.__version__}')
"
```

Expected output on DGX:
```
PyTorch: 2.6.0
CUDA: True
GPU: NVIDIA A100 / V100 / T4
VRAM: 16.0 / 32.0 / 40.0 GB
```

---

## 5. Step 3: Build Training Manifest

```bash
cd /workspace/ai-image-detection-research

# This drops BigGAN duplicates, caps Places365 at 400K,
# adds deepfake REAL frames, dedupes by MD5, and writes
# dataset/metadata/train_manifest.csv (~1.4M rows)
python dataset/scripts/prepare_training_manifest.py
```

**What it does:**
1. Removes duplicate BigGAN images (161,995 rows)
2. Caps Places365 at 400,000 images
3. Adds real frames from CelebDF/DFDC deepfake datasets
4. Deduplicates by MD5 hash
5. Writes `train_manifest.csv`

**Expected output:**
```
Loading clean_metadata.csv ...
  2,812,408 rows
  Dropped 161,995 'biggan' rows duplicated in 'genimage_biggan'
  Capped places365: 1,839,962 -> 400,000
  Added 51,878 deepfake-dataset REAL frames
  Removed X MD5 duplicates

Final composition:
ai_generated    623720
deepfake        125127
real            675479

Wrote 1,424,326 rows to dataset/metadata/train_manifest.csv
```

---

## 6. Step 4: Verify on DGX Before Training

**This is the critical step — DO NOT skip it.** Run this before burning GPU time:

```bash
cd /workspace/ai-image-detection-research
python dataset/validate_for_dgx.py
```

**What it checks:**
- All images referenced in `train_manifest.csv` exist on disk
- No zero-byte files
- Correct label distribution
- Split file compatibility

**Expected output:**
```
IMAGE DIRECTORIES:
  BigGAN                           98,000
  CelebDF_V2                      101,031
  ...
  real                              25,000
  Stable Diffusion                100,001

train_manifest.csv: 1,424,326 rows
  Labels: {'real': 675479, 'ai_generated': 623720, 'deepfake': 125127}

File check (20,000 sampled):
  OK:        20,000
  Missing:   0
  Zero-byte: 0

split_indices.json: NOT FOUND  ← Will be created by notebook 1

ALL CHECKS PASSED — ready for DGX!
```

**If you see issues, fix them before proceeding:**
- Missing images → re-transfer the dataset
- Zero-byte files → re-transfer specific directories
- Wrong row count → re-run `prepare_training_manifest.py`

---

## 7. Step 5: Run Notebooks (in order)

### Pre-flight Check (Every Notebook)

When you open any notebook, **check Cell 1 output**:
```
Device: cuda
SMOKE_TEST: False     ← Must be False on DGX
QUICK_5K: False       ← Must be False for full-scale
Dataset loaded: 1,424,326 samples  ← Must match manifest
```

If `SMOKE_TEST: True`, the GPU is not detected. Fix: `nvidia-smi` to verify GPU.

### Notebook Execution Order

```bash
cd /workspace/ai-image-detection-research/model

# Run each notebook. Options:
# A) Jupyter (if you have jupyter installed):
jupyter notebook --ip=0.0.0.0 --no-browser --port=8888

# B) Convert and run as script:
jupyter nbconvert --to script train_mfft_base.ipynb --stdout | python

# C) Run directly in terminal (if using VS Code / JupyterLab):
# Just open each .ipynb and Run All
```

### What Each Notebook Does

---

#### Notebook 1: `train_mfft_base.ipynb` — START HERE

| Cell | Purpose |
|------|---------|
| 1 | Imports, GPU detection, sets `SMOKE_TEST = False` |
| 2 | Config: 20 epochs, 384px, batch 64, 8 workers |
| 3 | **Creates shared train/val/test split** → `split_indices.json` |
| 4 | Builds MFFT-Base model (1.62M params) |
| 5 | Sanity check: one forward/backward pass |
| 6 | Quick validation on 3 batches |
| 7 | **Full training loop** (20 epochs with AMP) |
| 8 | Saves final model |
| 9 | Generates manuscript figures (15 figures) |
| 10 | Per-category accuracy breakdown |
| 11 | Saves all metrics as JSON + CSV tables |

**Produces:**
- `model/checkpoints/base_model/best_mfft_base.pt`
- `model/checkpoints/base_model/mfft_base_final.pt`
- `paper/result/full_scale/base_model/fig/` (15 figures)
- `paper/result/full_scale/base_model/table/` (9 tables)
- `dataset/metadata/split_indices.json` ← **SHARED by all later notebooks**

**Time:** ~3-4 hours

---

#### Notebook 2: `train_mfft_tiny.ipynb`

Same structure as base. Produces tiny model checkpoint (372K params).

**Produces:** `model/checkpoints/tiny_model/best_mfft_tiny.pt`
**Time:** ~2-3 hours

---

#### Notebook 3: `train_mfft_large.ipynb`

Same structure. Produces large model checkpoint (6.30M params).

**Produces:** `model/checkpoints/large_model/best_mfft_large.pt`
**Time:** ~5-6 hours

---

#### Notebook 4: `train_baselines.ipynb`

Trains all 10 baseline models:
- SimpleCNN, LightViT
- ResNet-18, ResNet-50
- EfficientNet-B0
- ViT-B/16, Swin-T
- CLIP ViT-B/32
- FreqDetect, DeiT-Small

Each trained for 20 epochs. Results saved to `paper/result/full_scale/baselines_model/`.

**Produces:** 10 baseline checkpoints
**Time:** ~8-12 hours (10 models × ~1h each)

---

#### Notebook 5: `train_ablation_study.ipynb`

Runs 9+ ablation configurations:
- spatial_only (no frequency decomposition)
- no_fga (no frequency-guided attention)
- fusion_avg, fusion_max, fusion_concat (alternative fusion)
- skip_low, skip_high, skip_mid (band ablation)
- +pretrained (ImageNet-pretrained extractors)
- +NPR (neighboring-pixel-relation residual)
- +soft (soft band masks)
- combined (all upgrades)

**Produces:** Ablation comparison tables
**Time:** ~10-15 hours

---

#### Notebook 6: `paper_evals.ipynb`

Evaluation-only notebook (no training):
- Bootstrap confidence intervals (1000 resamples)
- McNemar significance tests (MFFT vs each baseline)
- Temperature-scaled calibration
- Robustness curves (JPEG/downscale/blur)
- Per-generator accuracy breakdown

**Requires:** Checkpoints from notebooks 1-2
**Produces:** Statistical evaluation tables
**Time:** ~1-2 hours

---

#### Notebook 7: `train_logo.ipynb`

Leave-One-Generator-Out (LOGO) protocol:
- For each of 8 generators: train MFFT-Base on everything EXCEPT that generator
- Evaluate on held-out generator + unseen real images
- Writes `paper/result/full_scale/logo/logo_summary.csv`

**Generators:** BigGAN, Glide, Stable Diffusion, DALL-E3, Midjourney, Celeb-DF, FaceForensics, DFDC

**Budget:** `LOGO_EPOCHS = 10` by default (8 runs)
**Checkpoints:** `model/checkpoints/logo/<generator>/best.pt`
**Time:** ~6-8 hours

---

#### Notebook 8: `generate_paper_outputs.ipynb` — RUN LAST

Re-evaluates all 3 MFFT variants on the shared test split. Generates:
- All 15 manuscript figures
- All 9 manuscript tables
- `mfft_summary.csv` + `all_models_summary.csv`

**Produces:** Complete paper outputs in `paper/result/full_scale/`
**Time:** ~1-2 hours

---

## 8. Step 6: Monitor Training

### GPU Monitoring

```bash
# Check GPU utilization (should show 80-100% during training)
watch -n 1 nvidia-smi

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Training Log Output

The notebooks print epoch-by-epoch metrics:
```
Epoch 1/20 | Train: 0.3421/87.65% | Val: 0.2134/91.23% | F1: 90.87 AUC: 0.9634
Epoch 2/20 | Train: 0.1892/93.21% | Val: 0.1567/94.56% | F1: 94.12 AUC: 0.9789
...
```

### Checkpoint Sizes

```bash
ls -lh model/checkpoints/*_model/
# best_mfft_base.pt   ~6.5MB
# best_mfft_tiny.pt   ~1.5MB
# best_mfft_large.pt  ~25MB
```

### If Training Crashes

The notebooks have try/except at batch level — bad batches are skipped. If the crash is catastrophic:
1. Check GPU memory: `nvidia-smi` (if OOM, reduce batch size in Cell 2)
2. Check disk space: `df -h` (need ~50GB free for checkpoints + outputs)
3. Re-run the notebook — it will resume from the last saved checkpoint

---

## 9. Step 7: Retrieve Results

```bash
# From the DGX, copy back:

# Checkpoints (~33MB total)
scp -r dgx-host:/workspace/ai-image-detection-research/model/checkpoints/ \
  ./model/checkpoints/

# Paper results + figures
scp -r dgx-host:/workspace/ai-image-detection-research/paper/result/full_scale/ \
  ./paper/result/

# Or rsync everything:
rsync -avzP dgx-host:/workspace/ai-image-detection-research/paper/result/full_scale/ \
  ./paper/result/full_scale/
```

---

## 10. Run Modes Reference

| Mode | How to activate | Data | Epochs | Image Size | Split File |
|------|----------------|------|--------|------------|------------|
| Smoke | Run on CPU machine (auto) | 600 balanced | 1 | 224px | `split_indices_smoke.json` |
| QUICK_5K | Set `QUICK_5K = True` in Cell 1 | 5,000 balanced | 1 | 224px | `split_indices_quick5k.json` |
| **Full** | Run on DGX (auto) | ~1.42M | 20 | 384px | `split_indices.json` |

### QUICK_5K Mode (Fast Comparative Runs)

For quick A/B testing without burning full GPU time:
- 5,000 balanced samples (2,500 real / 2,500 AI)
- 1 epoch, 224px, batch 32
- Runs all models and experiments
- Uses `test_train_manifest.csv` + `split_indices_quick5k.json`

```bash
# Regenerate 5K manifest:
python dataset/scripts/prepare_test_manifest.py
```

---

## 11. LOGO Protocol

**Purpose:** Test cross-generator generalization (the paper's headline result).

**How it works:**
1. For each of 8 generators, hold out all images from that generator
2. Train MFFT-Base from scratch on remaining data
3. Evaluate on held-out generator + unseen real images
4. Report balanced accuracy per generator

**Generators tested:**
- BigGAN, Glide, Stable Diffusion, DALL-E3, Midjourney
- Celeb-DF, FaceForensics, DFDC

**Budget:** 10 epochs per generator × 8 generators = ~80 epochs total
**Checkpoints:** `model/checkpoints/logo/<generator>/best.pt`
**Output:** `paper/result/full_scale/logo/logo_summary.csv`

The **mean balanced accuracy** in that table is the paper's headline generalization number.

---

## 12. Troubleshooting

### Problem: `SMOKE_TEST: True` on DGX

**Cause:** GPU not detected.
```bash
# Check GPU:
nvidia-smi

# If no GPU output, check CUDA:
python -c "import torch; print(torch.cuda.is_available())"
```

### Problem: Dataset load shows 0 samples

**Cause:** Images missing on disk or wrong metadata path.
```bash
# Check:
ls dataset/images/           # Should show 12 directories
wc -l dataset/metadata/train_manifest.csv  # Should show ~1.4M
python dataset/validate_for_dgx.py         # Full check
```

### Problem: `split_indices.json` sample count mismatch

**Cause:** Manifest was regenerated after split was created.
```bash
# Delete the split file — it will be recreated by notebook 1:
rm dataset/metadata/split_indices.json
```

### Problem: Out of GPU memory (OOM)

**Cause:** Batch size too large for GPU VRAM.
```
# In Cell 2 of the notebook, reduce batch size:
cfg.training.batch_size = 32  # instead of 64
```

### Problem: Training is very slow

**Cause:** AMP not enabled or too few workers.
```bash
# Check:
python -c "import torch; print(torch.cuda.is_available())"
# Should print True

# In Cell 2:
cfg.training.mixed_precision = True  # AMP
cfg.training.num_workers = 8         # DataLoader workers
```

### Problem: `train_manifest.csv` has wrong row count

**Cause:** Manifest built from stale `clean_metadata.csv`.
```bash
# Rebuild:
python dataset/scripts/prepare_training_manifest.py

# Then delete stale split:
rm dataset/metadata/split_indices.json
```

### Problem: FaceForensics "face => fake" shortcut

**Cause:** No REAL frames from FaceForensics++ in the dataset.
```
# To fix: extract frames from FF++ original_sequences:
# dataset/images/FaceForensics/original_sequences/
# Then re-run:
python dataset/scripts/prepare_training_manifest.py
```

---

## 13. Output Locations

### Checkpoints

| Path | Contents |
|------|----------|
| `model/checkpoints/tiny_model/` | MFFT-Tiny (372K params) |
| `model/checkpoints/base_model/` | MFFT-Base (1.62M params) |
| `model/checkpoints/large_model/` | MFFT-Large (6.30M params) |
| `model/checkpoints/logo/<gen>/` | LOGO per-generator models |
| `model/checkpoints/verify/` | Local verification models (not used in paper) |

### Paper Results

| Path | Contents |
|------|----------|
| `paper/result/full_scale/base_model/fig/` | 15 manuscript figures |
| `paper/result/full_scale/base_model/table/` | 9 manuscript tables (JSON + CSV) |
| `paper/result/full_scale/baselines_model/` | Baseline comparison results |
| `paper/result/full_scale/ablation/` | Ablation study results |
| `paper/result/full_scale/logo/` | LOGO summary (`logo_summary.csv`) |

### Legacy Results (DO NOT USE)

| Path | Notes |
|------|-------|
| `paper/result/test/` | Pilot results (frozen, 1-epoch, 5K images) |
| `paper/result/verify/` | Local smoke verification |
| `paper/results/` | Legacy path (deprecated) |

---

## Quick Copy-Paste: Full DGX Session

```bash
# === ON YOUR LOCAL MACHINE ===
# Transfer
rsync -avzP /path/to/ai-image-detection-research/ dgx-host:/workspace/ai-image-detection-research/

# === ON THE DGX ===
cd /workspace/ai-image-detection-research

# Install
pip install -r model/requirements.txt
pip install open_clip_torch scipy

# Build manifest
python dataset/scripts/prepare_training_manifest.py

# Verify
python dataset/validate_for_dgx.py
# Must show: ALL CHECKS PASSED

# Run notebooks (in order)
cd model
# 1. Base (creates shared split)
jupyter nbconvert --to script train_mfft_base.ipynb --stdout | python
# 2. Tiny
jupyter nbconvert --to script train_mfft_tiny.ipynb --stdout | python
# 3. Large
jupyter nbconvert --to script train_mfft_large.ipynb --stdout | python
# 4. Baselines (10 models)
jupyter nbconvert --to script train_baselines.ipynb --stdout | python
# 5. Ablation (9+ configs)
jupyter nbconvert --to script train_ablation_study.ipynb --stdout | python
# 6. Paper evals
jupyter nbconvert --to script paper_evals.ipynb --stdout | python
# 7. LOGO
jupyter nbconvert --to script train_logo.ipynb --stdout | python
# 8. Paper outputs (LAST)
jupyter nbconvert --to script generate_paper_outputs.ipynb --stdout | python

# === BACK ON LOCAL MACHINE ===
# Retrieve results
rsync -avzP dgx-host:/workspace/ai-image-detection-research/model/checkpoints/ ./model/checkpoints/
rsync -avzP dgx-host:/workspace/ai-image-detection-research/paper/result/full_scale/ ./paper/result/full_scale/
```
