# DGX Full-Scale Run Guide

> The notebooks auto-detect the environment: **no GPU = smoke-verification mode**
> (600-image balanced subset, 1 epoch, 224px — for checking that everything runs),
> **GPU present = full paper protocol** (full manifest, 20 epochs, 384px, batch 64,
> AMP, 8 workers). You never edit a flag; just run on the right machine.
>
> There is also an opt-in **QUICK_5K mode** (see §4) for fast comparative runs.

## Run modes at a glance

| Mode | How | Data | Epochs | Size | Split file | Manifest |
|---|---|---|---|---|---|---|
| Smoke (auto, CPU) | run on a no-GPU machine | 600 balanced | 1 | 224px | `split_indices_smoke.json` | best available |
| **QUICK_5K** (opt-in) | set `QUICK_5K = True` in Cell 1 | 5,000 balanced | 1 | 224px | `split_indices_quick5k.json` | `test_train_manifest.csv` |
| Full (auto, GPU) | run on the DGX | full manifest (~1.31M) | 20 | 384px | `split_indices.json` | `train_manifest.csv` |

## 0. One-time setup on the DGX

```bash
# 1. Transfer the repo AND the dataset (~166 GB, local-disk-only — it is NOT
#    in git). dataset/images/ and dataset/metadata/ must land on the DGX.
# 2. Install deps:
python -m pip install -r model/requirements.txt

# 3. Build the confound-fixed manifest (safe to re-run; CSV + folder scan only):
python dataset/scripts/prepare_training_manifest.py
```

`prepare_training_manifest.py` drops the duplicated BigGAN source (the same
GenImage BigGAN images exist on disk twice, under `BigGAN/` and
`genimage_ai/BigGAN/`, with no MD5s — 161,995 rows removed by basename match),
caps Places365 at 400K, adds deepfake-dataset REAL frames, dedupes by MD5, and
writes `dataset/metadata/train_manifest.csv`. The notebooks pick it up
automatically; if it is absent they fall back to `clean_metadata.csv` and print
a warning. (`clean_metadata.csv` still contains both BigGAN copies — never
train the full protocol directly on it, or duplicates can leak across the
train/test split.)

**Status (2026-07-05, built locally — rebuild on the DGX after transfer):**
1,146,683 rows. Composition: 575,479 real (400K Places365 cap, 98K ImageNet,
25K Pexels/Unsplash, 50,360 CelebDF-v2 real + 1,518 DFDC real frames),
446,077 AI-generated (162K GenImage-BigGAN, 162K GLIDE, 100K Stable Diffusion,
19K DALL·E 3, 3K Midjourney), 125,127 deepfake.

**Known gap:** no REAL frames were found for **FaceForensics++** — its 20,351
fake frames have no real counterparts in the manifest (CelebDF/DFDC reals were
found and added). To close the residual "face ⇒ fake" shortcut completely,
extract frames from FF++ `original_sequences` into
`dataset/images/FaceForensics/` and re-run the script. If you regenerate the
manifest **after** a split file exists, delete `dataset/metadata/split_indices.json`
so it is rebuilt against the new manifest (the loader hard-fails on a
sample-count mismatch, so this cannot corrupt results silently).

**Check the first cell output of every notebook:** `SMOKE_TEST: False`,
`QUICK_5K: False`, and the dataset load must report **~1.15M samples** (the
train_manifest count — not the raw 2.7M of clean_metadata). If the count is
much lower, image files are missing on disk — stop and fix before burning GPU
time.

## 1. Run order (all in `model/`)

| # | Notebook | Produces | Time driver |
|---|---|---|---|
| 1 | `train_mfft_tiny.ipynb` | Tiny checkpoint + figures/tables | 20 ep full data |
| 2 | `train_mfft_base.ipynb` | Base checkpoint + figures/tables | 20 ep |
| 3 | `train_mfft_large.ipynb` | Large checkpoint + figures/tables | 20 ep |
| 4 | `train_baselines.ipynb` | all 10 baselines | 10 × 20 ep |
| 5 | `train_ablation_study.ipynb` | 9 ablations + 4 upgrade rows | 13 × 20 ep |
| 6 | `paper_evals.ipynb` | CIs, McNemar, calibration, robustness, per-generator, LOGO manifests | eval only (needs #1–2 checkpoints) |
| 7 | `train_logo.ipynb` | full LOGO protocol: one model per held-out generator, `logo_summary.csv` | ~8 × 10 ep |
| 8 | `generate_paper_outputs.ipynb` | re-evaluates MFFT tiny/base/large on the shared test split, manuscript figures, `mfft_summary.csv` + `all_models_summary.csv` aggregating every result under `paper/result/full_scale/` | eval only, run **last** |

The **first** notebook creates `dataset/metadata/split_indices.json`; every later
notebook reuses it, so all models share the identical train/val/test split.
Do **not** delete that file between notebooks.

> `generate_paper_outputs.ipynb` supersedes the legacy
> `generate_paper_outputs.py` (which used the old `paper/results/` path and its
> own random split instead of the shared one — do not use the .py script).

**Outputs:** checkpoints → `model/checkpoints/<variant>_model/`,
results/figures → `paper/result/full_scale/...`.

⚠️ **Never run the `model/train_*.ipynb` set on a CPU machine.** They would
drop into smoke mode but still write 1-epoch weights into the *full-scale*
checkpoint paths, polluting them. CPU testing belongs in
`model/test_model_verify/` (quarantined `verify/` paths).

## 2. LOGO (cross-generator) training — `train_logo.ipynb`

Fully automated: for each of the 8 generators it (re)builds the LOGO manifests
if missing, trains MFFT-Base from scratch on everything except that generator,
evaluates on the held-out generator + unseen real images, and writes
`paper/result/full_scale/logo/logo_summary.csv` — the mean balanced accuracy in
that table is the paper's headline generalization number.

Budget: `LOGO_EPOCHS = 10` by default (8 runs; acceptable — state it in the
paper). Set it to 20 in Cell 2 if your allocation allows. Checkpoints go to
`model/checkpoints/logo/<generator>/best.pt` and never touch the main run.

## 3. Verifying locally BEFORE requesting GPU time

Run the notebooks in `model/test_model_verify/` on your own machine (no GPU
= smoke mode automatically). They exercise every code path end to end in
minutes and write to quarantined locations:

- results → `paper/result/verify/...` (pilot results in `paper/result/test/` are untouched)
- checkpoints → `model/checkpoints/verify/...`

Order: `mfft_tiny` → `mfft_base` → `mfft_large` → `baselines` → `ablation` →
`paper_evals` → `train_logo` → `generate_paper_outputs`.
If all finish without a red cell, the code is ready for the DGX.

✅ **Status 2026-07-05:** the full verify suite completed in QUICK_5K mode —
checkpoints for all 3 MFFT variants, all 10 baselines, and 8 LOGO generators
exist under `model/checkpoints/verify/`.

## 4. QUICK_5K mode (fast comparative runs)

Every notebook in `model/test_model_verify/` has a `QUICK_5K` flag in Cell 1
(currently `False`). Setting it to `True` gives a *comparable* quick protocol
that — unlike smoke mode — runs **all** models and experiments:

- 5,000-sample balanced subset (2,500 real / 2,500 AI, seed 42),
  split 4,000/500/500, 1 epoch, 224px, batch 32
- all 10 baselines, all ~14 ablations, all LOGO generators
- dedicated manifest `dataset/metadata/test_train_manifest.csv` (fast load:
  5K rows instead of a 2.7M-row scan) + shared `split_indices_quick5k.json`,
  so every model sees the identical data

Regenerate the 5K manifest with:

```bash
python dataset/scripts/prepare_test_manifest.py
```

It replicates the seeded subset draw exactly and self-verifies against the
split file. If you change `clean_metadata.csv`, delete
`test_train_manifest.csv` **and** `split_indices_quick5k.json`, then re-run it.

QUICK_5K also works **on a GPU** (e.g. Kaggle T4): the whole suite fits in a
single free session. Only the ~1–3 GB of images listed in
`test_train_manifest.csv` need to be uploaded, not the full 166 GB.

## 5. What was fixed in the notebooks (summary)

1. Seed 42 everywhere; splits stratified and **saved/shared** across notebooks
   (previously: unseeded undersampling gave every notebook a different test set).
2. Weighted sampler instead of undersampling (keeps all real images).
3. `train_manifest.csv` support (Places365 cap + deepfake real frames).
4. AMP (autocast + GradScaler) in every training loop; batch 64 / workers 8 on GPU.
5. Demo cells replaced by fast sanity checks (previously trained a full stray epoch).
6. Full loop rebuilds a fresh model (previously reused the demo-trained model).
7. Ablation notebook gained 4 upgrade rows (+pretrained / +NPR / +soft / combined).
8. Baselines/ablation train a small subset in smoke mode.
9. Per-category AUC bug fixed (was computed on binary predictions).
10. 2.7M-row `iterrows` metadata map replaced with a fast dict build.
11. QUICK_5K mode added to all verify notebooks (5K subset, 1 epoch, all models).
12. `generate_paper_outputs.ipynb` added (full_scale + verify copies) using the
    shared split; legacy `generate_paper_outputs.py` deprecated.
