# DGX Full-Scale Run Guide

> The notebooks auto-detect the environment: **no GPU = smoke-verification mode**
> (600-image balanced subset, 1 epoch, 224px — for checking that everything runs),
> **GPU present = full paper protocol** (full manifest, 20 epochs, 384px, batch 64,
> AMP, 8 workers). You never edit a flag; just run on the right machine.

## 0. One-time setup on the DGX

```bash
python -m pip install -r model/requirements.txt
# verify all 2.7M images exist (extract Places365 etc. first!)
python dataset/scripts/prepare_training_manifest.py
```

`prepare_training_manifest.py` caps Places365 at 400K, adds deepfake-dataset REAL
frames (warns loudly if they are missing — fix that before training), dedupes by
MD5, and writes `dataset/metadata/train_manifest.csv`. The notebooks pick it up
automatically; if it is absent they fall back to `clean_metadata.csv` and print a
warning.

**Check the first cell output of every notebook:** `SMOKE_TEST: False` and the
"Total samples" count must match the manifest (~2.7M). If the sample count is
much lower, image files are missing on disk — stop and fix before burning GPU
time.

## 1. Run order (all in `model/`)

| # | Notebook | Produces | Time driver |
|---|---|---|---|
| 1 | `train_mfft_base.ipynb` | Base checkpoint + figures/tables | 20 ep full data |
| 2 | `train_mfft_tiny.ipynb` | Tiny checkpoint + figures/tables | 20 ep |
| 3 | `train_mfft_large.ipynb` | Large checkpoint + figures/tables | 20 ep |
| 4 | `train_baselines.ipynb` | all 10 baselines | 10 x 20 ep |
| 5 | `train_ablation_study.ipynb` | 9 ablations + 4 upgrade rows | 13 x 20 ep |
| 6 | `paper_evals.ipynb` | CIs, McNemar, calibration, robustness, per-generator, LOGO manifests | eval only |
| 7 | `train_logo.ipynb` | full LOGO protocol: trains one model per held-out generator, evaluates each, writes `logo_summary.csv` | ~8 x 10 ep |

The **first** notebook creates `dataset/metadata/split_indices.json`; every later
notebook reuses it, so all models share the identical train/val/test split.
Do **not** delete that file between notebooks.

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

- results -> `paper/result/verify/...` (pilot results in `paper/result/test/` are untouched)
- checkpoints -> `model/checkpoints/verify/...`

Order: `mfft_base.ipynb` -> `ablation.ipynb` -> `baselines.ipynb` -> `paper_evals.ipynb`.
If all four finish without a red cell, the code is ready for the DGX.

## 4. What was fixed in the notebooks (summary)

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
