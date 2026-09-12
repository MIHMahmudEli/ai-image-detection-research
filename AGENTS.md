# Session Memory — AI Image Detection Research

## Goal
Build and test a Multi-Frequency Fusion Transformer (MFFT) for AI-generated image detection with a complete training pipeline, baseline comparisons, and test-mode notebooks for validation.

## Current Status (2026-09-04)

### Dataset: VALIDATED ✅
- `train_manifest.csv`: **1,424,326 rows** — all images present on disk
- 20,000 sampled files: 0 missing, 0 zero-byte
- Label distribution: 47.4% real / 43.8% AI / 9% deepfake (balanced)
- `split_indices.json`: Does not exist yet — will be created by notebook 1 on DGX

### Architecture: COMPLETE ✅
- MFFT with 3 variants: tiny (372K), base (1.62M), large (6.30M params)
- 10 baselines implemented
- Full ablation support (9+ configurations)

### Training Pipeline: READY ✅
- 8 notebooks ready for DGX execution
- Auto-detects GPU/CPU (smoke mode on CPU)
- AMP, gradient accumulation, label smoothing, weighted sampler
- Shared split indices across all notebooks

### Deployment: READY ✅
- FastAPI server with tiered rate limiting
- Docker stack (api, web, nginx)
- HuggingFace Space deployment bundle

### Paper: WRITTEN ✅
- 1,225 lines, IEEE format, 47 references
- All accuracy numbers marked `[TBD]` — pending DGX training

## Completed Work

### Pipeline & Data
- Fixed `dataset.py`: pre-filter zero-byte/corrupted files at load, robust `__getitem__` with retry
- Fixed `train.py`: wrapped train/validate batch loops in try/except
- Wrote `dataset/validate_images.py` — standalone pre-flight scanner with --fix and --clean-csv
- Wrote `dataset/validate_for_dgx.py` — comprehensive DGX pre-flight check
- Wrote `dataset/kaggle_package.py` — Kaggle dataset packaging tool
- Deleted 6 zero-byte corrupted images from dataset

### Model Architecture (`model/src/model.py`)
- `MFFT.__init__` and `build_mfft` accept `ablation: Optional[dict]` with keys:
  - `spatial_only` — skip DCT, feed raw image
  - `skip_bands` — exclude specific bands by name
  - `fusion_mode` — `"attention"` / `"concat"` / `"avg"` / `"max"`
  - `use_fga` — enable/disable FrequencyGuidedAttention
- Three variants: tiny (128/4/3), base (384/6/3), large (768/12/4)
- Extensions: `PretrainedFrequencyFeatureExtractor`, `npr_residual`, soft band masks

### Baselines (`model/src/baselines.py`)
- `FreqDetect` — radial FFT magnitude profile + MLP
- `DeiT-Small` — 12-block transformer with class+distillation tokens
- Existing: SimpleCNN, LightViT, ResNet-18/50, EfficientNet-B0, ViT-B/16, Swin-T, CLIP

### Evaluation Suite
- `evaluate.py` — accuracy, precision, recall, F1, specificity, AUC-ROC, confusion matrix
- `robustness.py` — JPEG/downscale/blur degradation curves
- `calibration.py` — temperature scaling, ECE, Brier score
- `stats.py` — bootstrap confidence intervals, McNemar test
- `logo_eval.py` — leave-one-generator-out evaluation
- `visualize.py` — 15 publication-ready figures (988 lines)

### Notebooks (All use 20 epochs)
| Notebook | Purpose | Status |
|---|---|---|
| `train_mfft_tiny.ipynb` | Train MFFT-Tiny | Ready for DGX |
| `train_mfft_base.ipynb` | Train MFFT-Base (creates shared split) | Ready for DGX |
| `train_mfft_large.ipynb` | Train MFFT-Large | Ready for DGX |
| `train_baselines.ipynb` | Train all 10 baselines | Ready for DGX |
| `train_ablation_study.ipynb` | Run 9+ ablation configs | Ready for DGX |
| `paper_evals.ipynb` | CIs, McNemar, calibration, robustness | Ready for DGX |
| `train_logo.ipynb` | LOGO cross-generator eval | Ready for DGX |
| `generate_paper_outputs.ipynb` | All 15 figures + 9 tables | Ready for DGX |

### Test Notebooks (`model/test_model/`)
- Copies of originals (1 epoch, 5K subset, output to `paper/result/test/{name}_model/`)
- Includes: mfft_tiny, mfft_base, mfft_large, mfft_smoke, baselines, ablation

### Verification Notebooks (`model/test_model_verify/`)
- Local CPU verification (auto smoke mode)
- Output to quarantined `paper/result/verify/` and `model/checkpoints/verify/`
- All completed successfully as of 2026-07-05

### API & Deployment
- FastAPI server with tiered rate limiting (free/pro/enterprise)
- Batch prediction, heatmap generation, frequency band analysis
- `Dockerfile.api`, `Dockerfile.train`, `Dockerfile.web`
- `compose.yaml` — 3 services (api:8000, web:3000, nginx:80/443)
- HuggingFace Space deployment (`hf_space/`, `deploy_hf_space.py`)

### Kaggle Support
- `kaggle_mfft_train.py` — self-contained Kaggle training script
- `kaggle_train_full.py` — full MFFT architecture for Kaggle
- `dataset/kaggle_package.py` — packages 50K subset for Kaggle upload

### Paper (`paper/mfft_manuscript.tex`)
- 1,225 lines, IEEE format, 47 references
- Appendices A-E (architecture specs, dataset composition, hyperparameters, baselines, explainability)
- All accuracy numbers marked `[TBD]` — pending DGX training

### Scripts
- `model/generate_paper_outputs.py` — post-training pipeline (deprecated, use notebook version)
- `dataset/prepare_training_manifest.py` — builds confound-fixed manifest
- `dataset/prepare_test_manifest.py` — builds 5K test manifest
- `dataset/validate_for_dgx.py` — DGX pre-flight validation

## Next Steps

### DGX Training (in order)
1. Transfer repo + dataset (~166GB) to DGX
2. `pip install -r model/requirements.txt && pip install open_clip_torch scipy`
3. `python dataset/scripts/prepare_training_manifest.py`
4. `python dataset/validate_for_dgx.py` — must show ALL CHECKS PASSED
5. Run 8 notebooks in order (see `DGX_RUN_GUIDE.md`)
6. Retrieve checkpoints + paper/results from DGX

### Post-Training
1. Fill `[TBD]` values in `paper/manuscript.md`
2. Compile LaTeX manuscript
3. Submit to target journal (IEEE TIFS or Pattern Recognition)

## Manuscript
- Extended to ~974 lines (~15 pages equivalent) with full appendices A-E:
  - Appendix A: Complete architecture specs (6 sub-sections with per-component parameter counts)
  - Appendix B: Full dataset composition (sources, generator families, deepfake datasets, image categories)
  - Appendix C: Training hyperparameter details (optimizer, scheduler, augmentation pipeline)
  - Appendix D: Additional baseline details (SimpleCNN, FreqDetect architectures)
  - Appendix E: MFFT explainability features (heatmap generation, confidence calibration)
- New sections added: 2.4 (Comparison with Contemporary Frequency-Based Methods), 6.3 (Model Variant Analysis), 6.5 (Explainability Analysis), 6.7-6.9 (Practical Implications, Ethical Considerations, Broader Societal Impact)
- References expanded from 37 to 47

## Journal Recommendations (Scopus-Indexed)
- **IEEE Trans. Information Forensics and Security (TIFS)** — Q1, perfect fit (image forensics, frequency analysis)
- **Pattern Recognition** — Q1, computer vision + pattern analysis
- **Neurocomputing** — Q1, deep learning applications
- **Engineering Applications of AI** — Q1, applied systems
- **Expert Systems with Applications** — Q1, applied detection
- **Computer Vision and Image Understanding** — Q1
- **Signal Processing: Image Communication** — Q2
- **IEEE Access** — Q1, open access
- **Scientific Reports** — Q2, open access
- **PLOS One** — Q2, open access

## Key Decisions
- 20 epochs instead of 50 (2.7M dataset converges fast)
- try/except at batch level rather than pre-scanning all files
- Fusion modes: attention, concat, avg, max
- Band names: ["low", "mid", "high"]
- WeightedRandomSampler instead of undersampling (keeps all data)
- Shared split indices across all notebooks (reproducibility)
- Places365 capped at 400K (reduces content shortcut)
- Deepfake REAL frames added (reduces face⇒fake shortcut)
