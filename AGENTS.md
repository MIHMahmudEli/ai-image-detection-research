# Session Memory — AI Image Detection Research

## Goal
Build and test a Multi-Frequency Fusion Transformer (MFFT) for AI-generated image detection with a complete training pipeline, baseline comparisons, and test-mode notebooks for validation.

## Completed Work

### Pipeline & Data
- Fixed `dataset.py`: pre-filter zero-byte/corrupted files at load, robust `__getitem__` with retry
- Fixed `train.py`: wrapped train/validate batch loops in try/except
- Wrote `dataset/validate_images.py` — standalone pre-flight scanner with --fix and --clean-csv
- Deleted 6 zero-byte corrupted images from dataset

### Model Architecture (`model/src/model.py`)
- `MFFT.__init__` and `build_mfft` accept `ablation: Optional[dict]` with keys:
  - `spatial_only` — skip DCT, feed raw image
  - `skip_bands` — exclude specific bands by name
  - `fusion_mode` — `"attention"` / `"concat"` / `"avg"` / `"max"`
  - `use_fga` — enable/disable FrequencyGuidedAttention
- Three variants: tiny (128/4/3), base (256/8/3), large (512/12/4)

### Baselines (`model/src/baselines.py`)
- Added `FreqDetect` — radial FFT magnitude profile + MLP
- Added `DeiT-Small` — 12-block transformer with class+distillation tokens
- Existing: SimpleCNN, LightViT, ResNet-18/50, EfficientNet-B0, ViT-B/16, Swin-T, CLIP

### Notebooks (All use 20 epochs)
| Notebook | Purpose |
|---|---|
| `train_mfft_tiny/base/large.ipynb` | Train individual MFFT variants |
| `train_baselines.ipynb` | Train all 10 baselines |
| `train_ablation_study.ipynb` | Run 9 ablation configs (Tables 4 & 5) |

### Test Notebooks (`model/test_model/`)
- Copies of originals (1 epoch, 5K subset, output to `paper/result/test/{name}_model/`)
- Includes: mfft_tiny, mfft_base, mfft_large, mfft_smoke, baselines, ablation

### API & Deployment
- FastAPI server with tiered rate limiting (free/pro/enterprise)
- Batch prediction, heatmap generation, frequency band analysis
- `Dockerfile` for API deployment

### Paper (`paper/manuscript.md`)
- Updated from 50K → 2.7M images
- Added FreqDetect and DeiT-S to Table 4.2 baselines
- All accuracy numbers marked `[TBD]` — pending training
- Tables 1-9: placeholders, no real results yet

### Scripts
- `model/generate_paper_outputs.py` — post-training pipeline: evaluate model on test set, generate all 15 figures + tables

### Removed
- `model/train_mfft.ipynb` — redundant (same as train_mfft_base.ipynb)
- `model/test_model/mfft.ipynb` — test counterpart

## Next Steps (Post-Training)
1. Run `train_mfft_tiny.ipynb`, `train_mfft_base.ipynb`, `train_mfft_large.ipynb`
2. Run `train_baselines.ipynb` (all 10 models)
3. Run `train_ablation_study.ipynb` (9 configs)
4. Fill `[TBD]` values in `paper/manuscript.md`
5. Run `generate_paper_outputs.py --variant base --checkpoint ...` for each variant
6. Submit GPU lab access application (GPU_LAB_ACCESS_APPLICATION.md)

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
