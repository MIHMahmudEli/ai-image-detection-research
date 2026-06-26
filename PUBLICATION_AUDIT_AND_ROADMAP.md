# Publication Audit & Roadmap

> Generated: 2026-06-26
> Based on deep analysis of: code, dataset, paper manuscript, figures, tables, API, web, infrastructure

---

## Executive Summary

**The MFFT architecture (frequency decomposition + cross-attention fusion) is a valid and publishable research direction. However, the current manuscript contains fabricated results and the project requires 4-6 months of systematic rebuilding before any submission.**

| Area | Status | Verdict |
|---|---|---|
| Architecture novelty | ✅ Good | MFFT is genuinely novel |
| Code quality | ⚠️ Mixed | Core is clean, but critical bugs exist |
| Dataset | ❌ Invalid | PIL-based "deepfakes", not real AI manipulation |
| Results (claimed) | ❌ Fabricated | 97.2% vs actual 67.88% |
| Results (actual) | ❌ Below baseline | 67.88% accuracy, 43.96% recall |
| Baselines | ❌ Missing | None of 8 claimed baselines exist |
| API/Production | ❌ Broken | Serves random weights |
| Manuscript | ❌ Fraudulent | Every quantitative claim is unsupported |

---

## 1. Critical Discrepancies: Claimed vs. Actual

### 1.1 Performance Metrics

| Metric | Manuscript Claims | Actual (tables/) | Gap |
|---|---|---|---|
| Accuracy | **97.2%** | **67.88%** | −29.3 pp |
| Precision | 96.9% | 84.28% | −12.6 pp |
| Recall | 97.6% | **43.96%** | −53.6 pp |
| F1 Score | 97.2% | 57.78% | −39.4 pp |
| AUC-ROC | **99.1%** | **0.7389** | −25.2 pp |

### 1.2 Confusion Matrix (Actual)

```
                Pred Real    Pred AI
Actual Real      2295 TN       205 FP
Actual AI        1401 FN      1099 TP
```

The model misses **56% of AI images** (1401/2500 false negatives). It is heavily biased toward predicting "Real".

### 1.3 Per-Category Accuracy (Actual)

| Category | Count | Accuracy |
|---|---|---|
| Real | 2,500 | 91.80% |
| AI Generated | 274 | 55.84% |
| AI Altered | 2,226 | 42.50% |

### 1.4 Model Specs

| Property | Manuscript Claims | Actual |
|---|---|---|
| Parameters | **12.5M** | **866,498** |
| Input Resolution | 384×384 | 224×224 |
| Training Device | RTX 4090 | **CPU** |
| Training Epochs | 50 | **1** (of 50 planned) |
| Batch Size | 32 | 8 |
| Mixed Precision | Yes | No |

### 1.5 Dataset Composition

| Category | Manuscript Claims | Actual |
|---|---|---|
| Real | 15,000 | 25,001 |
| AI-Generated | 10,000 | **3,481** |
| AI-Altered | 25,000 | 28,616 |
| **Total** | **50,000** | **57,098** |

---

## 2. What Is Fabricated in the Manuscript

### 2.1 Baseline Comparisons (Manuscript Table 1)

The manuscript claims comparisons against **8 state-of-the-art methods**:

| Claimed Baseline | Exists in Codebase? |
|---|---|
| EfficientNet-B4 | ❌ **No** |
| ResNet-152 | ❌ **No** |
| ViT-B/16 | ❌ **No** |
| DeiT-S | ❌ **No** |
| Swin-T | ❌ **No** |
| CLIP (zero-shot) | ❌ **No** |
| CLIP + Linear Probe | ❌ **No** |
| FreqDetect | ❌ **No** |

Only `SimpleCNN` and `LightViT` exist in `model/src/baselines.py` — neither appears in the manuscript.

**Source:** `model/src/baselines.py`

### 2.2 Ablation Studies (Manuscript Tables 4-5)

All ablation results are fabricated:
- No single-band variants were actually trained
- No fusion strategy comparison was performed
- The "4.7 percentage point improvement from multi-frequency fusion" does not exist
- The "2-band / 3-band / 4-band" parameter count comparison is invented

**Evidence:** `train_baselines.py` and `train_variants.py` both crash on execution — they expect 6 return values from `create_dataloaders()` which returns only 2.

### 2.3 Per-Generator Analysis (Manuscript Table 2)

Claims include:
- DALL-E 3: 99.4% accuracy
- Midjourney v6: 97.8% accuracy

**Actual dataset contains zero images from DALL-E 3 or Midjourney.** Only CivitAI images were collected. The 3,481 AI-generated images in the dataset come exclusively from CivitAI (SD1.5/SDXL/LoRA).

**Source:** `dataset/collectors/` scripts

### 2.4 Error Analysis (Manuscript Table 6)

Claims 1.2% FP rate and 1.6% FN rate. Actual rates:
- FP rate: 205/2500 = **8.2%**
- FN rate: 1401/2500 = **56.0%**

### 2.5 Computational Efficiency (Manuscript Table 7)

Claims 12.5M parameters at 141 img/s on RTX 4090. Reality:
- 0.87M parameters (14× smaller)
- Never benchmarked on a GPU
- Inference speed numbers are entirely invented

---

## 3. Dataset Issues

### 3.1 "AI-Altered" Images Are Not Real Deepfakes

The 28,616 AI-altered images are created with **basic PIL operations**, not AI models:

| Method | Count | Description |
|---|---|---|
| `FILL_*` | 16,051 | PIL filter effects (MedianBlur, EdgeEnhance, Posterize, etc.) |
| `FS_*` | 5,065 | Center-crop face + paste with Gaussian blur |
| `FACESWAP_*` | 3,000 | Same simplistic face-swap operation |
| `INPAINT_*` | 2,400 | scipy `uniform_filter` smoothing over random rectangles |
| `OUTPAINT_*` | 1,200 | PIL edge extension |
| `STYLE_*` | 900 | PIL ImageFilter operations |

**Impact:** A model trained on this dataset learns to detect PIL filter artifacts, not real AI manipulation. Results have zero scientific validity for the claimed task.

### 3.2 Generator Diversity is Insufficient

| Generator | Claimed Count | Actual Count |
|---|---|---|
| CivitAI (SD1.5/SDXL/LoRA) | 4,000 | ~3,481 |
| DiffusionDB | 4,000 | **0** |
| Pollinations.ai | 2,000 | **0** |
| DALL-E 3 | Not in plan | **0** |
| Midjourney | Not in plan | **0** |

### 3.3 Metadata is Broken

- `dataset/metadata/all.csv` has `width=0` and `height=0` for all 57K images
- No train/val/test split files exist (`splits/` directory is absent)
- No `alteration_method` column for altered images
- No prompt or model metadata for AI-generated images
- 6,127 augmented real images are near-duplicates of originals — potential train/test leakage

### 3.4 API Keys Exposed

The file `dataset/.env` contains live API keys for Unsplash, Pexels, Pixabay, HuggingFace, CivitAI, Ultralytics, and Pollinations. These should be revoked immediately.

---

## 4. Code Bugs (Must Fix)

### 4.1 Critical (Will Crash or Give Wrong Results)

| # | File | Line | Issue |
|---|---|---|---|
| 1 | `train_baselines.py` | 23 | `create_dataloaders()` returns 2 values, code expects 6. **Will crash.** |
| 2 | `train_variants.py` | 24 | Same bug as above. **Will crash.** |
| 3 | `api/main.py` | 42 | Hardcoded path `model/checkpoints/best.pt` does not exist. API serves **random weights**. |
| 4 | `test_cells_10_12.py` | 197 | AUC computed on **binary predictions** instead of probability scores. Produces incorrect AUC. |
| 5 | `dataset.py` | 222-230 | Undersampling is silently broken — `_undersample()` runs on empty samples list. |

### 4.2 High Severity

| # | File | Line | Issue |
|---|---|---|---|
| 6 | `api/main.py` | 246-251 | Frequency band contributions are **hardcoded** `{low: 0.33, mid: 0.35, high: 0.32}` |
| 7 | `model.py` | 8-13 | Docstring says DCT but implementation uses **FFT** (`torch.fft.fft2`) |
| 8 | `model.py` | 130-134 | `FrequencyGuidedAttention` is **defined but never called** in `forward()` — dead code |
| 9 | `config.py` | - | `test_split` is defined but never used anywhere |
| 10 | `model_server.py` | 29 | Falls back to random weights silently if checkpoint not found |

### 4.3 Medium Severity

| # | File | Line | Issue |
|---|---|---|---|
| 11 | `visualize.py` | ~478 | `fig13_*` is overwritten by two different plots (error_analysis and calibration_curve) |
| 12 | `train.py` | - | `WandbConfig` is defined but wandb logging is never implemented |
| 13 | `train.py` | 176-201 | `metrics.json` is overwritten each epoch instead of appending |
| 14 | `.gitignore` | - | Excludes `dataset/metadata/*.csv` but `all.csv` is tracked in git |

---

## 5. What's Actually Good (Don't Throw Away)

### 5.1 Architecture

The MFFT design is genuinely novel:
- Frequency decomposition into 3 bands via FFT
- Per-band CNN feature extractors with depthwise separable convolutions
- Cross-attention fusion across bands
- Frequency-guided weighting mechanism
- Clean modular design in `model.py`

### 5.2 Visualization Pipeline

`visualize.py` (984 lines) produces 15 publication-quality figures with:
- IEEE-style formatting (serif fonts, proper color palettes, 300 DPI)
- Training history, confusion matrix, ROC/PR curves
- t-SNE embeddings, anomaly heatmaps, frequency response analysis
- Calibration curves, architecture diagrams

### 5.3 Infrastructure

- Professional Docker Compose setup with GPU support
- Clean Next.js frontend with drag-and-drop, responsive design
- FastAPI with proper error handling, CORS, rate limiting
- Nginx reverse proxy configuration

### 5.4 Paper Structure

The manuscript has:
- Well-structured narrative with clear research questions (RQ1-RQ5)
- Proper mathematical notation for all equations
- 37 references from credible venues (CVPR, NeurIPS, ICML, IEEE TIFS)
- Standard paper structure with all required sections

---

## 6. Roadmap: 4-6 Month Plan

### Phase 1: Fix the Foundation (Month 1)

| Task | Details | Priority |
|---|---|---|
| Rebuild dataset | Replace PIL alterations with real deepfakes (SimSwap, SD inpainting, etc.) | 🔴 Critical |
| Add diverse generators | Collect from DALL-E 3, Midjourney v6, SDXL, Flux, Imagen | 🔴 Critical |
| Fix metadata | Record width/height, source, prompt, alteration method for every image | 🔴 Critical |
| Deduplicate | Remove near-duplicate augmentations, hash-based deduplication | 🔴 Critical |
| Create proper splits | Stratified 70/15/15 with fixed seed, save split indices | 🔴 Critical |
| Revoke exposed API keys | Rotate all keys in `dataset/.env` | 🔴 Security |

### Phase 2: Train a Real Model (Month 1-2)

| Task | Details | Priority |
|---|---|---|
| Fix all critical code bugs | train_baselines.py, train_variants.py, dataset.py, etc. | 🔴 Critical |
| Train MFFT on GPU | RTX 4090, batch size 32, 384×384, mixed precision, 50 epochs | 🔴 Critical |
| Implement real baselines | EfficientNet-B4, ResNet-152, ViT-B/16, Swin-T (use torchvision models) | 🔴 Critical |
| Implement CLIP baselines | Zero-shot + linear probe using `open_clip` | 🟡 High |
| Implement FreqDetect baseline | Reproduce from Frank et al. 2020 | 🟡 High |
| Train all baselines | Same data, same protocol, same evaluation | 🔴 Critical |

### Phase 3: Run Actual Experiments (Month 2-3)

| Task | Details | Priority |
|---|---|---|
| Ablation: spatial-only baseline | Remove frequency decomposition | 🔴 Critical |
| Ablation: single-band variants | Train low-only, mid-only, high-only | 🔴 Critical |
| Ablation: fusion strategy | Compare cross-attention vs concat, avg, max pooling | 🟡 High |
| Ablation: FGA removal | Train without Frequency-Guided Attention | 🟡 High |
| Ablation: band count | Test 2, 3, 4 bands | 🟡 High |
| Per-generator analysis | Evaluate on each generator separately | 🟡 High |
| Per-category analysis | Evaluate on portraits, landscapes, objects, etc. | 🟡 High |
| Cross-dataset evaluation | Test on CIFAKE, GenImage, DiffusionDB | 🔴 Critical |
| Robustness analysis | JPEG compression, resizing, blur, noise, color jitter | 🟡 High |
| Compute confidence intervals | Bootstrapping with 1000 iterations | 🟡 High |

### Phase 4: Achieve Competitive Performance (Month 2-3)

| Task | Target |
|---|---|
| MFFT accuracy target | ≥95% on held-out test set |
| Baseline gap target | ≥3 pp over second-best method |
| Cross-dataset target | ≥85% on unseen generators |
| Robustness target | ≤5 pp drop under moderate JPEG compression |

If performance is below these targets, consider:
- Increasing model capacity (scale to 5-12M parameters)
- Adding pretraining on large-scale data (ImageNet, LAION)
- Training with stronger augmentations (RandAugment, MixUp)
- Ensemble of multiple frequency decomposition strategies

### Phase 5: Rewrite the Manuscript (Month 3-4)

| Task | Details | Priority |
|---|---|---|
| Replace all fabricated numbers | Every quantitative claim must come from actual experiments | 🔴 Critical |
| Update dataset description | Match actual dataset composition | 🔴 Critical |
| Rewrite Results section | Honest reporting of actual performance | 🔴 Critical |
| Update all tables | Replace Table 1-7 with real data | 🔴 Critical |
| Regenerate all figures | From actual evaluation outputs | 🔴 Critical |
| Complete appendices | Architecture details, hyperparameter search, additional results | 🟡 High |
| Write limitations section | Actual limitations, not invented ones | 🟡 High |
| Add statistical rigor | Confidence intervals, significance tests | 🟡 High |
| Add ethics statement | Dataset biases, intended use, failure modes | 🟡 High |
| Add code/data availability | Repository link, dataset hosting, model weights | 🟡 Medium |

### Phase 6: Fix Production (Month 3-4)

| Task | Details | Priority |
|---|---|---|
| Fix API checkpoint path | Point to real checkpoint or rename to `best.pt` | 🔴 Critical |
| Implement real frequency computation | Replace hardcoded values with actual model outputs | 🔴 Critical |
| Add model versioning | Track which checkpoint version is deployed | 🟡 High |
| Add proper logging | Structured logging with request/response tracking | 🟡 High |
| Add authentication | Real API key validation, not hardcoded "free" tier | 🟡 Medium |
| Add HTTPS | TLS termination at nginx | 🟡 Medium |

### Phase 7: Submit (Month 4-6)

| Task | Details |
|---|---|
| Target venue | IEEE TIFS, Pattern Recognition, CVIU (Q1 journals) |
| Secondary venue | WACV, BMVC, ICPR (top-tier conferences) |
| Revision buffer | 2 months for reviewer feedback and experiments |

---

## 7. Target Metrics for Q1 Publication

| Metric | Minimum Acceptable | Competitive (Q1-Ready) |
|---|---|---|
| Overall Accuracy | ≥90% | ≥95% |
| AI Recall | ≥85% | ≥92% |
| AUC-ROC | ≥0.95 | ≥0.98 |
| Cross-dataset (unseen gen.) | ≥80% | ≥88% |
| Baselines implemented | ≥4 SOTA methods | ≥6 SOTA methods |
| Ablations completed | 4 of 5 planned | All 5 planned |
| Dataset size | ≥50K | ≥100K |
| Generator families | ≥5 | ≥8 |

---

## 8. Venue Suggestions

| Venue | Type | Difficulty | Match |
|---|---|---|---|
| IEEE TIFS | Q1 Journal | Very High | ✅ Frequency forensics specialty |
| Pattern Recognition | Q1 Journal | High | ✅ General computer vision |
| CVIU | Q1 Journal | High | ✅ Image understanding |
| WACV | Top Conf | Medium | ⚠️ Lower bar, faster turnaround |
| BMVC | Top Conf | Medium | ⚠️ Good for initial submission |
| ICPR | Top Conf | Medium | ⚠️ Deadlines may align |
| IEEE Access | Open Access | Low | ❌ Not Q1, fallback only |

**Recommendation:** Target WACV or BMVC first to get peer review feedback, then submit an extended version to IEEE TIFS or Pattern Recognition.

---

## 9. Quick Reference: Key Files and Lines

| File | Line(s) | Issue |
|---|---|---|
| `paper/manuscript.md` | 17, 274-367 | Fabricated results throughout |
| `paper/tables/table3_evaluation_metrics.csv` | 1-7 | Actual metrics: 67.88% acc, 0.7389 AUC |
| `model/src/model.py` | 8-13 | Docstring says DCT, uses FFT |
| `model/src/model.py` | 116-134 | FrequencyGuidedAttention defined but never called |
| `model/src/test_cells_10_12.py` | 86-87 | `strict=False` masks architecture mismatches |
| `model/src/test_cells_10_12.py` | 197 | AUC computed on binary predictions (wrong) |
| `model/src/dataset.py` | 222-230 | Undersampling silently broken |
| `model/src/train_baselines.py` | 23 | Crashes: expects 6 returns, gets 2 |
| `model/src/train_variants.py` | 24 | Same crash bug |
| `model/src/baselines.py` | 1-100 | Only SimpleCNN + LightViT exist (no SOTA methods) |
| `api/main.py` | 42 | Checkpoint path `best.pt` doesn't exist |
| `api/main.py` | 246-251 | Frequency contributions are hardcoded |
| `model/train_mfft.ipynb` | Cell 1 | Training on CPU: `Torch: 2.12.1+cpu, CUDA: False` |
| `dataset/collectors/fill_to_50k.py` | 1-200 | 16K PIL filter images as "deepfakes" |

---

## 10. Immediate Action Items (Next 48 Hours)

- [ ] **Revoke API keys** in `dataset/.env` (all 7 services)
- [ ] **Delete `dataset/.env` from git history** if accidentally committed
- [ ] **Remove fabricated numbers from manuscript** or clearly mark as placeholder
- [ ] **Do not submit the current manuscript** anywhere
- [ ] **Fix the API checkpoint path** so it doesn't serve random predictions
- [ ] **Prioritize Phase 1** (dataset rebuild) before any other work
