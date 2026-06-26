# Project Status Report to Supervisor

## Current State of AI-Image-Detection-Research

---

### 1. Audit Summary

We completed a full audit of the project. The MFFT architecture is novel and worth publishing, but the current manuscript contains **fabricated results** and cannot be submitted in its current state.

| Metric | Manuscript Claims | Actual | Gap |
|---|---|---|---|
| Accuracy | 97.2% | 67.88% | −29.3 pp |
| Recall (AI detection) | 97.6% | 43.96% | −53.6 pp |
| AUC-ROC | 99.1% | 0.7389 | −25.2 pp |
| Model Parameters | 12.5M | 0.87M | 14× smaller |
| Training | RTX 4090, 50 epochs | CPU, 1 epoch | Not comparable |

**Key issues found:**
- All baseline comparisons (EfficientNet, ResNet, ViT, Swin, CLIP) were **never run** — no code exists for them
- All ablation studies were **fabricated** — the ablation scripts crash on execution
- The "AI-altered" dataset (28,616 images) is **PIL filter operations**, not real deepfakes
- No DALL-E 3 or Midjourney images exist in the dataset despite being claimed
- The API serves predictions from **random weights** (wrong checkpoint path)
- Several code bugs prevent reproduction of results

---

### 2. Work Completed This Week

**a) Metadata regeneration**
- Scanned all 57,097 images to record real dimensions, hashes, and file sizes
- Removed 562 exact duplicates → 56,535 unique images

**b) Data collection progress**
- Added ~165 new CivitAI images (3,480 → 3,645)
- Added 32 new Pollinations images (8 → 40)
- Discovered Pollinations rate-limits to 1 request/second — too slow for bulk collection

**c) Created publication roadmap**
- `PUBLICATION_AUDIT_AND_ROADMAP.md` with full 7-phase plan (4-6 months)

---

### 3. Current Dataset Status

| Category | Count | Problem |
|---|---|---|
| Real | 24,439 | ✅ Acceptable |
| AI-Generated | ~3,685 | ❌ Needs 6K+ more from diverse generators |
| AI-Altered | 28,616 | ❌ **Must be replaced** — all are PIL filters, not real deepfakes |

---

### 4. Recommendation: Pivot to Public Datasets

Scraping free APIs is too slow. We should download established public datasets instead:

| Dataset | Content | Size | Use |
|---|---|---|---|
| **FaceForensics++** | Real face-swap deepfakes | ~1 GB | Replace AI-altered images |
| **Celeb-DF** | High-quality deepfakes | ~2 GB | Replace AI-altered images |
| **DFDC** | 120K deepfakes | ~5 GB | Replace AI-altered images |
| **GenImage** | SD, Midjourney, DALL-E, Flux images | ~50 GB | Bulk up AI-generated class |

**Estimated time with public datasets:** 2-3 days to download vs. weeks of scraping.

---

### 5. Immediate Needs

- [ ] [Supervisor approval] to use FaceForensics++ — requires submitting a download request form. They ask for a project description (see below).
- [ ] GPU access (RTX 4090 or similar) for training — current machine is CPU-only
- [ ] Decision on target venue (suggested: WACV/BMVC conference → IEEE TIFS journal)

**Project description for FF++ download request:**
> "Academic research on a Multi-Frequency Fusion Transformer for AI-generated image detection. We need FF++ as a benchmark dataset to evaluate our frequency-domain deepfake detection method against real face-swap manipulations and compare with state-of-the-art baselines. Non-commercial use only."

---

### 6. Next Steps (if approved)

1. Download FaceForensics++ and integrate into dataset
2. Retrain MFFT on GPU (50 epochs, proper hyperparameters)
3. Implement real baselines (EfficientNet, ResNet, ViT using torchvision)
4. Run actual ablation studies
5. Rewrite manuscript with honest results
6. Target completion: ~4-6 months

---

*Full audit details: `PUBLICATION_AUDIT_AND_ROADMAP.md`*
