# Q1-Readiness Plan: Model Improvements + Required Experiments

> Written 2026-07-04, before the full-scale DGX training campaign.
> Companion to `paper/mfft_preliminary_draft.tex` (honest pilot draft) and
> `PUBLICATION_AUDIT_AND_ROADMAP.md` (project audit).

## STATUS UPDATE (2026-07-04, evening) — infrastructure implemented

| Item | Status | Where |
|---|---|---|
| 3-way split w/ saved indices | DONE | `model/src/dataset.py::create_split_dataloaders` (config: val 0.10 / test 0.10, `split_indices.json`) |
| JPEG + blur training augmentation (A5) | DONE | `model/src/dataset.py::ImageTransform` (JPEG q30-95 p=0.5, blur p=0.2) |
| Weighted sampler instead of undersampling | DONE | wired into `create_split_dataloaders`; `config.use_weighted_sampler=True` |
| LOGO cross-generator eval (B2) | DONE | `model/src/logo_eval.py` (manifest builder + per-generator evaluator) |
| Robustness suite (B3) | DONE | `model/src/robustness.py` (JPEG/downscale/blur grid -> CSV) |
| Bootstrap CI + McNemar (B5) | DONE | `model/src/stats.py` |
| Temperature scaling | DONE | `model/src/calibration.py` |
| Pretrained-extractor MFFT (A2/B7) | DONE | `model.py::PretrainedFrequencyFeatureExtractor`, `build_mfft(..., pretrained_extractors=True)` (MobileNetV3-S trunks, init-guarded; base = 4.63M params) |
| Per-category AUC bug | FIXED | patched in 8 train/test notebooks (scores, not binary preds) |
| Places365 cap + deepfake-real-frames check | SCRIPT READY | `dataset/scripts/prepare_training_manifest.py` (run before DGX; warns loudly if deepfake REAL frames are missing) |
| Stale metadata quarantine | DONE | `all.csv`, `dataset_metadata_real.*` moved to `dataset/_archive/metadata/` |
| Paper/code mismatches | FIXED | draft corrected: Base d=384/h=6, Large d=768/h=12/4-bands, per-RGB-channel FFT, param tables |
| Fusion-head fix (A-extra) | DONE | zero-pad hack replaced by learned `input_proj` for avg/max/skip-band/spatial configs; default variants byte-identical (checkpoint-safe) |
| NPR high-band residual (A3) | DONE | `build_mfft(..., use_npr=True)` concatenates the NPR residual (Tan et al., CVPR 2024) to the high-band input; works with pretrained extractors (stem inflated) |
| Soft band masks (A4) | DONE | `build_mfft(..., soft_masks=True)`, sigmoid transition tau=0.02*r_max; masks now cached per resolution (also a speedup for the default model) |
| FGA/static-fusion crash | FIXED | pre-existing bug: concat/avg/max + use_fga=True crashed; FGA now applies only to per-band token representations |

**DGX run matrix suggestion:** train the main table with defaults (checkpoint-compatible),
then add three ablation rows: `+pretrained_extractors`, `+use_npr`, `+soft_masks`
(and optionally all three combined) — each is one flag on `build_mfft`.

**Still on YOU before the DGX run:** (1) run `prepare_training_manifest.py` and
extract real frames from Celeb-DF/FF++/DFDC if it warns; (2) revoke the API keys
in `dataset/.env`; (3) optionally add ADM/VQDM/Wukong GenImage subsets for the
full LOGO protocol; (4) point `config.py::metadata_paths` at `train_manifest.csv`.

## The bar you must clear

Q1 forensics reviewers (IEEE TIFS, Pattern Recognition) in 2025-2026 expect:

1. **Cross-generator generalization** as the headline result, not in-distribution accuracy.
   The standard protocol is GenImage's: train on Stable Diffusion V1.4 subset only, test on
   all 8 generators (Midjourney, SD, ADM, GLIDE, Wukong, VQDM, BigGAN). Methods like
   NPR (CVPR 2024) and FreqNet (AAAI 2024) report ~86-93% mean cross-generator accuracy.
   You must meet or beat this, or show a different clear win (efficiency + interpretability).
2. **Robustness curves**: accuracy vs JPEG quality (30-95), downscale (0.25x-1x), blur.
3. **Beating strong baselines fairly**: same pretraining status, full convergence, CIs.
4. **In-the-wild sanity check**: Chameleon (ICLR 2025 "sanity check" paper) exposed that
   many detectors near-random on real-world images. Even a modest result here is credible.

## Part A — Model improvements (ordered by expected impact/effort)

### A1. Resolve the fusion paradox first (zero new code)
The pilot shows max-pool (90.6%) > cross-attn (90.0%) > full MFFT with FGA (88.6%).
Before adding anything, run the 9-config ablation at full scale. If CAF+FGA still loses
at 20 epochs, **ship the simpler model** — "we tested the fancy parts and kept what won"
is a *stronger* Q1 story than an unjustified complex architecture.

### A2. ImageNet-pretrain the band extractors (high impact, low effort)
Your pilot's clearest lesson: pretrained CNNs dominate (EfficientNet-B0 98.2%).
The per-band extractors are tiny custom CNNs trained from scratch — that's the gap.
Options:
- Swap each band extractor for a **pretrained EfficientNet-B0/MobileNetV3 stem**
  (first 2-3 stages, ~1-2M params each), keeping decomposition + fusion on top.
- Or pretrain your extractors on ImageNet once (cheap at their size) and fine-tune.
This gives an apples-to-apples "MFFT (pretrained)" row and likely closes the 4-7 pp gap.

### A3. High-pass residual input, not just band images (medium impact, low effort)
NPR and FreqNet succeed by *removing* semantic content so the model can't shortcut on it.
Feed each extractor `band_image` (already done) but for the high band consider the
NPR-style neighboring-pixel residual or SRM filter bank as extra channels.
Semantic shortcuts are the #1 cause of cross-generator failure.

### A4. Soft / learnable band boundaries (medium impact, medium effort)
Binary radial masks create ringing and hard information splits. Replace with:
- Gaussian-transition radial masks (fixed, sigma ~0.02 r_max), or
- Learnable cutoffs via sigmoid((r - c_b)/tau) with c_b trainable.
Differentiable, ~10 lines in `FrequencyDecomposition`, and gives a nice ablation table.

### A5. Augment in the frequency-robust direction (high impact for robustness section)
Add to training augs: random JPEG (quality 30-95), random downscale-upscale, Gaussian
blur. Wang et al. (CVPR 2020) showed this single change is the biggest generalization
lever. Without it your robustness curves will crater and reviewers will find it.

### A6. Contrastive band-consistency loss (optional, novelty booster)
Auxiliary loss: embeddings of the three bands of the *same real image* should be
consistent (natural spectral decay links bands); AI images break this. An InfoNCE-style
loss over band tokens is cheap and directly motivates the cross-attention design —
it would give CAF a reason to win over max-pool.

### A7. Keep color information in decomposition (small)
Current decomposition is grayscale-only. Diffusion fingerprints differ per channel;
per-channel FFT costs 3x the FFT (still <1ms) — worth an ablation row.

## Part B — Experiments the paper must contain

| # | Experiment | Protocol | Blocks submission? |
|---|---|---|---|
| B1 | Full training | 2.7M imgs, 20 ep, 384x384, AMP, all 13 models | YES |
| B2 | Leave-one-generator-out | GenImage protocol (train SD, test 8 gens) | YES |
| B3 | Robustness | JPEG 30-95, resize, blur curves | YES |
| B4 | Cross-dataset | CIFAKE + Chameleon in-the-wild | Strongly expected |
| B5 | Statistics | Bootstrap 95% CI (1000x), McNemar vs best baseline | Strongly expected |
| B6 | Full-scale ablation | Re-run all 9 configs at 20 epochs | YES (fusion paradox) |
| B7 | Matched-pretraining MFFT | A2 variant vs pretrained baselines | Expected |
| B8 | Per-generator table | Accuracy per generator family | Expected |

## Part C — Manuscript fixes before any submission

1. Remove unsourced numbers from `mfft_tifs.tex`: Celeb-DF 91% / FF++ 87% / DFDC 82%,
   FGA 45/35/20 weight split, LR-sensitivity claim (no experiment files exist for these).
2. Reconcile parameter counts (860K vs 1.62M Base; 3.1M vs 6.30M Large) everywhere.
3. Reconcile 224 vs 384 input and 1 vs 20 epochs between Methods and Results.
4. Delete Appendix H (journal recommendations) — not paper content.
5. Fix duplicated market-size sentence in the Introduction.
6. Abstract must claim only what tables show (current abstract claims CAF and FGA wins
   that Table III contradicts).

## Decision tree after full-scale results

- **MFFT (pretrained) beats all baselines + wins LOGO** → IEEE TIFS (Q1, IF ~6.8).
- **Competitive but not clearly SOTA, strong efficiency/interpretability story** →
  Pattern Recognition / EAAI if margins are solid; otherwise JVCIR, Signal Processing:
  Image Communication, FSI: Digital Investigation, Machine Vision & Applications
  (all Scopus Q1/Q2).
- **Want fast peer feedback first** → WACV / ICPR / BMVC conference, then extend to journal.
