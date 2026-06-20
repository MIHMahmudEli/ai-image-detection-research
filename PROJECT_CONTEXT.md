# AI Image Detection Research — Project Context

## Goal
Build a **Multi-Frequency Fusion Transformer (MFFT)** for AI-generated image detection with a commercial detection website, targeting Q1 journal publication.

---

## Dataset Status (as of 2026-06-20)

| Category | Count | Notes |
|---|---|---|
| Real photos | 17,903 | From Unsplash/Pexels/Pixabay APIs |
| AI-generated | 3,481 | From Pollinations.ai, CivitAI, DiffusionDB |
| AI-altered (deepfakes) | 28,616 | Face swap, inpainting, outpainting, style transfer |
| **TOTAL** | **50,000** | All consolidated in `ai_dataset_50k/` |

**Images are excluded from git** (17 GB) via `.gitignore`. Only code + metadata are tracked.

### Dataset Location
```
dataset/
├── images/
│   ├── real/             17,903 files
│   ├── ai_generated/      3,481 files
│   └── ai_altered/       28,616 files
├── metadata/
│   └── all.csv
├── collectors/            Python dataset collection scripts
├── docs/                  Documentation
└── .env                   API keys (gitignored)
```

---

## MFFT Architecture

```
INPUT RGB Image
    │
    ▼
┌─────────────────────────────────────────────┐
│ 1. FREQUENCY DECOMPOSITION (DCT)            │
│    FFT2 → fftshift → radial band-pass mask  │
│    ┌──────────┬──────────┬──────────────┐   │
│    │ LOW      │ MID      │ HIGH         │   │
│    │ 0-15%    │ 15-45%   │ 45-100%      │   │
│    │ (global) │(textures)│(noise,edges) │   │
│    └────┬─────┴────┬─────┴──────┬───────┘   │
└─────────┼──────────┼────────────┼───────────┘
          │          │            │
    ┌─────▼──┐ ┌─────▼──┐ ┌──────▼─────┐
    │ CNN    │ │ CNN    │ │ CNN        │
    │ Feat   │ │ Feat   │ │ Feat       │
    │ Extr.  │ │ Extr.  │ │ Extr.      │
    │ 256-dim│ │ 256-dim│ │ 256-dim    │
    └────┬───┘ └────┬───┘ └─────┬──────┘
         │          │           │
         └──────────┼───────────┘
                    ▼
    ┌───────────────────────────────┐
    │ 2. CROSS-ATTENTION FUSION    │
    │ Multi-head self-attention     │
    │ across 3 band tokens (8 heads)│
    │ Output: [B, 3, 256]          │
    └──────────────┬────────────────┘
                   ▼
    ┌───────────────────────────────┐
    │ 3. FREQUENCY-GUIDED ATTENTION │
    │ Band magnitude → softmax      │
    │ → weight & gate features      │
    └──────────────┬────────────────┘
                   ▼
    ┌───────────────────────────────┐
    │ 4. CLASSIFIER (MLP)          │
    │ [B, 768] → LayerNorm          │
    │ → Linear(768→256) → GELU      │
    │ → Dropout(0.2)                │
    │ → Linear(256→128) → GELU      │
    │ → Dropout(0.1)                │
    │ → Linear(128→2)               │
    └──────────────┬────────────────┘
                   ▼
          [B, 2] logits
          [real, ai_generated]
                   ▼
              Softmax
          real_prob / ai_prob
```

### Key Components

| Component | File | Purpose |
|---|---|---|
| FrequencyDecomposition | `model/src/model.py:8` | DCT band decomposition |
| FrequencyFeatureExtractor | `model/src/model.py:46` | Per-band CNN with depthwise conv |
| CrossAttentionFusion | `model/src/model.py:85` | Multi-head self-attention across bands |
| FrequencyGuidedAttention | `model/src/model.py:116` | Spatial gating from frequency magnitude |
| MFFT | `model/src/model.py:137` | Main model class |
| MFFTWithExplainability | `model/src/model.py:230` | Adds preprocess/predict/heatmaps |

### How Detection Works

AI generators leave **frequency-domain fingerprints**:
- **GANs**: Checkerboard artifacts (transposed convs) → HIGH band spikes
- **Diffusion**: Over-smoothing → missing MID band texture energy
- **All AI**: No camera sensor noise → flat HIGH band roll-off
- **Real photos**: Natural noise correlation across all bands

The model detects these by analyzing how the 3 frequency bands correlate.

---

## Project Map

```
ai-image-detection-research/
├── model/                         # MFFT training code
│   ├── src/
│   │   ├── model.py              # Core MFFT architecture
│   │   ├── dataset.py            # Dataset loader (reads from dataset/images/)
│   │   ├── train.py              # Training loop (mixed precision, AdamW, cosine LR)
│   │   ├── evaluate.py           # Evaluation metrics
│   │   └── config.py             # Hyperparameters
│   └── requirements.txt
├── api/                           # FastAPI inference server
│   ├── main.py                   # 3-tier rate limiting (Free/Pro/Enterprise)
│   ├── model_server.py           # Model warmup & GPU inference
│   ├── schemas.py                # Pydantic schemas
│   └── requirements.txt
├── web/                           # Next.js commercial website
│   ├── app/
│   │   ├── page.tsx              # Landing page + upload + results
│   │   ├── layout.tsx
│   │   └── docs/page.tsx         # API documentation
│   ├── package.json
│   └── next.config.js
├── paper/
│   └── manuscript.md             # Full Q1 journal paper (526 lines, 37+ refs)
├── dataset/                       # Consolidated 50K dataset
│   ├── images/                    # 50K images (gitignored)
│   │   ├── real/                  #   17,903 real photos
│   │   ├── ai_generated/          #   3,481 AI-generated
│   │   └── ai_altered/            #  28,616 AI-altered deepfakes
│   ├── metadata/all.csv           # Consolidated metadata
│   ├── collectors/                # Collection scripts
│   └── docs/                      # Documentation
├── compose.yaml                   # Docker Compose
├── Dockerfile.api
├── Dockerfile.web
├── nginx.conf
├── .gitignore
├── PROJECT_CONTEXT.md
└── README.md
```

---

## Commands

### Train the model
```bash
cd model
pip install -r requirements.txt
python -m src.train
```
Takes ~30 min on consumer GPU. Checkpoints saved to `model/checkpoints/`.

### Start API server
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```
Endpoints:
- `GET /` — health check
- `POST /predict` — single image prediction
- `POST /predict/batch` — batch prediction
- `GET /usage` — rate limit usage

### Start website
```bash
cd web
npm install
npm run dev
```
Opens at `http://localhost:3000`

### Production deployment
```bash
docker compose up -d
```

---

## Commercial Tiers (API Rate Limiting)

| Tier | Price | Rate | Features |
|---|---|---|---|
| Free | $0 | 10 req/min | Basic probability score |
| Pro | $9.99/mo | 100 req/min | +Heatmaps, PDF reports |
| Enterprise | Custom | 1000 req/min | +On-premise, SLA |

---

## Publication Target

| Journal | IF | Status |
|---|---|---|
| IEEE TIFS | 6.3 | Manuscript drafted |
| Pattern Recognition | 7.5 | Draft suitable |
| Expert Systems w/ Apps | 8.5 | Alternative |

---

## What's Next

1. **Train MFFT**: `cd model && python -m src.train`
2. **Evaluate**: `python -m src.evaluate`
3. **Deploy API**: `uvicorn api.main:app --reload`
4. **Deploy website**: `cd web && npm run dev`
5. **Refine manuscript**: Update accuracy numbers after training
6. **Submit to journal**

---

## API Issues (if collecting more data)

| API | Status | Error |
|---|---|---|
| Unsplash | ❌ Fails | ConnectionResetError (10054) |
| Pixabay | ⚠️ Partial | 400 after page 3 |
| Pexels | ✅ Works | Most reliable |
| CivitAI | ❌ Fails | NoneType .get error |
| HuggingFace | ❌ Fails | 500 error |
| Pollinations | ✅ Works | Very slow (~1.5s/img) |

---

## Key Files Quick Reference

| File | What It Does |
|---|---|
| `model/src/model.py` | MFFT architecture — 282 lines |
| `model/src/train.py` | Training loop with mixed precision |
| `model/src/dataset.py` | Loads images from dataset/images/ |
| `api/main.py` | FastAPI server with rate limiting |
| `api/model_server.py` | Model loading & inference pipeline |
| `web/app/page.tsx` | Commercial landing page |
| `paper/manuscript.md` | Full Q1 journal paper |
| `dataset/collectors/collect_to_50k.py` | Consolidate & collect remaining images |
| `compose.yaml` | Docker Compose for production |
