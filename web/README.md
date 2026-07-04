# ImageVerify AI — Web App

Next.js front-end for **MFFT** (Multi-Frequency Fusion Transformer), an
AI-generated image detector. Upload an image and get an explainable verdict:
real vs AI probability, per-frequency-band analysis, and an anomaly heatmap.

**Live detection API:** [`mfft-api` on Hugging Face Spaces](https://huggingface.co/spaces/MohsinEli/mfft-api)

## Features

- Drag-and-drop image upload (PNG/JPG/WebP, up to 20 MB)
- **Model selector** — switch between MFFT-Tiny (372K), MFFT-Base (1.62M),
  and MFFT-Large (6.30M parameters) from the settings button
- Explainable results: probability bars, frequency-band contributions,
  anomaly heatmap
- Model choice persisted in `localStorage`

> **Note:** the API currently serves *pilot* demo checkpoints (1 training
> epoch). Full-scale weights will replace them after the training campaign.

## Getting started

```bash
npm install
cp .env.example .env.local   # points at the live API by default
npm run dev                  # http://localhost:3000
```

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Detection API base URL | `http://localhost:8000` |

The app proxies `/api/*` to `NEXT_PUBLIC_API_URL` (see `next.config.js`), so
the browser never talks to the API cross-origin.

## Deploy (Vercel)

1. Import this repo in Vercel
2. Set env var `NEXT_PUBLIC_API_URL=https://mohsineli-mfft-api.hf.space`
3. Deploy

## Related

- Research codebase: MFFT — Multi-Frequency Fusion Transformer for
  AI-Generated Image Detection (paper in preparation)
- API source lives in the research monorepo (`api/`) and is deployed via
  `deploy_hf_space.py`
