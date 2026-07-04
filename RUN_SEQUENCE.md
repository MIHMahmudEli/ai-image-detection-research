# Run Sequence

> **This document is superseded by [`DGX_RUN_GUIDE.md`](DGX_RUN_GUIDE.md)** —
> use that for the full-scale training procedure, notebook order, LOGO
> protocol, and local verification instructions.

## Output tree (current layout)

| Location | Contents |
|---|---|
| `paper/result/test/` | Pilot results (frozen — used by the pilot draft) |
| `paper/result/verify/` | Local smoke-verification outputs (`model/test_model_verify/`) |
| `paper/result/full_scale/` | DGX outputs: train notebooks, `paper_evals`, `logo` |
| `model/checkpoints/{variant}_model/` | Full-scale checkpoints |
| `model/checkpoints/verify/` | Verification checkpoints |
| `model/checkpoints/logo/<gen>/` | LOGO per-generator checkpoints |

## Notebook order (details in DGX_RUN_GUIDE.md)

```
prepare_training_manifest.py          (script, once, before everything)
train_mfft_base.ipynb                 (creates the shared split)
train_baselines.ipynb
train_mfft_tiny.ipynb
train_mfft_large.ipynb
train_ablation_study.ipynb
paper_evals.ipynb
train_logo.ipynb
```

## Local verification (before requesting GPU time)

Run everything in `model/test_model_verify/` on a CPU machine — smoke mode
engages automatically and writes to the quarantined `verify` locations.

## API deployment (after training)

```bash
docker build -f Dockerfile.api -t mfft-api .
docker run -p 8000:8000 mfft-api
# refuses to start without a checkpoint; override with MFFT_CHECKPOINT=/path
# or MFFT_ALLOW_RANDOM=1 (development only)
```
