# Run Sequence — MFFT Training Pipeline

## Phase 1: Test (Pipeline Validation)
Run these first to verify everything works (5K subset, 1 epoch each):

| Order | Notebook | Purpose |
|-------|----------|---------|
| 1 | `model/test_model/mfft_smoke.ipynb` | Quickest sanity check (tiny variant) |
| 2 | `model/test_model/mfft_tiny.ipynb` | Validate tiny pipeline |
| 3 | `model/test_model/mfft_base.ipynb` | Validate base pipeline |
| 4 | `model/test_model/mfft_large.ipynb` | Validate large pipeline |
| 5 | `model/test_model/baselines.ipynb` | Validate all 10 baselines |
| 6 | `model/test_model/ablation.ipynb` | Validate 9 ablation configs |

## Phase 2: Full Training (2.7M images, 20 epochs each)

| Order | Notebook | Outputs |
|-------|----------|---------|
| 1 | `model/train_mfft_tiny.ipynb` | `model/checkpoints/tiny_model/` |
| 2 | `model/train_mfft_base.ipynb` | `model/checkpoints/base_model/` |
| 3 | `model/train_mfft_large.ipynb` | `model/checkpoints/large_model/` |
| 4 | `model/train_baselines.ipynb` | `model/checkpoints/{model}/` + `paper/results/{model}/` |
| 5 | `model/train_ablation_study.ipynb` | `paper/results/ablation/` |

## Phase 3: Paper Outputs

Generate figures and tables for each variant:

```bash
python model/generate_paper_outputs.py --variant tiny  --checkpoint model/checkpoints/tiny_model/best.pt
python model/generate_paper_outputs.py --variant base  --checkpoint model/checkpoints/base_model/best.pt
python model/generate_paper_outputs.py --variant large --checkpoint model/checkpoints/large_model/best.pt
```

Then fill `[TBD]` values in `paper/manuscript.md`.

## Phase 4: API Deployment

```bash
docker build -f Dockerfile.api -t mfft-api .
docker run -p 8000:8000 mfft-api
```

## Notes

- Test notebooks output to `paper/result/test/` (isolated from production results)
- Production outputs go to `paper/results/` (no 's' in 'result')
- Each full training takes ~4-8 hours per MFFT variant on a single GPU
- Baselines notebook trains all 10 models sequentially (~24+ hours)
- Check `model/checkpoints/` for trained weights after each run