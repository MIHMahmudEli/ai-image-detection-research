"""Assemble and deploy the MFFT detection API as a Hugging Face Docker Space.

Single source of truth: copies api/ and model/src/model.py from the repo at
deploy time (never edit hf_space/ by hand), bundles the demo checkpoints,
and pushes to the Space repo.

Usage (from repo root):
    python deploy_hf_space.py            # assemble + deploy
    python deploy_hf_space.py --dry-run  # assemble only
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SPACE_DIR = REPO / "hf_space"
SPACE_ID = "MohsinEli/mfft-api"

# demo checkpoints: pilot (1-epoch) models until the DGX run replaces them
CHECKPOINTS = {
    "tiny":  REPO / "model" / "checkpoints" / "test" / "tiny_model" / "best.pt",
    "base":  REPO / "model" / "checkpoints" / "test" / "base_model" / "best.pt",
    "large": REPO / "model" / "checkpoints" / "test" / "large_model" / "best.pt",
}

README = """---
title: MFFT AI Image Detector API
emoji: \U0001F50D
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# MFFT — Multi-Frequency Fusion Transformer, Detection API

FastAPI inference server for AI-generated image detection. Three model
variants are served and selectable per request.

> **Demo checkpoints:** these are *pilot* models (1 training epoch on a 5K
> subset) published for pipeline demonstration. Full-scale weights will
> replace them after the training campaign completes.

## Endpoints

- `GET /` — health
- `GET /models` — available variants (`tiny` 372K, `base` 1.62M, `large` 6.30M)
- `POST /predict?model=base` — multipart `file=<image>`; returns prediction,
  probabilities, anomaly heatmap (base64 PNG), frequency band contributions
- `POST /predict/batch?model=large` — multiple files

## Example

```bash
curl -X POST "https://mohsineli-mfft-api.hf.space/predict?model=large" \\
  -H "Authorization: Bearer free" \\
  -F "file=@image.jpg"
```
"""

DOCKERFILE = """FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \\
 && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MFFT_CHECKPOINT_DIR=/app/checkpoints \\
    MFFT_DEMO=1 \\
    MFFT_IMAGE_SIZE=224 \\
    HF_HOME=/tmp/hf

USER user
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
"""

REQUIREMENTS = """fastapi
uvicorn[standard]
python-multipart
pillow
numpy
"""


def assemble():
    if SPACE_DIR.exists():
        shutil.rmtree(SPACE_DIR)
    (SPACE_DIR / "api").mkdir(parents=True)
    (SPACE_DIR / "model" / "src").mkdir(parents=True)
    (SPACE_DIR / "checkpoints").mkdir()

    (SPACE_DIR / "README.md").write_text(README, encoding="utf-8")
    (SPACE_DIR / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (SPACE_DIR / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    # api package (same layout as the repo so model_server's path logic works)
    for f in ("main.py", "model_server.py", "schemas.py"):
        shutil.copy(REPO / "api" / f, SPACE_DIR / "api" / f)
    (SPACE_DIR / "api" / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy(REPO / "model" / "src" / "model.py",
                SPACE_DIR / "model" / "src" / "model.py")

    missing = []
    for variant, src in CHECKPOINTS.items():
        if src.exists():
            shutil.copy(src, SPACE_DIR / "checkpoints" / f"{variant}.pt")
        else:
            missing.append(f"{variant}: {src}")
    if missing:
        sys.exit("Missing checkpoints:\n  " + "\n  ".join(missing))

    total = sum(f.stat().st_size for f in SPACE_DIR.rglob("*") if f.is_file())
    print(f"Assembled {SPACE_DIR} ({total / 1e6:.1f} MB)")
    for f in sorted(SPACE_DIR.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(SPACE_DIR)}  ({f.stat().st_size / 1e6:.2f} MB)")


def deploy():
    from dotenv import dotenv_values
    from huggingface_hub import HfApi

    token = dotenv_values(REPO / "dataset" / ".env").get("HUGGINGFACE_TOKEN")
    if not token:
        sys.exit("HUGGINGFACE_TOKEN not found in dataset/.env")
    api = HfApi(token=token)
    print(f"Deploying as: {api.whoami()['name']}")

    api.create_repo(SPACE_ID, repo_type="space", space_sdk="docker",
                    exist_ok=True)
    api.upload_folder(repo_id=SPACE_ID, repo_type="space",
                      folder_path=str(SPACE_DIR),
                      commit_message="Deploy MFFT multi-model detection API")
    print(f"\nDeployed: https://huggingface.co/spaces/{SPACE_ID}")
    print("API URL:  https://mohsineli-mfft-api.hf.space")
    print("(first build takes a few minutes - watch the Space's Logs tab)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    assemble()
    if not args.dry_run:
        deploy()
