"""
Shared pipeline configuration.
Single source of truth for repo IDs, dataset mappings, and constants.
"""

import os
from pathlib import Path

# ── HF Repos ──
# Overridable via env var so code changes aren't needed for a new repo.
HF_MANIFEST_REPO = os.environ.get(
    "HF_MANIFEST_REPO_OVERRIDE",
    "studyhub991/mfft-master-manifest",
)
HF_CHECKPOINT_REPO = os.environ.get(
    "HF_CHECKPOINT_REPO_OVERRIDE",
    "studyhub991/mfft-checkpoints",
)

MANIFEST_PATH_IN_REPO = "manifest/split_manifest.json"

# ── Seed / splits ──
SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ── Labels ──
LABEL_MAP = {"real": 0, "ai_generated": 1, "deepfake": 2}
CLASS_NAMES = ["real", "ai_generated", "deepfake"]

# ── Image discovery ──
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Kaggle mount slug → (label, image_subdir inside the mount) ──
KAGGLE_DATASETS = {
    "stable-diffusion":                 ("ai_generated", "Stable Diffusion/images"),
    "places365":                        ("real",          "train"),
    "open-images-v7-dataset":           ("real",          "Open-Images-V7-Dataset/open-images-v7/train/images"),
    "ntire2026":                        ("real",          "NTIRE2026"),
    "midjourney":                       ("ai_generated",  "Midjourney/Datasetfordream"),
    "mfft-real":                        ("real",          "Mfft_real"),
    "genimage-ai":                      ("ai_generated",  "genimage_ai"),
    "faceforensics":                    ("deepfake",      "cropped_images"),
    "dfdc-faces-of-the-train-sample":   ("deepfake",      "train/fake"),
    "dall-e3":                          ("ai_generated",  "DALL-E3"),
    "celebdf-v2image-dataset":          ("deepfake",      "Celeb_V2"),
}

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
