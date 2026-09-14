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
    "MohsinElis/mfft-master-manifest",
)
HF_CHECKPOINT_REPO = os.environ.get(
    "HF_CHECKPOINT_REPO_OVERRIDE",
    "MohsinElis/mfft-checkpoints",
)

MANIFEST_PATH_IN_REPO = "manifest/split_manifest.json"

# ── Seed / splits ──
SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ── Rebuild manifest targets (FIX 1–4) ──
TARGET_TOTAL_IMAGES = 650_000
TARGET_CLASS_RATIOS = {"real": 0.474, "ai_generated": 0.438, "deepfake": 0.088}
TEST_MIN_PER_SHARD = 300
MAX_FRAMES_PER_VIDEO = 20

# ── Deepfake shards that require video/identity-grouped splitting ──
DEEPFAKE_VIDEO_SHARDS = {"faceforensics", "dfdc-faces-of-the-train-sample", "celebdf-v2image-dataset"}

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
    # ── awsaf49/artifact-dataset sub-datasets (AI-generated) ──
    "artifact-afhq":                    ("ai_generated",  "images"),
    "artifact-big_gan":                 ("ai_generated",  "images"),
    "artifact-celebahq":                ("ai_generated",  "images"),
    "artifact-cips":                    ("ai_generated",  "images"),
    "artifact-cycle_gan":               ("ai_generated",  "images"),
    "artifact-ddpm":                    ("ai_generated",  "images"),
    "artifact-denoising_diffusion_gan": ("ai_generated",  "images"),
    "artifact-diffusion_gan":           ("ai_generated",  "images"),
    "artifact-face_synthetics":         ("ai_generated",  "images"),
    "artifact-ffhq":                    ("ai_generated",  "images"),
    "artifact-gansformer":              ("ai_generated",  "images"),
    "artifact-gau_gan":                 ("ai_generated",  "images"),
    "artifact-generative_inpainting":   ("ai_generated",  "images"),
    "artifact-glide":                   ("ai_generated",  "images"),
    "artifact-lama":                    ("ai_generated",  "images"),
    "artifact-landscape":               ("ai_generated",  "images"),
    "artifact-latent_diffusion":        ("ai_generated",  "images"),
    "artifact-metfaces":                ("ai_generated",  "images"),
    "artifact-palette":                 ("ai_generated",  "images"),
    "artifact-pro_gan":                 ("ai_generated",  "images"),
    "artifact-projected_gan":           ("ai_generated",  "images"),
    "artifact-sfhq":                    ("ai_generated",  "images"),
    "artifact-star_gan":                ("ai_generated",  "images"),
    "artifact-stylegan1":               ("ai_generated",  "images"),
    "artifact-stylegan2":               ("ai_generated",  "images"),
    "artifact-stylegan3":               ("ai_generated",  "images"),
    "artifact-taming_transformer":      ("ai_generated",  "images"),
    "artifact-vq_diffusion":            ("ai_generated",  "images"),
}

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
