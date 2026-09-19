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
# Balanced training: cap each class to avoid bias toward any one type
# AI-generated: ~1.5M available (artifact 25 generators)
# Real: ~1.8M available (places365) — cap to match AI count
# Deepfake: ~125K available (faceforensics + dfdc)
TARGET_TOTAL_IMAGES = 1_800_000  # Balanced: ~600K per class
TARGET_CLASS_RATIOS = {"real": 0.33, "ai_generated": 0.33, "deepfake": 0.34}
TEST_MIN_PER_SHARD = 300
MAX_FRAMES_PER_VIDEO = 20

# ── Deepfake shards that require video/identity-grouped splitting ──
DEEPFAKE_VIDEO_SHARDS = {"faceforensics", "dfdc-faces-of-the-train-sample", "celebdf-v2image-dataset"}

# ── Labels ──
LABEL_MAP = {"real": 0, "ai_generated": 1, "deepfake": 2}
CLASS_NAMES = ["real", "ai_generated", "deepfake"]

# ── Image discovery ──
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Training datasets (attached to Kaggle inputs) ──
# Tier 1: TRAINING — these are seen during training
# Tier 2: CROSS-DOMAIN — held out completely, used for generalization tests
# Tier 3: NOT USED — too large or redundant
#
# Kaggle mount slug → (label, image_subdir inside the mount)
KAGGLE_DATASETS = {
    # ── Tier 1: TRAINING SET ──
    # Real images (capped at ~600K to match AI-generated count)
    "places365":                        ("real",          "train"),

    # Deepfake (face manipulation)
    "faceforensics":                    ("deepfake",      "cropped_images"),
    "dfdc-faces-of-the-train-sample":   ("deepfake",      "train/fake"),

    # ── Artifact sub-datasets (AI-generated, 25 generators) ──
    # Each has metadata.csv with columns: filename, image_path, target, category
    # target: 0=real, 1-6=fake classes
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

# ── Cross-domain validation datasets (NEVER seen during training) ──
# These are held out completely to test generalization to unseen generators
CROSS_DOMAIN_DATASETS = {
    "dall-e3":                          ("ai_generated",  "DALL-E3"),
    "celebdf-v2image-dataset":          ("deepfake",      "Celeb_V2"),
    "genimage-ai":                      ("ai_generated",  "genimage_ai"),
}

# ── Kaggle mount root ──
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
