# 🎯 DEEPFAKE DETECTION DATASET - COMPLETE GUIDE

## Executive Summary

For a **production deepfake detection system**, your dataset should be:

```
Total: 50,000 images

├── Real images (legitimate)        → 15,000 (30%)
├── AI-generated images (synthetic) → 10,000 (20%)
└── AI-altered images (deepfakes)   → 25,000 (50%) ⚠️ CRITICAL
```

**Why 50% AI-altered?** Deepfake detection is hardest. You need 5× more altered samples to train robust features that detect splicing artifacts, blending seams, inconsistent lighting/shadows, face distortions, etc.

---

## Dataset Breakdown by Component

### 1️⃣ REAL IMAGES (15,000)
**Purpose:** Provide baseline of completely untouched photographs

**Sources:**
- Unsplash   → 5,000
- Pexels     → 5,000
- Pixabay    → 5,000

**Characteristics:**
- Natural photos from real photographers
- Diverse: portraits, landscapes, food, objects, animals, cities
- Different resolutions, lighting, styles
- **NO AI modification**

**Script:** `1_real_images_collector.py`

**Output Structure:**
```
real_dataset/
├── images/ (15,000 .jpg files)
├── metadata/
│   ├── dataset_metadata_real.csv   ← 15,000 rows with metadata
│   └── dataset_metadata_real.json
└── collection_log.json
```

**CSV Columns:**
- `image_id`: Unique identifier
- `filename`: Path to saved image
- `source`: "unsplash" | "pexels" | "pixabay"
- `source_url`: Link to original
- `download_date`: ISO timestamp
- `image_type`: "real"
- `width`, `height`: Image dimensions
- `format`, `file_size_kb`, `color_space`: Technical specs
- `source_metadata`: Author, photographer info
- `md5_hash`: For deduplication
- `analysis_notes`: Query used

---

### 2️⃣ AI-GENERATED IMAGES (10,000)
**Purpose:** Provide "purely synthetic" images made entirely by AI

**Sources:**
- CivitAI (Stable Diffusion, LoRA models) → 4,000
- DiffusionDB (14M Stable Diffusion dataset) → 4,000
- Pollinations.ai (free generation) → 2,000

**Characteristics:**
- 100% AI-generated from text prompts
- No real photos involved
- Diverse styles: realistic, fantasy, abstract, anime, digital art
- Includes various generators' signatures
- **No human subjects altered (all AI creations)**

**Script:** `2_ai_generated_images_collector.py`

**Output Structure:**
```
ai_generated_dataset/
├── images/ (10,000 .jpg files)
├── metadata/
│   ├── dataset_metadata_ai_generated.csv   ← 10,000 rows
│   └── dataset_metadata_ai_generated.json
└── collection_log.json
```

**CSV Columns:**
(Same as real + these additional)
- `source`: "civitai" | "diffusiondb" | "pollinations"
- `source_metadata.prompt`: Text prompt used
- `source_metadata.seed`: Random seed (Pollinations)
- `source_metadata.model`: Generator model (CivitAI)
- `analysis_notes`: Generator info

---

### 3️⃣ AI-ALTERED IMAGES (25,000) ⭐ CRITICAL FOR DEEPFAKE DETECTION

**Purpose:** Real photos modified with AI (the actual deepfake threat)

**Alteration Methods (Ratio: 40% / 32% / 16% / 12%):**

```
├── Face Swap / Deepfake    (10,000 - 40%)
│   Real person A's body + AI-generated face B
│   Detection signals: Face seams, lighting mismatch, eye artifacts
│
├── Inpainting              (8,000 - 32%)
│   Real photo with AI-filled masked regions (~30-50% area)
│   Detection signals: Blending artifacts, unrealistic details in filled region
│
├── Outpainting             (4,000 - 16%)
│   Real photo extended with AI-generated edges
│   Detection signals: Edge inconsistencies, style breaks
│
└── Style Transfer          (3,000 - 12%)
    Real photo with AI artistic style applied
    Detection signals: Texture artifacts, color shifts
```

**Why This Ratio?**
- Face swaps are the BIGGEST threat (deepfakes = face swaps primarily)
- Inpainting is also common (removing unwanted objects/people)
- Outpainting is less common but growing
- Style transfer is least concerning for "deepfake" but included for robustness

**Script:** `3_ai_altered_images_generator.py`

**Requirements:**
```bash
pip install diffusers transformers torch accelerate Pillow numpy
# Optional (for better face detection):
pip install face-recognition opencv-python
```

**Output Structure:**
```
ai_altered_dataset/
├── images/ (25,000 .jpg files)
├── metadata/
│   ├── dataset_metadata_ai_altered.csv   ← 25,000 rows
│   └── dataset_metadata_ai_altered.json
└── collection_log.json
```

**CSV Columns:**
(Same as real + these additional)
- `alteration_method`: "face_swap" | "inpainting" | "outpainting" | "style_transfer"
- `base_image_source`: "real"
- `source_metadata.base_image`: Original real image name
- `source_metadata.ai_source_image`: AI source (for face swaps)
- `source_metadata.mask_coverage`: % of image masked (inpainting)
- `analysis_notes`: Detailed description of alteration

---

## 🚀 Execution Order

### Step 1: Collect Real Images (~30-45 mins)
```bash
python 1_real_images_collector.py
# Output: ./real_dataset/ (15,000 images)
```

### Step 2: Collect AI-Generated Images (~120-180 mins)
```bash
python 2_ai_generated_images_collector.py
# Output: ./ai_generated_dataset/ (10,000 images)
# Note: Pollinations is slow (~2-3 sec/image), so patience required
```

### Step 3: Generate AI-Altered Deepfakes (~60-90 mins)
```bash
python 3_ai_altered_images_generator.py
# Inputs: ./real_dataset/images/ + ./ai_generated_dataset/images/
# Output: ./ai_altered_dataset/ (25,000 images)
```

### Step 4: Combine All Metadata
```python
import pandas as pd

real_df = pd.read_csv('./real_dataset/metadata/dataset_metadata_real.csv')
ai_df = pd.read_csv('./ai_generated_dataset/metadata/dataset_metadata_ai_generated.csv')
altered_df = pd.read_csv('./ai_altered_dataset/metadata/dataset_metadata_ai_altered.csv')

combined = pd.concat([real_df, ai_df, altered_df], ignore_index=True)
combined.to_csv('./combined_dataset_50k_metadata.csv', index=False)
print(f"Total: {len(combined)} images")
print(combined['image_type'].value_counts())
```

---

## 📊 Final Dataset Statistics

```
Total Images: 50,000

By Type:
  Real          → 15,000 (30%)
  AI-Generated  → 10,000 (20%)
  AI-Altered    → 25,000 (50%)
                  ──────────────
  Total           50,000 (100%)

By Alteration Method (AI-Altered only):
  Face Swap     → 10,000 (40%)
  Inpainting    →  8,000 (32%)
  Outpainting   →  4,000 (16%)
  Style Transfer→  3,000 (12%)
```

---

## 🧠 Training Split

For your model training:
```python
# After combining all metadata:
from sklearn.model_selection import train_test_split

# 70% train, 15% val, 15% test
train, temp = train_test_split(combined, test_size=0.30, random_state=42, stratify=combined['image_type'])
val, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp['image_type'])

train.to_csv('./splits/train.csv', index=False)  # 35,000 images
val.to_csv('./splits/val.csv', index=False)      #  7,500 images
test.to_csv('./splits/test.csv', index=False)    #  7,500 images
```

---

## ⚠️ Important Notes

### Why NOT 50% real / 25% AI / 25% altered?
For **general AI detection** (detect ANY AI use), that ratio works.
But for **deepfake detection specifically**, the threat model changes:
- Users mostly worry about real → fake (face swaps, deepfakes)
- Not about distinguishing pure AI-gen from real
- So we weight heavily on realistic alterations (face swaps 40%)

### Generator Diversity (CRITICAL)
Your model will overfit to specific generators if you don't vary:
- **Real photos:** Different photographers, styles
- **AI-gen:** Use CivitAI (SD1.5/SDXL/LoRA), DiffusionDB (SD 1.5), Pollinations (Flux/ProtoVision)
- **Altered:** Mix face swap sources (real→AI faces)

Future versions should add:
- Midjourney images
- DALL-E 3 images
- Adobe Firefly
- Claude image generation

### Metadata is Critical
Your `.csv` files are **essential for training**:
- Labels (`image_type`, `alteration_method`)
- Source tracking (for cross-validation: "Is model biased to CivitAI?")
- Prompt + seed info (research: "Does specific prompt signature leak?")

---

## 📁 Final Directory Structure

```
ai_detection_research/
├── real_dataset/
│   ├── images/ (15,000 .jpg)
│   └── metadata/
│       ├── dataset_metadata_real.csv
│       ├── dataset_metadata_real.json
│       └── collection_log.json
│
├── ai_generated_dataset/
│   ├── images/ (10,000 .jpg)
│   └── metadata/
│       ├── dataset_metadata_ai_generated.csv
│       ├── dataset_metadata_ai_generated.json
│       └── collection_log.json
│
├── ai_altered_dataset/
│   ├── images/ (25,000 .jpg)
│   └── metadata/
│       ├── dataset_metadata_ai_altered.csv
│       ├── dataset_metadata_ai_altered.json
│       └── collection_log.json
│
└── combined/
    ├── dataset_metadata_50k_all.csv
    └── splits/
        ├── train.csv (35k)
        ├── val.csv (7.5k)
        └── test.csv (7.5k)
```

---

## ✅ Validation Checklist

Before training your model:
- [ ] All 50,000 images downloaded
- [ ] All `.csv` files have correct columns
- [ ] No duplicate images (check `md5_hash` column)
- [ ] Image dimensions: min 512×512 (resize if needed)
- [ ] Labels are balanced in train/val/test splits
- [ ] Can load images from file paths in CSV
- [ ] Metadata JSON files are valid JSON

---

## 🎯 Next Steps

1. **Run all 3 scripts** to build the dataset
2. **Create combined CSV** with all metadata
3. **Split train/val/test** with stratified sampling
4. **Train model** (EfficientNet, ViT, or custom CNN)
5. **Test on held-out set** to measure accuracy by alteration method

Good luck! 🚀
