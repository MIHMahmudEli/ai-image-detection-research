# Datasets for AI-Image-Detection-Research

---

## AI-Generated Images Datasets

### 1. GenImage ⭐ (Best — Multiple Generators)

| Detail | Info |
|---|---|
| **Link** | https://github.com/GenImage-Dataset/GenImage |
| **Images** | 1.35 million |
| **Generators** | Stable Diffusion, Midjourney, DALL-E, FLUX, Glide, Wukong, BigGAN, VQDM |
| **Size** | ~50 GB per generator (download individually) |
| **Download** | Google Drive links from the GitHub page |
| **License** | Research purposes |

### 2. DiffusionDB

| Detail | Info |
|---|---|
| **Link** | https://huggingface.co/datasets/poloclub/diffusiondb |
| **Images** | 2 million (or 14 million Large version) |
| **Generator** | Stable Diffusion 1.5 only |
| **Size** | ~500 MB per zip (~1000 images per zip) |
| **Download** | `download.py` script from the repo, or direct HTTP |
| **Access** | Public (no auth needed for downloads) |

### 3. CIFAKE (Kaggle)

| Detail | Info |
|---|---|
| **Link** | https://www.kaggle.com/datasets/sagnik1511/cifake-ai-image-detection-dataset |
| **Images** | 120,000 (60K real + 60K AI) |
| **Generator** | Stable Diffusion 2.1 |
| **Size** | ~2 GB |
| **Download** | Kaggle dataset page → Download button |

### 4. AI vs Real Images (Kaggle)

| Detail | Info |
|---|---|
| **Link** | https://www.kaggle.com/datasets/rhythmghai/ai-vs-real-images-dataset |
| **Images** | ~10,000 |
| **Generator** | Multiple |
| **Size** | 237 MB |
| **Download** | Kaggle dataset page → Download |

### 5. Detect AI-Generated Faces (Kaggle)

| Detail | Info |
|---|---|
| **Link** | https://www.kaggle.com/datasets/shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset |
| **Images** | High-quality AI face images |
| **Size** | 116 MB |
| **Download** | Kaggle dataset page → Download |

### 6. GenImage Subset (Kaggle)

| Detail | Info |
|---|---|
| **Link** | https://www.kaggle.com/datasets/renhuang8/genimage-subset-detection |
| **Images** | Subset of GenImage |
| **Size** | 3.4 GB |
| **Download** | Kaggle dataset page → Download |

### 7. DALL-E 3 Datasets (Various)

| Source | Link | Size |
|---|---|---|
| Kaggle | Search "DALL-E 3 Generated Images" | ~1-5 GB |
| HuggingFace | Search "dalle3-dataset" | ~10 GB |

---

## Deepfake / AI-Altered Images Datasets

### 8. FaceForensics++ ⭐

| Detail | Info |
|---|---|
| **Link** | https://github.com/ondyari/FaceForensics |
| **Images** | 1.8K real deepfakes |
| **Methods** | Deepfakes, FaceShifter, Face2Face, NeuralTextures |
| **Size** | ~1 GB |
| **Download** | Requires filling a Google Form on GitHub (approval in 1-3 days) |
| **Status** | ⏳ Pending submission |

### 9. Celeb-DF

| Detail | Info |
|---|---|
| **Link** | https://github.com/danmohaha/Celeb-DF |
| **Images** | 5,600 high-quality deepfakes |
| **Size** | ~2 GB |
| **Download** | Google Drive from GitHub |

### 10. DFDC (Deepfake Detection Challenge)

| Detail | Info |
|---|---|
| **Link** | https://www.kaggle.com/competitions/deepfake-detection-challenge/data |
| **Images** | 120,000 real-world deepfakes |
| **Size** | ~5 GB |
| **Download** | Kaggle competition page → Download |

### 11. WildDeepfake

| Detail | Info |
|---|---|
| **Link** | https://github.com/deepfakeinthewild/deepfake-in-the-wild |
| **Images** | 7,000 in-the-wild deepfakes |
| **Size** | ~3 GB |
| **Download** | Google Drive from GitHub |

---

## Recommendations

| Priority | Dataset | Why |
|---|---|---|
| **1** | **FaceForensics++** | Standard benchmark for deepfake detection |
| **2** | **GenImage** | Best diversity (SD, MJ, DE, FLUX, etc.) |
| **3** | **Celeb-DF** | High-quality deepfakes |
| **4** | **DiffusionDB** | Large-scale SD images |
| **5** | **DFDC** | Real-world deepfakes |

## After Downloading

Place downloaded images in:

```
dataset/images/ai_generated/    → for AI-generated (CivitAI + DiffusionDB + GenImage + Kaggle)
dataset/images/real/            → for real images (we already have 24K)
dataset/images/ai_altered/      → for deepfakes (FF++, Celeb-DF, DFDC)
```

Then run the metadata regeneration script to update the clean CSV.
