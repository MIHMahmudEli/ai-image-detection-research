# Dataset Collector API - Setup & Usage Guide

## Overview

The Dataset Collector API is a comprehensive Python tool for collecting images with detailed metadata for AI-generated image detection research. It:

✓ Collects real images from Unsplash, Pexels, and Pixabay
✓ Collects AI-generated images from HuggingFace datasets
✓ Imports images from local folders
✓ Extracts comprehensive metadata for each image
✓ Detects and flags duplicates
✓ Generates CSV and JSON tracking databases
✓ Provides collection reports and statistics

---

## Installation

### Step 1: Install Python Requirements

```bash
pip install requests pandas pillow python-dotenv aiohttp tqdm beautifulsoup4 huggingface_hub datasets
```

### Step 2: Get API Keys

You need API keys from free services:

#### **Unsplash API**
1. Go to: https://unsplash.com/oauth/applications
2. Click "New Application"
3. Accept terms and create app
4. Copy your "Access Key"

#### **Pexels API**
1. Go to: https://www.pexels.com/api/
2. Click "Get Started"
3. Sign up/Log in
4. Copy your API key

#### **Pixabay API**
1. Go to: https://pixabay.com/api/docs/
2. Sign up/Log in
3. Copy your API key

#### **HuggingFace Token** (Optional, for AI-generated images)
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Select "Read" access
4. Copy your token

### Step 3: Configure Environment

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

Your `.env` file should look like:
```
UNSPLASH_API_KEY=xxxxx
PEXELS_API_KEY=xxxxx
PIXABAY_API_KEY=xxxxx
HUGGINGFACE_TOKEN=xxxxx
```

---

## Quick Start (5 Minutes)

### Basic Usage

```python
from pathlib import Path
from dataset_collector_api import DatasetCollector

# Create collector
collector = DatasetCollector(output_dir=Path("./my_dataset"))

# Configure collection
config = {
    "real_image_count": 250,      # Collect 250 real images
    "ai_image_count": 250,        # Collect 250 AI images
    "local_folders": {}           # Optional: add local folders
}

# Run collection
report = collector.run_full_collection(config)

# Check results
print(f"Total images: {report['total_images']}")
print(f"Real images: {report['real_images']}")
print(f"AI images: {report['ai_generated_images']}")
```

### Run from Command Line

```bash
python dataset_collector_api.py
```

This will:
1. Load API keys from `.env`
2. Create `./ai_detection_dataset` directory
3. Collect 250 real + 250 AI images
4. Save metadata to CSV and JSON
5. Generate collection report

---

## Advanced Usage

### Customize Queries

```python
from dataset_collector_api import UnsplashCollector
from pathlib import Path

unsplash = UnsplashCollector(api_key="your_key", output_dir=Path("./dataset"))

# Custom queries
queries = ["portrait professional", "landscape nature", "urban street"]

for query in queries:
    collected = unsplash.collect(query, count=30)
    print(f"Collected {collected} images for '{query}'")

# Save metadata
unsplash.save_metadata(Path("./dataset/unsplash_metadata.csv"))
```

### Collect from Local Folders

```python
from dataset_collector_api import LocalFolderCollector
from pathlib import Path

# Collect real images from local folder
collector = LocalFolderCollector(output_dir=Path("./dataset"))
collected = collector.collect_from_folder(
    folder_path=Path("./my_real_images"),
    image_type="real"
)

print(f"Imported {collected} images")

# Save metadata
collector.save_metadata(Path("./dataset/local_metadata.csv"))
```

### Collect from Multiple Sources

```python
from dataset_collector_api import DatasetCollector
from pathlib import Path

collector = DatasetCollector(output_dir=Path("./full_dataset"))

# Load API keys from .env
api_keys = collector.load_api_keys()

# Collect real images from all three sources
collector.collect_real_images(api_keys, total_count=300)

# Collect AI images
collector.collect_ai_images(api_keys, total_count=300)

# Collect from local folders
local_folders = {
    "real": Path("./additional_real_images"),
    "ai_generated": Path("./additional_ai_images"),
}
collector.collect_from_local_folders(local_folders)

# Generate report and save
report = collector.generate_dataset_report()
collector.save_all_metadata()
```

### Access Collected Metadata

After collection, metadata is saved in multiple formats:

#### CSV Format (for spreadsheet analysis)
```python
import pandas as pd

# Load metadata
df = pd.read_csv("./ai_detection_dataset/dataset_metadata.csv")

# Analyze
print(f"Total images: {len(df)}")
print(f"Real vs AI:")
print(df['image_type'].value_counts())

print(f"\nBy source:")
print(df['source'].value_counts())

print(f"\nDuplicates found:")
print(f"{df['is_duplicate'].sum()} duplicates")
```

#### JSON Format (for detailed access)
```python
import json

# Load metadata
with open("./ai_detection_dataset/dataset_metadata.json") as f:
    metadata = json.load(f)

# Access first image metadata
first_image = metadata[0]
print(f"Image ID: {first_image['image_id']}")
print(f"Source: {first_image['source']}")
print(f"Dimensions: {first_image['width']}x{first_image['height']}")
print(f"File size: {first_image['file_size_kb']:.2f} KB")
print(f"Hash: {first_image['md5_hash']}")
```

---

## Output Directory Structure

After collection, your dataset directory looks like:

```
ai_detection_dataset/
├── real_images/
│   ├── UNSPLASH_12345.jpg
│   ├── UNSPLASH_67890.jpg
│   ├── PEXELS_11111.jpg
│   ├── PIXABAY_22222.jpg
│   └── ... (250+ real images)
│
├── ai_generated_images/
│   ├── STABLE_DIFFUSION_HF_multimodalart_0.png
│   ├── STABLE_DIFFUSION_HF_multimodalart_1.png
│   └── ... (250+ AI images)
│
├── dataset_metadata.csv          # All metadata in table format
├── dataset_metadata.json         # All metadata in JSON format
└── collection_log.json           # Collection statistics and report
```

---

## Metadata Fields Explanation

Each image has the following metadata recorded:

| Field | Description | Example |
|-------|-------------|---------|
| `image_id` | Unique identifier | `UNSPLASH_12345` |
| `filename` | Saved filename | `UNSPLASH_12345.jpg` |
| `source` | Collection source | `unsplash`, `pexels`, `stable_diffusion` |
| `source_url` | Original URL | `https://unsplash.com/photos/...` |
| `download_date` | When downloaded | `2024-05-20T14:30:00.123456` |
| `image_type` | Classification | `real` or `ai_generated` |
| `width` | Image width in pixels | `1920` |
| `height` | Image height in pixels | `1280` |
| `format` | Image format | `JPEG`, `PNG`, `GIF` |
| `file_size_kb` | File size in kilobytes | `245.5` |
| `color_space` | Color mode | `RGB`, `RGBA`, `CMYK` |
| `source_metadata` | Source-specific data | `{"author": "Name", "likes": 150}` |
| `md5_hash` | MD5 hash for duplicate detection | `a1b2c3d4e5f6...` |
| `is_duplicate` | Whether image is duplicate | `true` or `false` |
| `duplicate_of` | If duplicate, original image ID | `UNSPLASH_99999` |
| `processed` | Whether analyzed | `true` or `false` |
| `analysis_notes` | Notes from collection | `Query: portrait` |

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'requests'"

**Solution:** Install dependencies
```bash
pip install requests pandas pillow python-dotenv aiohttp tqdm beautifulsoup4 huggingface_hub datasets
```

### Issue: "UNSPLASH_API_KEY not found"

**Solution:** 
1. Check you created `.env` file
2. Check API key is in `.env` file
3. Run from same directory as `.env`

```bash
# Verify .env exists
ls -la .env

# Verify keys are set
cat .env | grep API_KEY
```

### Issue: "Rate limit exceeded"

**Solution:** The APIs have rate limits
- Unsplash: 50 requests/hour (free tier)
- Pexels: 200 requests/hour
- Pixabay: 50 requests/hour

Wait an hour or reduce `count` parameter.

### Issue: "timeout" errors during download

**Solution:** Network connection issue
```python
# Increase timeout
# In the code, change timeout from 10 to 30
response = self.session.get(image_url, timeout=30)
```

### Issue: "Authentication failed"

**Solution:** Check API key
```bash
# Test Unsplash API
curl -H "Authorization: Client-ID YOUR_KEY" \
  "https://api.unsplash.com/search/photos?query=test&per_page=1"
```

---

## Data Collection Best Practices

### 1. **Diverse Real Images**
Collect real images across categories:
- Portraits (faces, people)
- Landscapes (nature, outdoor)
- Objects (products, items)
- Text-heavy (documents, signs)
- Abstract (colors, patterns)

### 2. **Balanced Dataset**
Aim for 50/50 real vs AI-generated:
```python
config = {
    "real_image_count": 250,      # 50%
    "ai_image_count": 250,        # 50%
}
```

### 3. **Multiple AI Models**
Collect from different AI generators to get variety:
- DALL-E (distinctive hand/text artifacts)
- Midjourney (edge artifacts)
- Stable Diffusion (color/geometry artifacts)

### 4. **Document Everything**
Keep notes in metadata:
```python
metadata.analysis_notes = "Collected for portrait category, high resolution"
```

### 5. **Regular Backups**
```bash
# Backup your dataset
cp -r ai_detection_dataset ai_detection_dataset.backup

# Or use cloud storage
# gsutil -m cp -r ai_detection_dataset gs://your-bucket/
```

---

## Analyzing Your Dataset

### Quick Statistics

```python
import pandas as pd

df = pd.read_csv("ai_detection_dataset/dataset_metadata.csv")

print("=" * 50)
print("DATASET STATISTICS")
print("=" * 50)

print(f"\nTotal images: {len(df)}")

print(f"\nImage Type Distribution:")
print(df['image_type'].value_counts())

print(f"\nSource Distribution:")
print(df['source'].value_counts())

print(f"\nImage Format Distribution:")
print(df['format'].value_counts())

print(f"\nDimension Statistics:")
print(f"Average width: {df['width'].mean():.0f}px")
print(f"Average height: {df['height'].mean():.0f}px")
print(f"Average file size: {df['file_size_kb'].mean():.2f}KB")

print(f"\nDuplicates: {df['is_duplicate'].sum()}")

print(f"\nColor Space Distribution:")
print(df['color_space'].value_counts())
```

### Visualize Dataset Distribution

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ai_detection_dataset/dataset_metadata.csv")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Real vs AI
df['image_type'].value_counts().plot(kind='bar', ax=axes[0, 0], title='Image Type Distribution')
axes[0, 0].set_ylabel('Count')

# 2. By Source
df['source'].value_counts().plot(kind='barh', ax=axes[0, 1], title='Images by Source')
axes[0, 1].set_xlabel('Count')

# 3. File Size Distribution
df['file_size_kb'].hist(bins=30, ax=axes[1, 0], title='File Size Distribution')
axes[1, 0].set_xlabel('File Size (KB)')
axes[1, 0].set_ylabel('Count')

# 4. Dimensions
df.plot.scatter(x='width', y='height', ax=axes[1, 1], alpha=0.5, title='Image Dimensions')
axes[1, 1].set_xlabel('Width (px)')
axes[1, 1].set_ylabel('Height (px)')

plt.tight_layout()
plt.savefig('dataset_analysis.png', dpi=150)
print("Saved dataset_analysis.png")
```

---

## Integration with Research Pipeline

After collecting your dataset, integrate it with the research pipeline:

```bash
# Directory structure
research-project/
├── dataset_collector_api.py      # This file
├── .env                          # Your API keys
├── ai_detection_dataset/         # Collected images & metadata
│   ├── dataset_metadata.csv
│   ├── dataset_metadata.json
│   └── collection_log.json
│
├── analysis/                     # Your analysis scripts
│   ├── visual_analysis.py
│   ├── technical_analysis.py
│   └── results/
│
└── manuscript/                   # Your research writing
    ├── paper.md
    └── figures/
```

---

## Next Steps

1. **Setup** - Install packages, get API keys, configure `.env`
2. **Collect** - Run dataset collection
3. **Verify** - Check dataset_metadata.csv for completeness
4. **Analyze** - Use metadata for your research
5. **Track** - Use CSV in your analysis pipeline (see Research Pipeline document)

---

## API Reference

### DatasetCollector

```python
collector = DatasetCollector(output_dir=Path("./dataset"))

# Methods
collector.load_api_keys()              # Load from .env
collector.collect_real_images(keys)    # Collect real images
collector.collect_ai_images(keys)      # Collect AI images
collector.collect_from_local_folders() # Import local images
collector.generate_dataset_report()    # Get statistics
collector.save_all_metadata()          # Save CSV and JSON
collector.run_full_collection(config)  # Run everything
```

### Individual Collectors

```python
# Real image sources
unsplash = UnsplashCollector(api_key, output_dir)
unsplash.collect(query="portrait", count=50)

pexels = PexelsCollector(api_key, output_dir)
pexels.collect(query="landscape", count=50)

pixabay = PixabayCollector(api_key, output_dir)
pixabay.collect(query="nature", count=50)

# AI images
sd = StableDiffusionCollector(output_dir)
sd.collect_from_dataset("dataset-name", count=100)

# Local folders
local = LocalFolderCollector(output_dir)
local.collect_from_folder(Path("./my_images"), image_type="real")
```

---

## Support & Contributing

For issues or improvements:
1. Check the Troubleshooting section
2. Review the log output for error messages
3. Verify API keys are correct
4. Check internet connection

---

**Document Version:** 1.0
**Last Updated:** May 2026
**Python Version:** 3.8+
