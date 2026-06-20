# Complete Dataset Collector - Setup & Execution Guide

## 📦 What You Have

**4 Complete Files Ready to Use:**

1. **complete_dataset_collector.py** - Main collector code (2000+ lines)
2. **main.py** - Run this to start collection
3. **requirements.txt** - Python dependencies
4. **.env.example** - API key template

---

## ⚡ Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Setup API Keys

Copy your API keys into `.env`:

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your keys:
UNSPLASH_API_KEY=your_unsplash_key_here
PEXELS_API_KEY=your_pexels_key_here
PIXABAY_API_KEY=your_pixabay_key_here
```

**Get free API keys:**
- Unsplash: https://unsplash.com/oauth/applications
- Pexels: https://www.pexels.com/api/
- Pixabay: https://pixabay.com/api/docs/

### Step 3: Run Collection

```bash
python main.py
```

**That's it!** Collection starts automatically.

---

## 📊 What You'll Collect

```
Target: 1000+ Real + 1000+ AI = 2000+ Total Images

Real Images (1000):
  - Unsplash:  ~333 images (API required)
  - Pexels:    ~333 images (API required)
  - Pixabay:   ~334 images (API required)

AI Images (1000):
  - Lexica.art:  ~250 images (NO API needed)
  - OpenArt.ai:  ~250 images (NO API needed)
  - CivitAI:     ~250 images (NO API needed)
  - Local:       ~250 images (if you provide)
```

---

## ⏱️ Timeline

```
Setup:           5 minutes
Collection:      90-120 minutes
Total:           ~2 hours
```

---

## 📁 Output Structure

After running, you'll have:

```
ai_dataset/
├── real_images/              (1000+ photos)
│   ├── UNSPLASH_*.jpg        (333 images)
│   ├── PEXELS_*.jpg          (333 images)
│   └── PIXABAY_*.jpg         (334 images)
│
├── ai_generated_images/      (1000+ AI images)
│   ├── LEXICA_*.jpg          (250 images)
│   ├── OPENART_*.jpg         (250 images)
│   ├── CIVITAI_*.jpg         (250 images)
│   └── LOCAL_*.jpg           (if you provided)
│
├── dataset_metadata.csv      (All metadata in table)
├── dataset_metadata.json     (All metadata in JSON)
└── collection_log.json       (Statistics & report)
```

---

## 🚀 Running the Collector

### Basic Run (Default: 1000 real + 1000 AI)

```bash
python main.py
```

### Custom Counts (Edit main.py)

Change the config in main.py:

```python
config = {
    "output_dir": Path("./ai_dataset"),
    "real_count": 500,      # Change this
    "ai_count": 500,        # Change this
    "local_folders": None
}
```

Then run:
```bash
python main.py
```

### With Local Folders (Your Own Images)

```python
config = {
    "output_dir": Path("./ai_dataset"),
    "real_count": 1000,
    "ai_count": 1000,
    "local_folders": {
        "real": Path("./my_real_images"),
        "ai_generated": Path("./my_ai_images"),
    }
}
```

Then run:
```bash
python main.py
```

---

## 📊 Monitoring Progress

The collector shows real-time progress:

```
=======================================================================
COLLECTING REAL IMAGES (Target: 1000)
=======================================================================

→ Unsplash Collection
  Collecting 333 from Unsplash: 'portrait professional'
  100%|██████████| 30/30 [00:45<00:00,  1.50s/it]
  ✓ Collected 30/333: UNSPLASH_abc123
  ...
  ✓ Unsplash: 333 images

→ Pexels Collection
  ...
  ✓ Pexels: 333 images

→ Pixabay Collection
  ...
  ✓ Pixabay: 334 images

✓ Real images collected: 1000

=======================================================================
COLLECTING AI-GENERATED IMAGES (Target: 1000)
=======================================================================

→ Lexica.art Collection
  ...
  ✓ Lexica: 247 images

→ OpenArt.ai Collection
  ...
  ✓ OpenArt: 251 images

→ CivitAI Collection
  ...
  ✓ CivitAI: 252 images

✓ AI images collected: 750

=======================================================================
DATASET COLLECTION COMPLETE
=======================================================================

Total Images: 1750
  Real:        1000
  AI:          750
  Duplicates:  1

Breakdown by Source:
  Unsplash..................................... 333 ( 19.0%)
  Pexels...................................... 333 ( 19.0%)
  Pixabay..................................... 334 ( 19.1%)
  Lexica.art.................................. 247 ( 14.1%)
  OpenArt.ai.................................. 251 ( 14.3%)
  CivitAI..................................... 252 ( 14.4%)

Dataset Location: ./ai_dataset

Saving metadata files...

✓ CSV: ./ai_dataset/dataset_metadata.csv (1750 records)
✓ JSON: ./ai_dataset/dataset_metadata.json
✓ Log: ./ai_dataset/collection_log.json

=======================================================================
SUCCESS! Dataset collection complete!
=======================================================================
```

---

## 📝 Metadata Included

Each image has complete metadata:

```json
{
  "image_id": "UNSPLASH_abc123",
  "filename": "UNSPLASH_abc123.jpg",
  "source": "unsplash",
  "source_url": "https://unsplash.com/photos/...",
  "download_date": "2024-05-26T15:30:45.123456",
  "image_type": "real",
  "width": 4000,
  "height": 2667,
  "format": "JPEG",
  "file_size_kb": 245.5,
  "color_space": "RGB",
  "source_metadata": {
    "author": "Photographer Name"
  },
  "md5_hash": "a1b2c3d4e5f6...",
  "is_duplicate": false,
  "duplicate_of": null,
  "processed": false,
  "analysis_notes": "Query: portrait professional"
}
```

---

## ✅ Verification After Collection

Check if everything worked:

```bash
# Count images
ls ai_dataset/real_images/ | wc -l           # Should be ~1000
ls ai_dataset/ai_generated_images/ | wc -l   # Should be ~1000

# Check metadata
head ai_dataset/dataset_metadata.csv          # Preview data
wc -l ai_dataset/dataset_metadata.csv         # Count rows

# View statistics
cat ai_dataset/collection_log.json | python -m json.tool
```

---

## 🔧 Troubleshooting

### Issue: "API key not found"

**Solution:** Check your .env file

```bash
cat .env | grep API_KEY
```

Make sure all 3 keys are there and not empty.

### Issue: "No AI images collected"

**Solution:** This is normal - AI sources are slower. They still collect without API keys.

The system will collect whatever it can from each source.

### Issue: "Too slow, taking too long"

**Solution:** 
- Reduce counts in main.py
- Run during off-peak hours
- Skip local folder import if you don't need it

### Issue: "Connection timeout"

**Solution:** Increase timeout in code

In `complete_dataset_collector.py`, change:
```python
timeout=10  # Change to timeout=30
```

### Issue: "Out of memory or disk space"

**Check disk space:**
```bash
df -h    # See available space (need at least 5GB)
```

**If low on space, reduce target counts:**
```python
config = {
    "real_count": 500,    # Reduced from 1000
    "ai_count": 500,      # Reduced from 1000
}
```

---

## 📚 Using the Collected Dataset

### Load in Python

```python
import pandas as pd

# Load metadata
df = pd.read_csv("ai_dataset/dataset_metadata.csv")

# Get statistics
print(f"Total images: {len(df)}")
print(f"Real: {len(df[df['image_type'] == 'real'])}")
print(f"AI: {len(df[df['image_type'] == 'ai_generated'])}")

# Filter
real_images = df[df['image_type'] == 'real']
ai_images = df[df['image_type'] == 'ai_generated']

# Export specific subset
high_res = df[(df['width'] > 1920) | (df['height'] > 1080)]
high_res.to_csv("high_resolution_images.csv", index=False)
```

### Image File Locations

```python
from pathlib import Path

# Real images
real_dir = Path("ai_dataset/real_images")
real_files = list(real_dir.glob("*.jpg"))

# AI images
ai_dir = Path("ai_dataset/ai_generated_images")
ai_files = list(ai_dir.glob("*.jpg")) + list(ai_dir.glob("*.png"))

# Load with PIL
from PIL import Image

img = Image.open(real_files[0])
img.show()
```

---

## 🎯 Next Steps After Collection

1. **Analyze metadata:**
   ```bash
   python analyze_dataset.py  # (You can create this)
   ```

2. **Use for training:**
   - Visual analysis (manual inspection)
   - Technical analysis (forensics)
   - ML detection testing

3. **Integrate with pipeline:**
   - Use metadata in Data_Tracking_Templates.md
   - Begin your research analysis
   - Track findings in spreadsheets

---

## 📞 Common Questions

**Q: Do I need all 3 real image API keys?**
A: No, use whatever you have. The system continues with available APIs.

**Q: Can I resume an interrupted collection?**
A: Not automatically, but you can add more images manually by copying files into the output folders.

**Q: How do I add my own images?**
A: Create folders and set local_folders in config.

**Q: Can I change image quality/size?**
A: The APIs provide various sizes. Current code uses largest available.

**Q: How often should I run this?**
A: Once for dataset collection, then once per research phase if you want to add more images.

---

## 🎉 You're Ready!

Everything is configured and ready to go. Just run:

```bash
python main.py
```

And wait 1.5-2 hours for your 2000-image dataset to be collected!

---

## 📋 Checklist Before Running

- [ ] Python 3.7+ installed
- [ ] `pip install -r requirements.txt` completed
- [ ] `.env` file created with 3 API keys
- [ ] At least 5GB free disk space
- [ ] Internet connection stable
- [ ] `main.py` file in current directory
- [ ] `complete_dataset_collector.py` in current directory

✓ All set? Run: `python main.py`

---

**Questions? Check the troubleshooting section above.**

**Ready to start? Type: `python main.py` and press Enter! 🚀**
