# Example Scripts for Dataset Collector API

## Example 1: Quick Collection (5 minutes)

```python
"""
Minimal example - collect 50 real and 50 AI images quickly
"""
from pathlib import Path
from dataset_collector_api import DatasetCollector

if __name__ == "__main__":
    # Create collector
    collector = DatasetCollector(output_dir=Path("./quick_test_dataset"))
    
    # Minimal config
    config = {
        "real_image_count": 50,
        "ai_image_count": 50,
        "local_folders": {}
    }
    
    # Run
    report = collector.run_full_collection(config)
    
    # Print results
    print("\n" + "="*50)
    print("Collection Complete!")
    print("="*50)
    print(f"Total: {report['total_images']}")
    print(f"Real: {report['real_images']}")
    print(f"AI: {report['ai_generated_images']}")
    print(f"Duplicates: {report['duplicates_found']}")
```

**Run with:**
```bash
python example_1_quick_collection.py
```

---

## Example 2: Full Research Dataset (1 hour)

```python
"""
Complete dataset collection for research
- 250 real images from 3 sources
- 250 AI-generated images
- Balanced categories
"""
from pathlib import Path
from dataset_collector_api import DatasetCollector
import json

if __name__ == "__main__":
    print("="*60)
    print("RESEARCH DATASET COLLECTION")
    print("="*60)
    
    # Create collector
    collector = DatasetCollector(output_dir=Path("./research_dataset"))
    
    # Configuration
    config = {
        "real_image_count": 250,      # Unsplash, Pexels, Pixabay
        "ai_image_count": 250,        # HuggingFace datasets
        "local_folders": {
            # Uncomment if you have local image folders
            # "real": Path("./my_real_images"),
            # "ai_generated": Path("./my_ai_images"),
        }
    }
    
    # Run full collection
    report = collector.run_full_collection(config)
    
    # Print detailed report
    print("\n" + "="*60)
    print("COLLECTION REPORT")
    print("="*60)
    print(f"\nTotal Images: {report['total_images']}")
    print(f"Real Images: {report['real_images']}")
    print(f"AI-Generated Images: {report['ai_generated_images']}")
    print(f"Duplicates Found: {report['duplicates_found']}")
    print(f"Average File Size: {report['average_file_size_kb']:.2f} KB")
    
    print(f"\nBreakdown by Source:")
    for source, count in report['by_source'].items():
        percentage = (count / report['total_images']) * 100
        print(f"  {source:.<40} {count:>4} ({percentage:>5.1f}%)")
    
    print(f"\nDataset Location: ./research_dataset")
    print(f"Metadata Files:")
    print(f"  - dataset_metadata.csv (for spreadsheet analysis)")
    print(f"  - dataset_metadata.json (for programmatic access)")
    print(f"  - collection_log.json (collection statistics)")
```

**Run with:**
```bash
python example_2_full_research_dataset.py
```

---

## Example 3: Collect from Specific Source with Custom Queries

```python
"""
Advanced example - collect from specific sources with custom queries
Demonstrates how to use individual collectors
"""
from pathlib import Path
from dataset_collector_api import UnsplashCollector, PexelsCollector
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    # Load API keys
    load_dotenv()
    unsplash_key = os.getenv("UNSPLASH_API_KEY")
    pexels_key = os.getenv("PEXELS_API_KEY")
    
    output_dir = Path("./custom_collection")
    output_dir.mkdir(exist_ok=True)
    
    if not unsplash_key:
        print("ERROR: UNSPLASH_API_KEY not found in .env")
        exit(1)
    
    # ===== UNSPLASH COLLECTION =====
    print("="*60)
    print("UNSPLASH COLLECTION")
    print("="*60)
    
    unsplash = UnsplashCollector(unsplash_key, output_dir)
    
    # Custom queries for balanced collection
    queries = {
        "portrait professional": 30,      # Professional portraits
        "landscape nature": 30,            # Natural landscapes
        "objects still life": 20,          # Objects and products
        "urban street": 20,                # Urban/architectural
        "food": 20,                        # Food photography
        "animals wildlife": 20,            # Animals
    }
    
    total_unsplash = 0
    for query, count in queries.items():
        print(f"\nCollecting '{query}' ({count} images)...")
        collected = unsplash.collect(query, count=count)
        total_unsplash += collected
    
    # Save Unsplash metadata
    unsplash.save_metadata(output_dir / "unsplash_metadata.csv")
    print(f"\n✓ Unsplash: {total_unsplash} images collected")
    
    # ===== PEXELS COLLECTION =====
    print("\n" + "="*60)
    print("PEXELS COLLECTION")
    print("="*60)
    
    pexels = PexelsCollector(pexels_key, output_dir)
    
    # Different queries for Pexels
    pexel_queries = {
        "person face": 40,
        "mountain landscape": 40,
        "city urban": 30,
        "nature forest": 30,
    }
    
    total_pexels = 0
    for query, count in pexel_queries.items():
        print(f"\nCollecting '{query}' ({count} images)...")
        collected = pexels.collect(query, count=count)
        total_pexels += collected
    
    # Save Pexels metadata
    pexels.save_metadata(output_dir / "pexels_metadata.csv")
    print(f"\n✓ Pexels: {total_pexels} images collected")
    
    # ===== SUMMARY =====
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Unsplash: {total_unsplash}")
    print(f"Total Pexels: {total_pexels}")
    print(f"Grand Total: {total_unsplash + total_pexels}")
    print(f"Saved to: {output_dir}")
```

**Run with:**
```bash
python example_3_custom_queries.py
```

---

## Example 4: Import Local Images and Track Duplicates

```python
"""
Import images from local folders and detect duplicates
"""
from pathlib import Path
from dataset_collector_api import LocalFolderCollector
import pandas as pd

if __name__ == "__main__":
    print("="*60)
    print("LOCAL FOLDER IMPORT")
    print("="*60)
    
    output_dir = Path("./imported_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Define your local folders
    local_folders = {
        "real": Path("./my_real_images"),           # Your real images
        "ai_generated": Path("./my_ai_images"),     # Your AI images
    }
    
    all_metadata = []
    
    # Collect from each folder
    for image_type, folder_path in local_folders.items():
        if not folder_path.exists():
            print(f"⚠ Folder not found: {folder_path}")
            continue
        
        print(f"\nImporting {image_type} images from {folder_path}...")
        
        collector = LocalFolderCollector(output_dir)
        collected = collector.collect_from_folder(
            folder_path=folder_path,
            image_type=image_type
        )
        
        all_metadata.extend(collector.metadata_list)
        print(f"✓ Collected {collected} images")
    
    # Save combined metadata
    if all_metadata:
        df = pd.DataFrame([m.to_dict() for m in all_metadata])
        df.to_csv(output_dir / "imported_metadata.csv", index=False)
        
        # Print statistics
        print("\n" + "="*60)
        print("IMPORT STATISTICS")
        print("="*60)
        print(f"Total images: {len(df)}")
        print(f"\nBy type:")
        print(df['image_type'].value_counts())
        print(f"\nDuplicates found: {df['is_duplicate'].sum()}")
        
        if df['is_duplicate'].sum() > 0:
            print(f"\nDuplicate list:")
            duplicates = df[df['is_duplicate'] == True]
            for _, row in duplicates.iterrows():
                print(f"  {row['image_id']} is duplicate of {row['duplicate_of']}")
```

**Run with:**
```bash
python example_4_local_import.py
```

---

## Example 5: Analyze Collected Dataset

```python
"""
Analyze and visualize your collected dataset
"""
from pathlib import Path
import pandas as pd
import json

if __name__ == "__main__":
    dataset_path = Path("./research_dataset")
    
    # Load metadata
    df = pd.read_csv(dataset_path / "dataset_metadata.csv")
    
    print("="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"  Total images: {len(df)}")
    print(f"  Real images: {len(df[df['image_type'] == 'real'])}")
    print(f"  AI-generated images: {len(df[df['image_type'] == 'ai_generated'])}")
    print(f"  Duplicates: {df['is_duplicate'].sum()}")
    
    # Distribution by source
    print(f"\nDistribution by Source:")
    source_counts = df['source'].value_counts()
    for source, count in source_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {source:.<35} {count:>4} ({pct:>5.1f}%)")
    
    # File size statistics
    print(f"\nFile Size Statistics:")
    print(f"  Average: {df['file_size_kb'].mean():.2f} KB")
    print(f"  Min: {df['file_size_kb'].min():.2f} KB")
    print(f"  Max: {df['file_size_kb'].max():.2f} KB")
    
    # Dimension statistics
    print(f"\nImage Dimensions:")
    print(f"  Average width: {df['width'].mean():.0f} px")
    print(f"  Average height: {df['height'].mean():.0f} px")
    print(f"  Min dimensions: {df['width'].min()}x{df['height'].min()} px")
    print(f"  Max dimensions: {df['width'].max()}x{df['height'].max()} px")
    
    # Format distribution
    print(f"\nImage Formats:")
    format_counts = df['format'].value_counts()
    for fmt, count in format_counts.items():
        print(f"  {fmt}: {count}")
    
    # Color space distribution
    print(f"\nColor Spaces:")
    color_counts = df['color_space'].value_counts()
    for color, count in color_counts.items():
        print(f"  {color}: {count}")
    
    # Export filtered datasets
    print(f"\n" + "="*60)
    print("Creating filtered datasets...")
    
    # Real images only
    real_df = df[df['image_type'] == 'real']
    real_df.to_csv(dataset_path / "real_images_only.csv", index=False)
    print(f"✓ Real images: {len(real_df)} (saved to real_images_only.csv)")
    
    # AI images only
    ai_df = df[df['image_type'] == 'ai_generated']
    ai_df.to_csv(dataset_path / "ai_images_only.csv", index=False)
    print(f"✓ AI images: {len(ai_df)} (saved to ai_images_only.csv)")
    
    # High resolution images (>1920x1080)
    high_res = df[(df['width'] > 1920) | (df['height'] > 1080)]
    high_res.to_csv(dataset_path / "high_resolution_images.csv", index=False)
    print(f"✓ High resolution: {len(high_res)} (saved to high_resolution_images.csv)")
    
    # Load and display collection log
    with open(dataset_path / "collection_log.json") as f:
        log = json.load(f)
    
    print(f"\n" + "="*60)
    print("Collection Summary:")
    print(f"  Started: {log['start_time']}")
    print(f"  Ended: {log['end_time']}")
    print(f"  Collections made: {len(log['collections'])}")
```

**Run with:**
```bash
python example_5_analyze_dataset.py
```

---

## Example 6: Integrate with Research Pipeline

```python
"""
Prepare dataset metadata for use in research analysis pipeline
"""
from pathlib import Path
import pandas as pd
import json

if __name__ == "__main__":
    dataset_path = Path("./research_dataset")
    analysis_path = Path("./analysis")
    analysis_path.mkdir(exist_ok=True)
    
    # Load metadata
    df = pd.read_csv(dataset_path / "dataset_metadata.csv")
    
    print("="*60)
    print("PREPARING DATA FOR ANALYSIS")
    print("="*60)
    
    # Create analysis-ready CSV with required columns for visual analysis
    analysis_df = pd.DataFrame({
        'Image_ID': df['image_id'],
        'Filename': df['filename'],
        'Type': df['image_type'],
        'Source': df['source'],
        'Category': '',  # You'll fill this in during analysis
        'Analyst': '',
        'Analysis_Date': '',
        'Hands_Assessment': '',
        'Text_Clarity': '',
        'Eye_Consistency': '',
        'Physics_Valid': '',
        'Background_Quality': '',
        'Artifact_Score': '',
        'Manual_Classification': '',
        'Confidence_1_10': '',
        'Notes': ''
    })
    
    # Save for manual analysis
    analysis_df.to_csv(analysis_path / "visual_analysis_template.csv", index=False)
    print(f"✓ Created visual_analysis_template.csv")
    
    # Create technical analysis template
    technical_df = pd.DataFrame({
        'Image_ID': df['image_id'],
        'Filename': df['filename'],
        'EXIF_Present': '',
        'FotoForensics_Score': '',
        'Compression_Normal': '',
        'Forensic_Anomalies': '',
        'Reverse_Search_Found': '',
        'TinEye_Results': '',
        'Technical_Notes': ''
    })
    
    technical_df.to_csv(analysis_path / "technical_analysis_template.csv", index=False)
    print(f"✓ Created technical_analysis_template.csv")
    
    # Create tool detection template
    tools_df = pd.DataFrame({
        'Image_ID': df['image_id'],
        'Filename': df['filename'],
        'Tool_1_Name': '',
        'Tool_1_Score': '',
        'Tool_1_Result': '',
        'Tool_2_Name': '',
        'Tool_2_Score': '',
        'Tool_2_Result': '',
        'Consensus_Result': '',
        'Tool_Agreement': ''
    })
    
    tools_df.to_csv(analysis_path / "tool_detection_template.csv", index=False)
    print(f"✓ Created tool_detection_template.csv")
    
    # Create image index file for quick reference
    index_data = {
        'total_images': len(df),
        'real_images': len(df[df['image_type'] == 'real']),
        'ai_images': len(df[df['image_type'] == 'ai_generated']),
        'image_directory': str(dataset_path),
        'metadata_file': str(dataset_path / 'dataset_metadata.csv'),
        'images': []
    }
    
    for _, row in df.iterrows():
        index_data['images'].append({
            'id': row['image_id'],
            'filename': row['filename'],
            'type': row['image_type'],
            'source': row['source'],
            'path': str(dataset_path / ('real_images' if row['image_type'] == 'real' else 'ai_generated_images') / row['filename'])
        })
    
    with open(analysis_path / "image_index.json", 'w') as f:
        json.dump(index_data, f, indent=2)
    print(f"✓ Created image_index.json")
    
    print(f"\n" + "="*60)
    print("Ready for Analysis!")
    print("="*60)
    print(f"Analysis templates created in: {analysis_path}")
    print(f"\nNext steps:")
    print(f"1. Use visual_analysis_template.csv for manual inspection")
    print(f"2. Use technical_analysis_template.csv for forensic analysis")
    print(f"3. Use tool_detection_template.csv for ML detection")
    print(f"4. Reference image_index.json to locate images")
```

**Run with:**
```bash
python example_6_prepare_for_analysis.py
```

---

## Running All Examples

Create a `run_examples.py`:

```python
import subprocess
import sys

examples = [
    ("Quick Collection", "example_1_quick_collection.py"),
    ("Full Research Dataset", "example_2_full_research_dataset.py"),
    ("Custom Queries", "example_3_custom_queries.py"),
    ("Local Import", "example_4_local_import.py"),
    ("Analyze Dataset", "example_5_analyze_dataset.py"),
    ("Prepare for Analysis", "example_6_prepare_for_analysis.py"),
]

print("="*60)
print("DATASET COLLECTOR - EXAMPLE MENU")
print("="*60)

for i, (name, script) in enumerate(examples, 1):
    print(f"{i}. {name}")

choice = input("\nSelect example to run (1-6): ")

try:
    idx = int(choice) - 1
    if 0 <= idx < len(examples):
        script = examples[idx][1]
        print(f"\nRunning: {examples[idx][0]}")
        print("="*60)
        subprocess.run([sys.executable, script])
    else:
        print("Invalid choice")
except ValueError:
    print("Please enter a number")
```

**Run with:**
```bash
python run_examples.py
```

---

**All examples use the same `dataset_collector_api.py` module**
**Copy these files to your project directory and customize as needed**
