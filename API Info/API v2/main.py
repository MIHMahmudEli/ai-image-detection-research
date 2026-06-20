#!/usr/bin/env python3
"""
================================================================================
MAIN EXECUTION SCRIPT - Run This to Collect 1000+ Real + 1000+ AI Images
================================================================================

Usage:
    python main.py

Requirements:
    - .env file with API keys (UNSPLASH_API_KEY, PEXELS_API_KEY, PIXABAY_API_KEY)
    - Python 3.7+
    - See requirements.txt for packages

Output:
    - ai_dataset/real_images/ (1000+ real photos)
    - ai_dataset/ai_generated_images/ (1000+ AI images)
    - ai_dataset/dataset_metadata.csv (all metadata in table format)
    - ai_dataset/dataset_metadata.json (all metadata in JSON format)
    - ai_dataset/collection_log.json (collection statistics)

Time:
    - ~1.5-2 hours to collect 2000 images
"""

from pathlib import Path
from complete_dataset_collector import DatasetCollectorV2
import sys

def main():
    """Main entry point"""
    
    print("\n" + "="*70)
    print("AI IMAGE DATASET COLLECTOR V2")
    print("="*70)
    print("\nTarget: 1000+ Real Images + 1000+ AI-Generated Images")
    print("Total Dataset: 2000+ Images")
    print("Time Estimate: 1.5-2 hours")
    print("\n" + "="*70 + "\n")
    
    # Configuration
    config = {
        "output_dir": Path("./ai_dataset"),
        "real_count": 1000,
        "ai_count": 1000,
        "local_folders": None  # Set to {"real": Path("./my_real"), "ai_generated": Path("./my_ai")} if you have local folders
    }
    
    print(f"Configuration:")
    print(f"  Output Directory: {config['output_dir']}")
    print(f"  Real Images Target: {config['real_count']}")
    print(f"  AI Images Target: {config['ai_count']}")
    print(f"  Total Target: {config['real_count'] + config['ai_count']}")
    print()
    
    try:
        # Create collector
        collector = DatasetCollectorV2(output_dir=config["output_dir"])
        
        # Run collection
        print("Starting collection...\n")
        report = collector.run_full_collection(
            real_count=config["real_count"],
            ai_count=config["ai_count"],
            local_folders=config["local_folders"]
        )
        
        # Print summary
        print("="*70)
        print("COLLECTION COMPLETE!")
        print("="*70)
        print(f"\nResults:")
        print(f"  Total Images: {report['total_images']}")
        print(f"  Real Images: {report['real_images']}")
        print(f"  AI-Generated Images: {report['ai_generated_images']}")
        print(f"  Duplicates Found: {report['duplicates_found']}")
        
        print(f"\nDataset Location: {config['output_dir']}")
        print(f"  - Real images: {config['output_dir']}/real_images/")
        print(f"  - AI images: {config['output_dir']}/ai_generated_images/")
        print(f"  - Metadata CSV: {config['output_dir']}/dataset_metadata.csv")
        print(f"  - Metadata JSON: {config['output_dir']}/dataset_metadata.json")
        print(f"  - Collection Log: {config['output_dir']}/collection_log.json")
        
        print(f"\nBreakdown by Source:")
        for source in sorted(report["by_source"].keys()):
            count = report["by_source"][source]
            pct = (count / report['total_images']) * 100
            print(f"  {source:.<50} {count:>5} ({pct:>5.1f}%)")
        
        print("\n" + "="*70)
        print("✓ Dataset ready for analysis!")
        print("="*70 + "\n")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
