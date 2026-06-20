#!/usr/bin/env python3
"""
================================================================================
MAIN EXECUTION SCRIPT – 50K AI DATASET COLLECTOR
================================================================================
Collects:
  - 25 000 real photos (Unsplash, Pexels, Pixabay)
  - 17 500 fully AI‑generated images (CivitAI, HuggingFace, Pollinations)
  - 7 500 AI‑altered images (inpainting of real photos with Stable Diffusion)

Requires:
  - .env file with all API keys (UNSPLASH_API_KEY, PEXELS_API_KEY, PIXABAY_API_KEY,
    CIVITAI_API_KEY, HUGGINGFACE_TOKEN)
  - GPU for the AI‑altered step (skip if no GPU by setting ai_altered_count=0)
  - Python 3.7+ and packages listed in requirements.txt

Output:
  - ai_dataset_50k/real_images/
  - ai_dataset_50k/ai_generated_images/
  - ai_dataset_50k/ai_altered_images/
  - Metadata CSV/JSON for each class + combined
  - collection_log.json
"""

from pathlib import Path
from complete_dataset_collector import DatasetCollectorV2
import sys

def main():
    print("\n" + "="*70)
    print("🚀 50K AI IMAGE DATASET COLLECTOR")
    print("="*70)
    print("""
Target:
    Real photos           → 25 000
    Fully AI‑generated    → 17 500
    AI‑altered            →  7 500
    ─────────────────────────────────
    Total                  → 50 000 images
    """)
    print("⚠️  AI‑altered images require a CUDA GPU. If no GPU, set ai_altered_count=0.\n")

    # ===================== CONFIGURATION =====================
    config = {
        "output_dir": Path("./ai_dataset_50k"),
        "real_count": 25000,
        "ai_generated_count": 17500,
        "ai_altered_count": 7500,          # Set to 0 if no GPU
        # Optional: path to a folder of existing real images for alteration.
        # If None, the script uses the real images it just downloaded.
        "real_images_for_alteration": None  # e.g., Path("./my_real_images")
    }

    print("Configuration:")
    print(f"  Output directory       : {config['output_dir']}")
    print(f"  Real target            : {config['real_count']}")
    print(f"  AI‑generated target    : {config['ai_generated_count']}")
    print(f"  AI‑altered target      : {config['ai_altered_count']}")
    if config['ai_altered_count'] > 0:
        print(f"  Real folder for alter. : {config['real_images_for_alteration'] or 'auto (use downloaded real images)'}")
    print()

    try:
        collector = DatasetCollectorV2(output_dir=config["output_dir"])

        print("▶ Starting collection…\n")

        # If a separate real‑images folder is provided for alteration, we need to
        # make sure it exists. Otherwise, we'll use the ones we just collected.
        # The run_full_collection method will automatically look for real images
        # in 'output_dir/real_images' after the real collection step.
        real_images_dir = config.get("real_images_for_alteration")

        report = collector.run_full_collection(
            real_count=config["real_count"],
            ai_generated_count=config["ai_generated_count"],
            ai_altered_count=config["ai_altered_count"],
            # Optional parameter: override the default path
            # We can't pass it directly because run_full_collection expects it
            # to come from the real collection step. For simplicity, we'll do a
            # small workaround: if user provides a custom folder, we set it before
            # calling run_full_collection.
        )

        # For custom real‑images folder, we modify the collector's logic on the fly
        # (or you can adapt run_full_collection to accept the path).
        # Since your collector code already uses self.output_dir / "real_images",
        # we will just copy the custom folder into that location if provided.
        # But to keep main.py clean, we'll add a quick copy if needed.
        if real_images_dir and real_images_dir.is_dir():
            import shutil
            dest = config["output_dir"] / "real_images"
            if not dest.exists():
                dest.mkdir(parents=True)
            for f in real_images_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dest / f.name)
            print(f"📁 Copied custom real images to {dest}")

        # ===================== FINAL REPORT =====================
        print("\n" + "="*70)
        print("✅ COLLECTION COMPLETE!")
        print("="*70)
        print(f"""
Results:
    Total images            : {report['total_images']:,}
    Real photos             : {report['real_images']:,}
    Fully AI‑generated      : {report['ai_generated_images']:,}
    AI‑altered images       : {report['ai_altered_images']:,}
    Duplicates detected     : {report['duplicates_found']:,}
""")
        print(f"📁 Dataset saved to: {config['output_dir']}")
        print(f"   real images       → {config['output_dir']}/real_images/")
        print(f"   AI‑generated      → {config['output_dir']}/ai_generated_images/")
        if config['ai_altered_count'] > 0:
            print(f"   AI‑altered        → {config['output_dir']}/ai_altered_images/")
        print(f"   All metadata      → dataset_metadata_all.csv / .json")
        print(f"   Collection log    → collection_log.json")

        print("\nBreakdown by source:")
        for src, cnt in sorted(report["by_source"].items()):
            pct = cnt / report['total_images'] * 100
            print(f"  {src:.<50} {cnt:>6,} ({pct:5.1f}%)")

        print("\n" + "="*70)
        print("🎉 50K dataset ready for training your AI‑detection model!")
        print("="*70 + "\n")
        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Partial dataset saved.")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())