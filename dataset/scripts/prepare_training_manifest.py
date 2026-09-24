"""Build the full-scale training manifest from clean_metadata.csv,
correcting the two dataset confounds identified in the review:

1. Places365 dominance: real class is 94% Places365 scenes while fakes are
   ImageNet-content, letting a detector learn content instead of
   authenticity. --cap-places365 limits its contribution (default 400K).

2. Faces only in the fake class: Celeb-DF/FF++/DFDC currently contribute
   only FAKE frames, so "close-up face => fake" is a winning shortcut.
   This script scans the deepfake dataset folders for real/original frame
   directories and adds them to the real class; if none are found it
   prints a loud warning telling you to extract them before training.

Also deduplicates by MD5 and writes dataset/metadata/train_manifest.csv.

Run from the repo root:
    python dataset/scripts/prepare_training_manifest.py
    python dataset/scripts/prepare_training_manifest.py --cap-places365 400000
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
IMAGES = REPO / "dataset" / "images"
METADATA = REPO / "dataset" / "metadata"

# directory names that hold REAL/original frames inside deepfake datasets
DEEPFAKE_REAL_HINTS = {
    "FaceForensics": ["original_sequences", "original", "real"],
    "DFDC": ["real", "REAL", "original"],
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def find_deepfake_real_frames() -> pd.DataFrame:
    rows = []
    for ds_dir, hints in DEEPFAKE_REAL_HINTS.items():
        base = IMAGES / ds_dir
        if not base.exists():
            continue
        for hint in hints:
            for d in base.rglob(hint):
                if not d.is_dir():
                    continue
                for p in d.rglob("*"):
                    if p.suffix.lower() in IMAGE_EXTS and p.stat().st_size > 0:
                        rows.append({
                            "image_id": p.stem,
                            "filename": str(p.relative_to(IMAGES)),
                            "label": "real",
                            "source": f"{ds_dir.lower()}_real",
                            "generator": "",
                        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(METADATA / "clean_metadata.csv"))
    ap.add_argument("--out", default=str(METADATA / "train_manifest.csv"))
    ap.add_argument("--cap-places365", type=int, default=400_000,
                    help="max Places365 real images to keep (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {args.manifest} ...")
    df = pd.read_csv(args.manifest, dtype=str, low_memory=False)
    print(f"  {len(df):,} rows")

    # --- 0. drop the duplicated BigGAN source ---
    # The same GenImage BigGAN images exist on disk twice: source 'biggan'
    # (BigGAN/imagenet_ai_0419_biggan/...) and 'genimage_biggan'
    # (genimage_ai/BigGAN/...). Neither copy carries an MD5, so the MD5
    # dedupe below cannot catch them; matched by basename instead.
    if {"biggan", "genimage_biggan"} <= set(df["source"].dropna().unique()):
        gi_names = set(
            df.loc[df["source"] == "genimage_biggan", "filename"]
            .str.replace("\\", "/").str.rsplit("/", n=1).str[-1])
        bg = df[df["source"] == "biggan"]
        dup_mask = (bg["filename"].str.replace("\\", "/")
                    .str.rsplit("/", n=1).str[-1].isin(gi_names))
        df = df.drop(index=bg[dup_mask].index)
        print(f"  Dropped {int(dup_mask.sum()):,} 'biggan' rows duplicated "
              f"in 'genimage_biggan'")

    # --- 0.5. drop strictly held-out datasets ---
    # To prove zero-shot cross-domain generalization, these datasets must NEVER be seen in training
    held_out_sources = ["dalle3", "celebdf"]
    
    # Also drop any genimage sources (e.g. genimage_biggan, genimage_stylegan, etc)
    drop_mask = df["source"].isin(held_out_sources) | df["source"].str.startswith("genimage_", na=False)
    
    if drop_mask.any():
        df = df[~drop_mask]
        print(f"  Dropped {int(drop_mask.sum()):,} rows from strictly held-out datasets (DALL-E 3, CelebDF, GenImage)")

    # --- 1. cap Places365 ---
    if args.cap_places365 > 0:
        places = df[df["source"] == "places365"]
        if len(places) > args.cap_places365:
            keep = places.sample(args.cap_places365, random_state=args.seed)
            df = pd.concat([df[df["source"] != "places365"], keep],
                           ignore_index=True)
            print(f"  Capped places365: {len(places):,} -> {args.cap_places365:,}")

    # --- 2. add deepfake real frames ---
    real_frames = find_deepfake_real_frames()
    if len(real_frames) == 0:
        print("\n" + "!" * 72)
        print("! WARNING: no real/original frames found under the deepfake")
        print("! dataset folders (CelebDF_V2, FaceForensics, DFDC).")
        print("! Without them, faces exist almost exclusively in the FAKE class")
        print("! and the model can learn 'face => fake'. Extract the real")
        print("! frames from those datasets before full-scale training.")
        print("!" * 72 + "\n")
    else:
        before = len(df)
        df = pd.concat([df, real_frames], ignore_index=True)
        print(f"  Added {len(df) - before:,} deepfake-dataset REAL frames "
              f"({real_frames['source'].value_counts().to_dict()})")

    # --- 3. dedupe by md5 where available ---
    if "md5" in df.columns:
        with_md5 = df[df["md5"].notna() & (df["md5"] != "")]
        dupes = with_md5.duplicated(subset="md5", keep="first")
        n_dupes = int(dupes.sum())
        if n_dupes:
            df = df.drop(index=with_md5[dupes].index)
            print(f"  Removed {n_dupes:,} MD5 duplicates")

    # --- summary + write ---
    print("\nFinal composition:")
    print(df["label"].value_counts().to_string())
    print("\nBy source:")
    print(df.groupby(["label", "source"]).size().to_string())

    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df):,} rows to {args.out}")
    print("Point config.py metadata_paths at train_manifest.csv for the "
          "full-scale run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
