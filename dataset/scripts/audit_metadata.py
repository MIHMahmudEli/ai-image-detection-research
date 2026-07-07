import pandas as pd
import os

df = pd.read_csv("dataset/metadata/clean_metadata.csv", low_memory=False)
print(f"Total rows: {len(df):,}\n")

print("=== Source breakdown ===")
for src, grp in df.groupby("source"):
    labels = grp["label"].value_counts().to_dict()
    print(f"  {src}: {len(grp):>8,} rows — {labels}")

print()

# Check BigGAN sources specifically
print("=== BigGAN file existence check ===")
for src_name in ["biggan", "genimage_biggan"]:
    rows = df[df["source"] == src_name]
    sample = rows.head(5)["filename"].tolist()
    print(f"  source='{src_name}' — {len(rows):,} rows")
    print(f"    sample paths: {sample[:3]}")
    # Check first 100
    found = sum(1 for f in rows.head(100)["filename"] if os.path.isfile(os.path.join("dataset/images", f)))
    print(f"    first 100 on disk: {found}/100")

print()

# Check all unique sources for missing files (sample 200 each)
print("=== Spot-check: first 200 files per source ===")
for src, grp in df.groupby("source"):
    fnames = grp["filename"].head(200)
    found = sum(1 for f in fnames if os.path.isfile(os.path.join("dataset/images", f)))
    if found < 200:
        print(f"  {src}: {found}/200 exist on disk  <<< MISSING FILES")
    else:
        print(f"  {src}: {found}/200 exist on disk  OK")
