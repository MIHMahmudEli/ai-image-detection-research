import pandas as pd

df = pd.read_csv("dataset/metadata/clean_metadata.csv", low_memory=False)
print(f"Total: {len(df):,} rows")
print(f"Sources: {sorted(df['source'].unique())}")
print()
for src, grp in df.groupby("source"):
    labels = dict(grp["label"].value_counts())
    print(f"  {src}: {len(grp):>8,} — {labels}")

print()
total_real = len(df[df["label"] == "real"])
total_ai = len(df[df["label"] == "ai_generated"])
total_df = len(df[df["label"] == "deepfake"])
print(f"real={total_real:,}  ai_generated={total_ai:,}  deepfake={total_df:,}")
print(f"Total: {total_real+total_ai+total_df:,}")
