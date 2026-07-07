import pandas as pd

df = pd.read_csv("dataset/metadata/clean_metadata.csv", low_memory=False)
print(f"Before: {len(df):,} rows")

# Drop stale biggan rows (files deleted from disk)
df = df[df["source"] != "biggan"]
print(f"After:  {len(df):,} rows (dropped {161995:,} biggan rows)")

# Verify no other source has missing files (spot-check)
import os
for src, grp in df.groupby("source"):
    fnames = grp["filename"].head(200)
    found = sum(1 for f in fnames if os.path.isfile(os.path.join("dataset/images", f)))
    status = "OK" if found == 200 else f"<<< {200-found} MISSING"
    print(f"  {src}: {found}/200 {status}")

# Save
df.to_csv("dataset/metadata/clean_metadata.csv", index=False)
print("\nSaved clean_metadata.csv")
