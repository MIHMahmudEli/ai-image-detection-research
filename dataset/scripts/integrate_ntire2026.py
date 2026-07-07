import pandas as pd
import os
import time

BASE = "dataset/images"
NTIRE = "NTIRE2026"
CSV_PATH = "dataset/metadata/clean_metadata.csv"

# Load existing
existing = pd.read_csv(CSV_PATH, low_memory=False)
last_id = len(existing)
print(f"Existing: {len(existing):,} rows, last_id={last_id}")

# Build NTIRE dataframe from labels
shard_dfs = []
for i in range(6):
    ldf = pd.read_csv(f"{BASE}/{NTIRE}/shard_{i}/labels.csv", index_col=0)
    ldf["shard"] = i
    shard_dfs.append(ldf)

ntire = pd.concat(shard_dfs, ignore_index=True)
print(f"NTIRE: {len(ntire):,} rows")

# Vectorized column building
start = last_id + 1
ntire["image_id"] = [f"ntire2026_{start + j:08d}" for j in range(len(ntire))]
ntire["filename"] = ntire.apply(lambda r: f"{NTIRE}\\shard_{r['shard']}\\images\\{r['image_name']}", axis=1)
ntire["label"] = ntire["label"].map({0: "real", 1: "ai_generated"})
ntire["source"] = "ntire2026"
ntire["generator"] = "unknown"
ntire["width"] = 0
ntire["height"] = 0
ntire["md5"] = None

# Fast file size lookup
t0 = time.time()
ntire["file_size_bytes"] = ntire["filename"].apply(lambda f: os.path.getsize(f"{BASE}\\{f}") if os.path.isfile(f"{BASE}\\{f}") else 0)
print(f"File sizes: {time.time()-t0:.1f}s")

ntire = ntire[["image_id", "filename", "label", "source", "generator", "width", "height", "file_size_bytes", "md5"]]

# Merge and save
combined = pd.concat([existing, ntire], ignore_index=True)
combined.to_csv(CSV_PATH, index=False)
print(f"\nSaved: {len(combined):,} rows total")
for label, grp in combined.groupby("label"):
    print(f"  {label}: {len(grp):,}")
