"""
Step 2: Remove exact duplicates (by MD5), keeping the first occurrence.
Keeps a log of removed files and prints final counts.
Run: python dataset/scripts/02_deduplicate.py
"""
import csv
from pathlib import Path

METADATA = Path("dataset/metadata/clean_metadata.csv")
OUTPUT_DIR = Path("dataset/scripts/removed_duplicates.txt")

import pandas as pd
df = pd.read_csv(METADATA)
before = len(df)

# Track which md5 we've seen, remove duplicates keeping first
seen = set()
keep = []
remove_log = []
for _, row in df.iterrows():
    md5 = row["md5"]
    if md5 in seen:
        remove_log.append(row["filename"])
    else:
        seen.add(md5)
        keep.append(row)

# Write deduplicated metadata
deduped = pd.DataFrame(keep)
deduped.to_csv(METADATA, index=False)

# Write list of removed files
Path("dataset/scripts/removed_duplicates.txt").write_text("\n".join(remove_log))

print(f"Before: {before}")
print(f"After:  {len(keep)}")
print(f"Removed: {len(remove_log)} duplicate files")
print(f"\nUpdated: {METADATA}")
print(f"Removed file list: dataset/scripts/removed_duplicates.txt")
