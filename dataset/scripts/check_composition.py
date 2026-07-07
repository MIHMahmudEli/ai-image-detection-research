import pandas as pd
import os

df = pd.read_csv("dataset/metadata/clean_metadata.csv", low_memory=False)
print("=== CURRENT DATASET (clean_metadata.csv) ===")
print(f"Total rows: {len(df):,}")
for label, grp in df.groupby("label"):
    print(f"  {label}: {len(grp):,}")

print()
print("=== NTIRE2026 ===")
total_real = 0
total_ai = 0
for i in range(6):
    ldf = pd.read_csv(f"dataset/images/NTIRE2026/shard_{i}/labels.csv", index_col=0)
    real = (ldf["label"]==0).sum()
    fake = (ldf["label"]==1).sum()
    total_real += real
    total_ai += fake
    print(f"  shard_{i}: real={real:,}, ai={fake:,}")

print(f"  Total NTIRE2026: real={total_real:,}, ai={total_ai:,}")

curr_real = sum(len(grp) for lb, grp in df.groupby("label") if "real" in lb.lower())
curr_ai = sum(len(grp) for lb, grp in df.groupby("label") if "real" not in lb.lower())

print()
print("=== COMBINED TOTALS ===")
print(f"  Real images:     {curr_real + total_real:>10,}")
print(f"  AI-generated:    {curr_ai + total_ai:>10,}")
print(f"  Total:           {curr_real + total_real + curr_ai + total_ai:>10,}")
ratio = (curr_real + total_real) / (curr_real + total_real + curr_ai + total_ai)
print(f"  Ratio:           {ratio:.1%} real / {1-ratio:.1%} ai")
