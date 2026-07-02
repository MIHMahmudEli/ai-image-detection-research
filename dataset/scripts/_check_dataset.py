from pathlib import Path
import pandas as pd, time
from collections import Counter

base = Path('dataset/images')
print('=== DISK COUNTS (recursive, image files only) ===')
total = 0
for d in sorted(base.iterdir()):
    if d.is_dir():
        count = sum(1 for _ in d.rglob('*') if _.suffix.lower() in ('.png','.jpg','.jpeg','.webp'))
        print(f'  {d.name}: {count:,}')
        total += count
print(f'  TOTAL: {total:,}')

print('\n=== METADATA REPORT ===')
meta = pd.read_csv('dataset/metadata/clean_metadata.csv')
print(f'  CSV rows: {len(meta):,}')

print('\n  By label:')
print(meta['label'].value_counts().to_string())

print('\n  By source:')
print(meta['source'].value_counts().to_string())

print('\n  By generator:')
g_counts = meta['generator'].value_counts()
print(g_counts.to_string())

print('\n=== IMAGE PROPERTIES ===')
total_bytes = meta['file_size_bytes'].sum()
print(f'  File size total: {total_bytes/1e9:.1f} GB')
print(f'  Avg file size: {total_bytes/len(meta)/1024:.1f} KB')

print('\n=== FILE EXTENSIONS (from disk) ===')
exts = Counter()
for d in sorted(base.iterdir()):
    if d.is_dir():
        for p in d.rglob('*'):
            if p.is_file():
                exts[p.suffix.lower()] += 1
for ext, cnt in exts.most_common():
    print(f'  {ext}: {cnt:,}')

print('\n=== QUALITY CHECK ===')
t0 = time.time()
sample = meta.sample(min(2000, len(meta)))
missing = 0
for _, row in sample.iterrows():
    p = base / row['filename']
    if not p.exists():
        missing += 1
print(f'  Random 2000 checked: {missing} missing ({time.time()-t0:.1f}s)')
print('  All references exist: OK' if missing == 0 else f'  WARNING: {missing} references missing!')
