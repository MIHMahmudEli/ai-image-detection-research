"""Add missing augmented real images to all.csv metadata."""
from pathlib import Path
import hashlib, csv
from datetime import datetime
import pandas as pd

base = Path(r'C:\Users\ROWTECH\Desktop\ReSearch\ai-image-detection-research\dataset')
real_dir = base / 'images' / 'real'
meta_csv = base / 'metadata' / 'all.csv'

df = pd.read_csv(meta_csv)
existing_in_csv = set(df[df['image_type'] == 'real']['filename'])

all_files = set(f.name for f in real_dir.glob('*') if f.suffix in ('.jpg', '.jpeg', '.png'))
missing = all_files - existing_in_csv

print(f'Files in real/ directory: {len(all_files)}')
print(f'Real entries in all.csv:  {len(existing_in_csv)}')
print(f'Missing entries to add:   {len(missing)}')

if not missing:
    print('Nothing to do.')
    exit()

now = datetime.now().isoformat()
new_rows = []
for fname in sorted(missing):
    fpath = real_dir / fname
    data = fpath.read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    new_rows.append({
        'image_id': fpath.stem,
        'filename': fname,
        'source': 'augmented',
        'source_url': '',
        'download_date': now,
        'image_type': 'real',
        'width': 0, 'height': 0, 'format': '',
        'file_size_kb': round(len(data)/1024, 2),
        'color_space': '',
        'md5_hash': md5,
        'is_duplicate': False,
        'duplicate_of': '',
    })

df_new = pd.DataFrame(new_rows)
df_all = pd.concat([df, df_new], ignore_index=True)
df_all.to_csv(meta_csv, index=False)

real_final = len(df_all[df_all['image_type'] == 'real'])
print(f'\nUpdated real entries in all.csv: {real_final}')
print(f'Total entries in all.csv:  {len(df_all)}')
