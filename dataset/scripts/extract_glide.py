import pyarrow.parquet as pq
import time, os
from pathlib import Path

DL_DIR = Path('genimage_zips/hf_shards/glide')
OUT_DIR = Path('dataset/images/genimage_ai/glide')
OUT_DIR.mkdir(parents=True, exist_ok=True)

shards = sorted([f for f in os.listdir(DL_DIR) if f.endswith('.parquet')])
print(f'Extracting {len(shards)} glide shards to {OUT_DIR}')

total_extracted = 0
t_start = time.time()

for si, shard_name in enumerate(shards):
    shard_path = DL_DIR / shard_name
    pf = pq.ParquetFile(shard_path)
    total_rows = pf.metadata.num_rows
    shard_extracted = 0
    t0 = time.time()

    print(f'[{si+1}/{len(shards)}] {shard_name}: {total_rows} rows...')

    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx)
        struct_arr = table.column('image').combine_chunks()

        for i in range(len(struct_arr)):
            row = struct_arr[i].as_py()
            img_bytes = row['bytes']
            rel_path = row['path']

            out_path = OUT_DIR / rel_path
            if out_path.exists():
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(img_bytes)
            shard_extracted += 1

    elapsed = time.time() - t0
    print(f'  Extracted {shard_extracted} images in {elapsed:.0f}s ({shard_extracted/elapsed:.0f} img/s)')
    total_extracted += shard_extracted

elapsed = time.time() - t_start
print(f'\nTotal: {total_extracted} images extracted in {elapsed:.0f}s')
