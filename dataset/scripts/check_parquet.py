import pyarrow.parquet as pq
from PIL import Image
import io, sys, os

f = 'genimage_zips/bitmind_biggan_shard_00000.parquet'
pf = pq.ParquetFile(f)

total_rows = pf.metadata.num_rows
print(f"Total rows: {total_rows}")
print(f"Num row groups: {pf.metadata.num_row_groups}")

schema = pf.schema_arrow
print(f"Schema: {schema}")
for i in range(len(schema)):
    field = schema.field(i)
    print(f"  Field {i}: {field.name}: {field.type}")

table = pf.read_row_group(0)
print(f"\nRow group 0: {table.num_rows} rows")

struct_arr = table.column('image').combine_chunks()
first = struct_arr[0].as_py()
print(f"First row keys: {list(first.keys())}")
print(f"bytes length: {len(first['bytes'])}")
print(f"path: {first['path']}")

img = Image.open(io.BytesIO(first['bytes']))
print(f"Image: {img.mode} {img.size}")

# Count total images and estimate size
total_bytes = 0
total_images = 0
seen_paths = set()
for i in range(min(100, total_rows)):
    row = struct_arr[i].as_py()
    total_bytes += len(row['bytes'])
    total_images += 1
    seen_paths.add(row['path'])

print(f"\nSampled {total_images} images: {total_bytes/1e6:.1f} MB")
print(f"Unique paths: {len(seen_paths)}")
avg_size = total_bytes / total_images if total_images else 0
print(f"Avg image size: {avg_size/1024:.1f} KB")
print(f"Estimated total size: {total_rows * avg_size / 1e9:.2f} GB")
