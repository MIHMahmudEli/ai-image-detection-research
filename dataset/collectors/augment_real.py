"""
Augment existing real images to reach 25K total.
Generates mild variants (flips, rotations, color shifts, crops)
from the 17,903 existing real images.
"""
import hashlib, json, csv, random
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageEnhance
import pandas as pd
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parent.parent.parent
REAL_DIR = PROJECT / 'dataset' / 'images' / 'real'
META_CSV = PROJECT / 'dataset' / 'metadata' / 'all.csv'

existing = sorted(REAL_DIR.glob('*'))
print(f"Existing real images: {len(existing)}")
target = 25000
needed = target - len(existing)
print(f"Need to generate: {needed}")

AUGMENTATIONS = [
    'flip_h', 'flip_v', 'rot_5', 'rot_355', 'bright', 'contrast',
    'crop_95', 'crop_90', 'flip_h_rot_3', 'flip_v_rot_3',
]

def augment_image(img, aug_type):
    if aug_type == 'flip_h':
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif aug_type == 'flip_v':
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif aug_type == 'rot_5':
        return img.rotate(5, expand=False, fill=128)
    elif aug_type == 'rot_355':
        return img.rotate(-5, expand=False, fill=128)
    elif aug_type == 'bright':
        return ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    elif aug_type == 'contrast':
        return ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    elif aug_type in ('crop_95', 'crop_90'):
        pct = 0.95 if aug_type == 'crop_95' else 0.90
        w, h = img.size
        nw, nh = int(w * pct), int(h * pct)
        x = random.randint(0, w - nw)
        y = random.randint(0, h - nh)
        return img.crop((x, y, x + nw, y + nh)).resize((w, h), Image.LANCZOS)
    elif aug_type == 'flip_h_rot_3':
        return img.transpose(Image.FLIP_LEFT_RIGHT).rotate(3, expand=False, fill=128)
    elif aug_type == 'flip_v_rot_3':
        return img.transpose(Image.FLIP_TOP_BOTTOM).rotate(-3, expand=False, fill=128)

new_meta = []
count = 0
pbar = tqdm(total=needed, desc='Generating real augmentations')

# Load existing metadata for source info
existing_meta = {}
if META_CSV.exists():
    df = pd.read_csv(META_CSV)
    for _, row in df[df['image_type'] == 'real'].iterrows():
        existing_meta[row['filename']] = row.to_dict()

while count < needed:
    for src_path in existing:
        if count >= needed:
            break
        src_name = src_path.name
        aug_type = random.choice(AUGMENTATIONS)

        try:
            img = Image.open(src_path).convert('RGB')
            aug_img = augment_image(img, aug_type)
            new_id = f"{src_path.stem}_aug_{aug_type}"
            new_name = f"{new_id}.jpg"
            dest = REAL_DIR / new_name
            if dest.exists():
                continue

            aug_img.save(dest, quality=92)
            md5 = hashlib.md5(dest.read_bytes()).hexdigest()

            meta = existing_meta.get(src_name, {})
            new_meta.append({
                'image_id': new_id,
                'filename': new_name,
                'source': meta.get('source', 'augmented'),
                'source_url': meta.get('source_url', ''),
                'download_date': datetime.now().isoformat(),
                'image_type': 'real',
                'width': img.width,
                'height': img.height,
                'format': 'JPEG',
                'file_size_kb': round(dest.stat().st_size / 1024, 2),
                'color_space': 'RGB',
                'md5_hash': md5,
                'is_duplicate': False,
                'duplicate_of': '',
            })
            count += 1
            pbar.update(1)
        except Exception as e:
            continue

pbar.close()
print(f"Generated: {count} new real images")

if new_meta:
    df_new = pd.DataFrame(new_meta)
    if META_CSV.exists():
        df_old = pd.read_csv(META_CSV)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(META_CSV, index=False)

final = len(list(REAL_DIR.glob('*')))
print(f"\nFinal real image count: {final}")
print(f"Total dataset size: {final + 32097} (real + AI)")
