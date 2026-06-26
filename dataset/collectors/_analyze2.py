from pathlib import Path
from PIL import Image
import random, statistics

base = Path(r'C:\Users\ROWTECH\Desktop\ReSearch\ai-image-detection-research\dataset\images')

for folder in ['real', 'ai_generated', 'ai_altered']:
    d = base / folder
    files = list(d.glob('*'))[:200]
    sizes = []
    ratios = []
    errors = 0
    for f in files:
        try:
            img = Image.open(f)
            sizes.append(img.size)
            ratios.append(img.width / img.height)
        except:
            errors += 1
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    print(f'=== {folder} ({len(list(d.glob("*")))} total, sampled {len(sizes)}) ===')
    print(f'  Avg: {statistics.mean(widths):.0f}x{statistics.mean(heights):.0f}')
    print(f'  Min: {min(widths)}x{min(heights)}, Max: {max(widths)}x{max(heights)}')
    print(f'  Aspect ratio: mean={statistics.mean(ratios):.2f}, min={min(ratios):.2f}, max={max(ratios):.2f}')
    print(f'  Corrupted files: {errors}')
    print()
