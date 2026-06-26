from pathlib import Path
import pandas as pd

base = Path(r'C:\Users\ROWTECH\Desktop\ReSearch\ai-image-detection-research\dataset')
csv = pd.read_csv(base / 'metadata' / 'all.csv')

for t in ['real', 'ai_generated', 'ai_altered']:
    subset = csv[csv['image_type'] == t]
    print(f'=== {t} ({len(subset)} images) ===')
    print(f'  Width:  mean={subset["width"].mean():.0f}, min={subset["width"].min()}, max={subset["width"].max()}')
    print(f'  Height: mean={subset["height"].mean():.0f}, min={subset["height"].min()}, max={subset["height"].max()}')
    print(f'  File size (KB): mean={subset["file_size_kb"].mean():.0f}, min={subset["file_size_kb"].min()}, max={subset["file_size_kb"].max()}')
    print(f'  Sources: {subset["source"].value_counts().to_dict()}')
    print()
