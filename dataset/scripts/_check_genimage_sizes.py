"""Check GenImage file sizes by generator"""
import gdown

folder_id = "1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS"
url = f"https://drive.google.com/drive/folders/{folder_id}"

output = gdown.download_folder(url, quiet=True, skip_download=True)

# Group by folder and sum sizes
from collections import defaultdict
folder_sizes = defaultdict(lambda: {"files": 0, "bytes": 0})

for item in output:
    parts = item.path.replace("\\", "/").split("/")
    folder = parts[0]
    folder_sizes[folder]["files"] += 1
    folder_sizes[folder]["bytes"] += item.size if hasattr(item, 'size') and item.size else 0

for f in sorted(folder_sizes.keys()):
    info = folder_sizes[f]
    gb = info["bytes"] / (1024**3)
    print(f"{f:30s} {info['files']:3d} files  {gb:.1f} GB")
