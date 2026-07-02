"""Check DiffusionDB file listing with sizes"""
import requests, sys

r = requests.get("https://huggingface.co/api/datasets/poloclub/diffusiondb", timeout=15)
data = r.json()
siblings = data.get("siblings", [])

import urllib.parse
part_files = [s for s in siblings if "part-" in s.get("rfilename", "")]
print(f"Total part files: {len(part_files)}")
print()

# Check sizes of first few files
for sib in part_files[:5]:
    rfilename = sib.get("rfilename", "")
    size = sib.get("size", 0)
    url = f"https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/{rfilename}"
    print(f"  {rfilename}")
    print(f"    Size: {size/1024/1024:.0f} MB")
    print(f"    URL: {url}")
    print()
