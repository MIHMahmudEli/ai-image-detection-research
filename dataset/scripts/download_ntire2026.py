import os
import zipfile
from huggingface_hub import hf_hub_download, login

TOKEN = os.environ.get("HF_TOKEN")
if TOKEN:
    login(token=TOKEN, add_to_git_credential=False)

REPO = "deepfakesMSU/NTIRE-RobustAIGenDetection-train"
OUT_DIR = "dataset/images/NTIRE2026"
os.makedirs(OUT_DIR, exist_ok=True)

for i in range(6):
    fname = f"shard_{i}.zip"
    zip_path = os.path.join(OUT_DIR, fname)
    shard_dir = os.path.join(OUT_DIR, f"shard_{i}")

    if os.path.isdir(shard_dir):
        print(f"Shard {i} already extracted, skipping")
        continue

    print(f"Downloading {fname}...")
    hf_hub_download(REPO, fname, repo_type="dataset", local_dir=OUT_DIR, local_dir_use_symlinks=False)

    print(f"Extracting {fname}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(OUT_DIR)

    os.remove(zip_path)
    print(f"Done shard {i}")

print("All shards downloaded and extracted")
