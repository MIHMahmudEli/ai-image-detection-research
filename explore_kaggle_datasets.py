"""
Explore Kaggle Datasets
========================
Local, read-only, multi-account dataset lister via Kaggle API.

Lists all datasets owned by each configured account, printing:
  slug, size, file count, last updated

Purpose: quickly see which dataset shards exist across accounts
and are ready to attach via the Kaggle Input panel.

Does NOT touch training, checkpoints, or the manifest.

Usage:
    python explore_kaggle_datasets.py

Requires:
    .env file with KAGGLE_USERNAME, KAGGLE_KEY (one per account)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv(Path(__file__).parent / ".env")


def get_kaggle_creds():
    """Extract all KAGGLE_USERNAME/KAGGLE_KEY pairs from .env."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print("ERROR: .env not found")
        sys.exit(1)

    # Parse all lines — look for KAGGLE_USERNAME_N patterns or just KAGGLE_USERNAME
    accounts = []
    current = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            if val.startswith("'") and val.endswith("'"):
                val = val[1:-1]

            # Check for multi-account pattern: KAGGLE_USERNAME_1, KAGGLE_KEY_1, etc.
            for prefix in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
                if key == prefix:
                    current[prefix] = val
                elif key.startswith(prefix + "_"):
                    # Numbered account
                    idx = key[len(prefix) + 1:]
                    while len(accounts) <= int(idx):
                        accounts.append({})
                    accounts[int(idx)][prefix] = val

    if current.get("KAGGLE_USERNAME") and current.get("KAGGLE_KEY"):
        accounts.insert(0, current)

    # Merge: numbered accounts override single set
    if accounts:
        return accounts
    return [{"KAGGLE_USERNAME": os.environ.get("KAGGLE_USERNAME", ""),
             "KAGGLE_KEY": os.environ.get("KAGGLE_KEY", "")}]


def list_datasets(account):
    """List datasets for a single account."""
    username = account.get("KAGGLE_USERNAME", "")
    key = account.get("KAGGLE_KEY", "")
    if not username or not key:
        return []

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        datasets = api.dataset_list(owner=username, sort_by="dateUpdated", page_size=100)
        return datasets
    except ImportError:
        print("  ERROR: kaggle package not installed. Run: pip install kaggle")
        return []
    except Exception as e:
        print(f"  ERROR: {e}")
        return []


def format_size(size_bytes):
    """Human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def main():
    print("=" * 70)
    print("Kaggle Dataset Explorer — Multi-Account")
    print("=" * 70)

    accounts = get_kaggle_creds()
    if not accounts:
        print("No Kaggle accounts found in .env")
        sys.exit(1)

    total_datasets = 0
    for i, acc in enumerate(accounts):
        username = acc.get("KAGGLE_USERNAME", "???")
        print(f"\n{'─'*70}")
        print(f"Account #{i}: {username}")
        print(f"{'─'*70}")

        datasets = list_datasets(acc)
        if not datasets:
            print("  No datasets found or authentication failed.")
            continue

        for ds in datasets:
            total_datasets += 1
            slug = ds.ref
            size = format_size(ds.totalBytes) if hasattr(ds, 'totalBytes') and ds.totalBytes else "?"
            files = ds.fileCount if hasattr(ds, 'fileCount') else "?"
            updated = ds.lastUpdated[:10] if hasattr(ds, 'lastUpdated') and ds.lastUpdated else "?"

            print(f"  {slug}")
            print(f"    Size: {size} | Files: {files} | Updated: {updated}")

    print(f"\n{'='*70}")
    print(f"Total datasets across all accounts: {total_datasets}")
    print(f"{'='*70}")

    # Remind about required shards
    required = [
        "stable-diffusion", "places365", "open-images-v7-dataset",
        "ntire2026", "midjourney", "mfft-real", "genimage-ai",
        "faceforensics", "dfdc-faces-of-the-train-sample",
        "dall-e3", "celebdf-v2image-dataset",
    ]
    print(f"\nRequired shards for training:")
    for r in required:
        print(f"  [ ] {r}")


if __name__ == "__main__":
    main()
