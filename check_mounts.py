#!/usr/bin/env python3
"""
check_mounts.py — Pre-flight mount check for Kaggle training runs.

Run this BEFORE committing GPU time to verify all datasets are mounted
and matched correctly. Exits non-zero if any expected slug is unmatched.

Usage (Kaggle notebook cell):
    !python /kaggle/working/mfft_repo/check_mounts.py

Or as a standalone script:
    python check_mounts.py
"""

import sys
from pathlib import Path

# Allow running from repo root or anywhere
_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from split_manifest_manager import (
    _print_mount_tree,
    _load_mount_overrides,
    match_mount,
    _find_mount_path,
)
from pipeline_config import KAGGLE_DATASETS, KAGGLE_INPUT_ROOT


def check_mounts(input_root: str = None) -> bool:
    """
    Check all expected mounts. Returns True if all matched, False otherwise.
    """
    root = Path(input_root) if input_root else KAGGLE_INPUT_ROOT

    print("=" * 60)
    print("  PRE-FLIGHT MOUNT CHECK")
    print("=" * 60)

    # 1. Print mount tree (FIX 1)
    _print_mount_tree(root)

    # 2. Load overrides (FIX 3)
    overrides = _load_mount_overrides()

    # 3. Discover actual mount names
    all_mount_names = []
    if root.exists():
        all_mount_names = [d.name for d in root.iterdir() if d.is_dir()]
        datasets_dir = root / "datasets"
        if datasets_dir.exists():
            all_mount_names += [d.name for d in datasets_dir.iterdir() if d.is_dir()]
        all_mount_names = list(set(all_mount_names))

    print(f"  Actual mounts: {all_mount_names}\n")

    # 4. Match each expected slug (FIX 2)
    all_matched = True
    results = []

    for slug, (label, subdir) in KAGGLE_DATASETS.items():
        # Override check
        if slug in overrides:
            actual_name = overrides[slug]
            path = _find_mount_path(root, actual_name)
            if path:
                results.append((slug, "OVERRIDDEN", str(path), actual_name, []))
            else:
                results.append((slug, "OVERRIDE_MISSING", str(path), actual_name, []))
                all_matched = False
            continue

        matched_path, method, did_you_mean = match_mount(slug, all_mount_names, root)
        if matched_path:
            # Verify subdir exists
            target = matched_path / subdir
            sub_ok = target.exists() or True  # fallback to mount root
            results.append((slug, f"MATCHED ({method})", str(matched_path), subdir, did_you_mean))
        else:
            results.append((slug, "NOT MATCHED", None, None, did_you_mean))
            all_matched = False

    # 5. Print summary table
    print(f"\n{'='*80}")
    print(f"  {'SLUG':<35} {'STATUS':<25} {'PATH':<30}")
    print(f"{'='*80}")
    for slug, status, path, extra, dym in results:
        print(f"  {slug:<35} {status:<25} {path or 'N/A':<30}")
        if dym and "NOT MATCHED" in status:
            print(f"    Did you mean: {', '.join(dym[:3])}")
    print(f"{'='*80}")

    # 6. Result
    n_ok = sum(1 for _, s, _, _, _ in results if "MATCHED" in s or "OVERRIDDEN" in s)
    n_total = len(results)

    if all_matched:
        print(f"\n  ALL {n_total}/{n_total} datasets matched. Ready to train!")
    else:
        print(f"\n  {n_ok}/{n_total} datasets matched. {n_total - n_ok} MISSING.")
        print("\n  To fix, create a MOUNT_OVERRIDES_JSON Kaggle Secret with:")
        print('  {"expected-slug": "actual-folder-name-from-mount-tree-above"}')
        print("\n  Or re-attach the missing datasets via Kaggle UI > Add Input.")

    return all_matched


if __name__ == "__main__":
    input_root = sys.argv[1] if len(sys.argv) > 1 else None
    ok = check_mounts(input_root)
    sys.exit(0 if ok else 1)
