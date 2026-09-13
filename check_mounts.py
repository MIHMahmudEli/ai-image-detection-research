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
    _discover_mounted_slugs,
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

    # 3. Discover slug-level mounts (BUG A fix: walk 2 levels deep)
    slug_path_map = _discover_mounted_slugs(root)
    print(f"  Discovered slug-level mounts: {list(slug_path_map.keys())}\n")

    # 4. Match each expected slug (FIX 2)
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
            continue

        # Special case: artifact-* sub-datasets (same logic as _bootstrap_manifest)
        artifact_mount = slug_path_map.get("artifact-dataset")
        if slug.startswith("artifact-") and artifact_mount is not None:
            sub_name = slug[len("artifact-"):]  # e.g. "afhq"
            candidate = artifact_mount / sub_name
            if candidate.exists():
                results.append((slug, "MATCHED (artifact-subdataset)", str(candidate), subdir, []))
                continue

        matched_path, method, did_you_mean = match_mount(slug, slug_path_map, root)
        if matched_path:
            results.append((slug, f"MATCHED ({method})", str(matched_path), subdir, did_you_mean))
        else:
            results.append((slug, "NOT MATCHED", None, None, did_you_mean))

    # 5. Print summary table
    print(f"\n{'='*80}")
    print(f"  {'SLUG':<35} {'STATUS':<25} {'PATH':<30}")
    print(f"{'='*80}")
    for slug, status, path, extra, dym in results:
        print(f"  {slug:<35} {status:<25} {path or 'N/A':<30}")
        if dym and "NOT MATCHED" in status:
            print(f"    Did you mean: {', '.join(dym[:3])}")
    print(f"{'='*80}")

    # 6. Compute summary FROM the results table (BUG B fix)
    n_ok = sum(1 for _, s, _, _, _ in results if "MATCHED" in s or "OVERRIDDEN" in s)
    n_not_ok = sum(1 for _, s, _, _, _ in results if s == "NOT MATCHED" or s == "OVERRIDE_MISSING")
    n_total = len(results)

    # Sanity assertion: counts must be consistent
    assert n_ok + n_not_ok == n_total, (
        f"Count mismatch: {n_ok} ok + {n_not_ok} not_ok != {n_total} total"
    )

    if n_not_ok == 0:
        print(f"\n  ALL {n_total}/{n_total} datasets matched. Ready to train!")
    else:
        print(f"\n  {n_ok}/{n_total} datasets matched. {n_not_ok} MISSING.")
        print("\n  To fix, create a MOUNT_OVERRIDES_JSON Kaggle Secret with:")
        print('  {"expected-slug": "actual-folder-name-from-mount-tree-above"}')
        print("\n  Or re-attach the missing datasets via Kaggle UI > Add Input.")

    return n_not_ok == 0


if __name__ == "__main__":
    input_root = sys.argv[1] if len(sys.argv) > 1 else None
    ok = check_mounts(input_root)
    sys.exit(0 if ok else 1)
