"""
Dataset Image Validator
Scans all images referenced in metadata to find corrupted/unreadable files.
Run BEFORE long training sessions to avoid 12-hour crashes.

Usage:
    python dataset/validate_images.py                          # full scan (all files)
    python dataset/validate_images.py --quick                  # size check only (fast)
    python dataset/validate_images.py --fix                    # delete zero-byte files
    python dataset/validate_images.py --clean-csv              # produce clean metadata
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image


PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
METADATA_PATH = PROJECT_ROOT / "dataset" / "metadata" / "clean_metadata.csv"
IMAGES_ROOT = PROJECT_ROOT / "dataset" / "images"


def check_file_size(path: Path) -> str:
    size = path.stat().st_size
    if size == 0:
        return "ZERO_BYTE"
    return "OK"


def try_open_image(path: Path) -> str:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return "OK"
    except (IOError, SyntaxError, Exception) as e:
        return f"CORRUPT: {e}"


def scan_metadata(quick: bool = False):
    if not METADATA_PATH.exists():
        print(f"Metadata not found: {METADATA_PATH}")
        sys.exit(1)

    total = 0
    results = {"OK": 0, "ZERO_BYTE": 0, "MISSING": 0, "CORRUPT": defaultdict(list)}
    zero_byte_files = []
    missing_files = []
    start = time.time()

    print(f"Scanning: {METADATA_PATH}")
    print(f"Mode: {'quick (size only)' if quick else 'full (open + verify)'}")
    print()

    with open(METADATA_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            filename = row.get("filename", row.get("image_id", ""))
            img_path = IMAGES_ROOT / filename

            if not img_path.exists():
                results["MISSING"] += 1
                missing_files.append(str(img_path))
                continue

            status = check_file_size(img_path)
            if status != "OK":
                results["ZERO_BYTE"] += 1
                zero_byte_files.append(str(img_path))
                continue

            if not quick:
                open_status = try_open_image(img_path)
                if open_status != "OK":
                    results["CORRUPT"][open_status].append(str(img_path))
                    continue

            results["OK"] += 1

            if total % 50000 == 0:
                elapsed = time.time() - start
                print(f"  Progress: {total:,} files scanned ({elapsed:.0f}s)")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE — {total:,} files in {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"  OK:          {results['OK']:>7,}")
    print(f"  Missing:     {results['MISSING']:>7,}")
    print(f"  Zero-byte:   {results['ZERO_BYTE']:>7,}")

    corrupt_count = sum(len(v) for v in results["CORRUPT"].values())
    print(f"  Corrupted:   {corrupt_count:>7,}")

    if results["ZERO_BYTE"] > 0:
        print(f"\n--- Zero-byte files ({results['ZERO_BYTE']}) ---")
        for f in zero_byte_files[:10]:
            print(f"  {f}")
        if len(zero_byte_files) > 10:
            print(f"  ... and {len(zero_byte_files) - 10} more")

    if corrupt_count > 0:
        print(f"\n--- Corrupted files ({corrupt_count}) ---")
        all_corrupt = []
        for reason, files in results["CORRUPT"].items():
            all_corrupt.extend(files)
        for f in all_corrupt[:10]:
            print(f"  {f}")
        if len(all_corrupt) > 10:
            print(f"  ... and {len(all_corrupt) - 10} more")

    print(f"\nTotal issues: {results['MISSING'] + results['ZERO_BYTE'] + corrupt_count}")
    return results, zero_byte_files, missing_files


def delete_zero_byte(files: list):
    deleted = 0
    for f in files:
        try:
            os.remove(f)
            deleted += 1
        except Exception as e:
            print(f"  Failed to delete {f}: {e}")
    print(f"Deleted {deleted} zero-byte files")


def generate_clean_csv(zero_byte_files: list, missing_files: list):
    bad_paths = set(zero_byte_files) | set(missing_files)
    clean_path = METADATA_PATH.with_name("clean_metadata_validated.csv")

    total_in = 0
    total_out = 0
    with open(METADATA_PATH, "r") as fin, open(clean_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            total_in += 1
            filename = row.get("filename", row.get("image_id", ""))
            img_path = str(IMAGES_ROOT / filename)
            if img_path not in bad_paths:
                writer.writerow(row)
                total_out += 1

    print(f"Clean CSV: {clean_path}")
    print(f"  Original: {total_in:,} rows")
    print(f"  Clean:    {total_out:,} rows")
    print(f"  Removed:  {total_in - total_out:,} bad entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset images")
    parser.add_argument("--quick", action="store_true", help="Size check only (fast)")
    parser.add_argument("--fix", action="store_true", help="Delete zero-byte files")
    parser.add_argument("--clean-csv", action="store_true", help="Generate clean metadata CSV")
    args = parser.parse_args()

    results, zero_byte, missing = scan_metadata(quick=args.quick)

    if args.fix and zero_byte:
        print(f"\n{'='*60}")
        delete_zero_byte(zero_byte)

    if args.clean_csv and (zero_byte or missing):
        print(f"\n{'='*60}")
        generate_clean_csv(zero_byte, missing)

    total_bad = results["MISSING"] + results["ZERO_BYTE"] + sum(len(v) for v in results["CORRUPT"].values())
    if total_bad > 0:
        print(f"\n⚠ Found {total_bad} problematic file(s). Run with --fix and --clean-csv to clean up.")
    else:
        print("\n✓ All files OK — happy training!")
"
