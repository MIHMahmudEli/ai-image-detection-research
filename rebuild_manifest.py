#!/usr/bin/env python3
"""
rebuild_manifest.py
===================
Rebuild the frozen split manifest with:
  FIX 1: Per-class, per-shard proportional allocation (largest-remainder)
  FIX 2: Minimum per-shard test floor (Q1-defensible)
  FIX 3: Video/identity-grouped sampling + splitting for deepfake sources
  FIX 4: Rebuild in place (overwrite HF manifest, version stays 1)

Usage (Kaggle notebook):
    !python rebuild_manifest.py --yes
    !python rebuild_manifest.py --force-rebuild --yes

Standalone:
    python rebuild_manifest.py --input-root /kaggle/input --yes
"""

import os
import sys
import json
import math
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

import numpy as np

from pipeline_config import (
    HF_MANIFEST_REPO, MANIFEST_PATH_IN_REPO, KAGGLE_INPUT_ROOT,
    SEED, VAL_SPLIT, TEST_SPLIT, LABEL_MAP, CLASS_NAMES,
    IMG_EXTENSIONS, KAGGLE_DATASETS,
    TARGET_TOTAL_IMAGES, TARGET_CLASS_RATIOS,
    TEST_MIN_PER_SHARD, MAX_FRAMES_PER_VIDEO,
    DEEPFAKE_VIDEO_SHARDS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Image ID (deterministic, same as split_manifest_manager.py)
# ═══════════════════════════════════════════════════════════════

def stable_image_id(filename: str, shard: str, label: str) -> str:
    return hashlib.sha1(f"{filename}|{shard}|{label}".encode()).hexdigest()[:20]


def compute_manifest_sha256(manifest: dict) -> str:
    m = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════
# FIX 3: Group ID extraction for deepfake video shards
# ═══════════════════════════════════════════════════════════════

def extract_group_id(path: str, shard: str) -> str:
    """
    Derive a group_id (source video / identity) from a file path.

    Per-shard conventions:
      faceforensics: cropped_images/{identity}_{video}/{frame}.png
                     -> group_id = "{identity}_{video}" (parent dir name)
      dfdc:          train/fake/{video_id}_{frame}.png
                     -> group_id = video_id (first token before '_')
      celebdf-v2:    Celeb_V2/{split}/fake/{src}_{tgt}_{vid}_face_{frame}.jpg
                     Celeb_V2/{split}/real/{vid}_face_{frame}.jpg
                     -> group_id = everything before '_face_'
    """
    fname = Path(path).stem

    if shard == "faceforensics":
        return Path(path).parent.name

    elif shard == "dfdc-faces-of-the-train-sample":
        return fname.split("_")[0]

    elif shard == "celebdf-v2image-dataset":
        if "_face_" in fname:
            return fname.rsplit("_face_", 1)[0]
        return fname

    else:
        return fname


# ═══════════════════════════════════════════════════════════════
# Mount discovery & scanning (adapted from split_manifest_manager.py)
# ═══════════════════════════════════════════════════════════════

_SLUG_NORMALIZE_RE = __import__("re").compile(r"[-_\s]+")

def _normalize_slug(s: str) -> str:
    return _SLUG_NORMALIZE_RE.sub("-", s.lower().strip())

def _tokenize(s: str) -> Set[str]:
    return set(__import__("re").findall(r"[a-z0-9]+", s.lower()))

def _token_overlap_score(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    return len(ta & tb) / len(ta) if ta and tb else 0.0

def _load_mount_overrides() -> Dict[str, str]:
    raw = None
    try:
        from kaggle_secrets import UserSecretsClient
        raw = UserSecretsClient().get_secret("MOUNT_OVERRIDES_JSON")
    except Exception:
        pass
    if raw is None:
        raw = os.environ.get("MOUNT_OVERRIDES_JSON")
    if raw is None:
        for p in [Path("/kaggle/working/mount_overrides.json"),
                  Path(__file__).parent / "mount_overrides.json"]:
            if p.exists():
                raw = p.read_text()
                break
    if raw is None:
        return {}
    try:
        overrides = json.loads(raw)
        if overrides:
            print(f"  [MountOverrides] Loaded {len(overrides)} override(s)")
        return overrides
    except json.JSONDecodeError as e:
        print(f"  [MountOverrides] WARNING: Invalid JSON: {e}")
        return {}

def _discover_mounted_slugs(input_root: Path) -> Dict[str, Path]:
    slug_map: Dict[str, List[Path]] = {}
    def _scan(base: Path):
        if not base.exists():
            return
        for owner in base.iterdir():
            if not owner.is_dir():
                continue
            for child in owner.iterdir():
                if child.is_dir():
                    slug_map.setdefault(child.name, []).append(child)
    _scan(input_root)
    _scan(input_root / "datasets")
    result: Dict[str, Path] = {}
    for slug, paths in slug_map.items():
        if len(paths) > 1:
            print(f"  WARNING: slug '{slug}' found under multiple owners, using first")
        result[slug] = paths[0]
    return result

def _find_mount_path(input_root: Path, mount_name: str) -> Optional[Path]:
    for c in [input_root / mount_name, input_root / "datasets" / mount_name]:
        if c.exists():
            return c
    for base in [input_root, input_root / "datasets"]:
        if base.exists():
            for owner in base.iterdir():
                if owner.is_dir():
                    candidate = owner / mount_name
                    if candidate.exists():
                        return candidate
    return None

def match_mount(slug, slug_path_map, input_root):
    all_candidates = list(slug_path_map.keys())
    if slug in slug_path_map:
        return slug_path_map[slug], "exact", []
    slug_lower = slug.lower()
    for c in all_candidates:
        if slug_lower == c.lower():
            return slug_path_map[c], "case-insensitive", []
    slug_norm = _normalize_slug(slug)
    for c in all_candidates:
        if slug_norm == _normalize_slug(c):
            return slug_path_map[c], "normalized", []
    for c in all_candidates:
        if slug in c or c in slug:
            return slug_path_map[c], "substring", []
    best, best_score, dym = None, 0.0, []
    for c in all_candidates:
        score = _token_overlap_score(slug, c)
        if score > 0:
            dym.append(f"{c} (overlap={score:.0%})")
        if score >= 0.5 and score > best_score:
            best_score, best = score, c
    if best:
        return slug_path_map[best], f"token-overlap({best_score:.0%})", dym
    return None, "no-match", dym


def discover_and_scan_mounts(input_root: Path) -> Dict[str, List[dict]]:
    """
    Discover mounts, scan each matched shard, return {shard: [row, ...]}.
    Each row: {image_id, shard, label, label_int, rel_path, group_id}.
    """
    overrides = _load_mount_overrides()
    slug_path_map = _discover_mounted_slugs(input_root)
    print(f"  Discovered slug-level mounts: {list(slug_path_map.keys())}")

    artifact_mount = slug_path_map.get("artifact-dataset")
    matched_paths: Dict[str, Path] = {}

    for slug in KAGGLE_DATASETS:
        if slug in overrides:
            path = _find_mount_path(input_root, overrides[slug])
            if path:
                matched_paths[slug] = path
                print(f"  {slug}: OVERRIDDEN -> {overrides[slug]}", flush=True)
                continue
        if slug.startswith("artifact-") and artifact_mount is not None:
            sub = artifact_mount / slug[len("artifact-"):]
            if sub.exists():
                matched_paths[slug] = sub
                print(f"  {slug}: MATCHED (artifact-subdataset) -> {sub}", flush=True)
                continue
        mp, method, _ = match_mount(slug, slug_path_map, input_root)
        if mp:
            matched_paths[slug] = mp
            print(f"  {slug}: MATCHED ({method})", flush=True)
        else:
            print(f"  {slug}: NOT MATCHED", flush=True)

    IMG_EXT_TUPLE = tuple(IMG_EXTENSIONS)

    def _fast_scan(root):
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith(IMG_EXT_TUPLE):
                    full = os.path.join(dirpath, fname)
                    try:
                        if os.path.getsize(full) > 0:
                            yield full
                    except OSError:
                        continue

    def _scan_artifact_metadata(mount_dir, slug, label):
        """Scan artifact sub-dataset using its metadata.csv."""
        import pandas as pd
        csv_path = mount_dir / "metadata.csv"
        if not csv_path.exists():
            # Try one level deeper
            for child in mount_dir.iterdir():
                if child.is_dir() and (child / "metadata.csv").exists():
                    csv_path = child / "metadata.csv"
                    break
        if not csv_path.exists():
            return []

        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"  WARNING: Cannot read {csv_path}: {e}")
            return []

        rows = []
        # Columns: filename, image_path, target, category
        # target: 0=real, 1-6=fake — we only want fake (target > 0)
        for _, r in df.iterrows():
            target = int(r.get("target", 0))
            if target == 0:
                continue  # Skip real images from artifact
            img_path = str(r.get("image_path", r.get("filename", "")))
            if not img_path:
                continue
            # Make absolute if relative
            if not os.path.isabs(img_path):
                img_path = str(mount_dir / img_path)
            if not os.path.exists(img_path):
                continue
            try:
                if os.path.getsize(img_path) == 0:
                    continue
            except OSError:
                continue
            fname = os.path.basename(img_path)
            img_id = stable_image_id(fname, slug, label)
            group_id = extract_group_id(img_path, slug)
            rows.append({
                "image_id": img_id,
                "shard": slug,
                "label": label,
                "label_int": LABEL_MAP[label],
                "rel_path": img_path,
                "group_id": group_id,
            })
        return rows

    shard_rows: Dict[str, List[dict]] = {}
    for slug, (label, subdir) in KAGGLE_DATASETS.items():
        if slug not in matched_paths:
            continue
        mount_dir = matched_paths[slug]
        image_root = mount_dir / subdir
        if not image_root.exists():
            image_root = mount_dir

        print(f"  Scanning {slug}...", end=" ", flush=True)

        # Artifact sub-datasets use metadata.csv — read it directly
        if slug.startswith("artifact-"):
            rows = _scan_artifact_metadata(mount_dir, slug, label)
        else:
            rows = []
            for img_path in _fast_scan(str(image_root)):
                img_id = stable_image_id(os.path.basename(img_path), slug, label)
                group_id = extract_group_id(img_path, slug)
                rows.append({
                    "image_id": img_id,
                    "shard": slug,
                    "label": label,
                    "label_int": LABEL_MAP[label],
                    "rel_path": img_path,
                    "group_id": group_id,
                })

        print(f"{len(rows)} images ({label})", flush=True)
        shard_rows[slug] = rows

    return shard_rows


# ═══════════════════════════════════════════════════════════════
# FIX 1: Proportional allocation (largest-remainder / waterfill)
# ═══════════════════════════════════════════════════════════════

def _allocate_proportional(
    availability: Dict[str, int],
    target: int,
    seed: int = SEED,
) -> Dict[str, int]:
    """
    Largest-remainder proportional allocation, capped at availability.
    Unallocated budget is redistributed via waterfilling across remaining capacity.
    """
    total_avail = sum(availability.values())
    if total_avail <= 0 or target <= 0:
        return {s: 0 for s in availability}

    target = min(target, total_avail)
    shards = sorted(availability.keys())

    # Floor division: integer proportional share
    floor_alloc = {}
    for s in shards:
        exact = target * availability[s] / total_avail
        floor_alloc[s] = min(availability[s], int(exact))

    remainder = target - sum(floor_alloc.values())
    remainders = {s: (target * availability[s] / total_avail) - floor_alloc[s] for s in shards}

    # Distribute remainder 1-by-1 to shards with largest fractional part
    for s in sorted(shards, key=lambda x: -remainders[x]):
        if remainder <= 0:
            break
        if floor_alloc[s] < availability[s]:
            floor_alloc[s] += 1
            remainder -= 1

    # Waterfill: if budget still remains, distribute across shards with room
    budget_left = target - sum(floor_alloc.values())
    if budget_left > 0:
        rng = np.random.RandomState(seed)
        order = list(rng.permutation(shards))
        for s in order:
            if budget_left <= 0:
                break
            room = availability[s] - floor_alloc[s]
            give = min(budget_left, room)
            if give > 0:
                floor_alloc[s] += give
                budget_left -= give

    return floor_alloc


# ═══════════════════════════════════════════════════════════════
# FIX 2: Enforce per-shard test floor
# ═══════════════════════════════════════════════════════════════

def _redistribute_test_floor_boosts(
    allocations: Dict[str, int],
    availability: Dict[str, int],
    class_shards: List[str],
    test_floor: int,
    holdout: float,
) -> Tuple[Dict[str, int], List[dict]]:
    """
    Iteratively boost shards below the test floor by pulling proportionally
    from other shards in the same class. Stops when stable or max rounds reached.
    """
    max_rounds = 10
    for _ in range(max_rounds):
        boosts_needed = {}
        for s in class_shards:
            alloc = allocations.get(s, 0)
            avail = availability.get(s, 0)
            est_test = int(alloc * holdout)
            if est_test < test_floor and alloc < avail:
                needed_total = math.ceil(test_floor / holdout)
                boosts_needed[s] = min(needed_total - alloc, avail - alloc)

        if not boosts_needed:
            break

        total_boost = sum(boosts_needed.values())
        donors = [s for s in class_shards if s not in boosts_needed and allocations[s] > 0]
        total_donor_alloc = sum(allocations[s] for s in donors)

        for s in sorted(boosts_needed.keys()):
            boost = boosts_needed[s]
            if boost <= 0:
                continue
            pulled = 0
            for d in donors:
                if pulled >= boost:
                    break
                # Pull proportionally from each donor
                share = int(boost * allocations[d] / total_donor_alloc) if total_donor_alloc > 0 else 0
                give = min(share, allocations[d], boost - pulled)
                if give > 0:
                    allocations[d] -= give
                    allocations[s] += give
                    pulled += give
            # If still short, take remainder from any available donor
            if pulled < boost:
                for d in donors:
                    if pulled >= boost:
                        break
                    give = min(allocations[d], boost - pulled)
                    if give > 0:
                        allocations[d] -= give
                        allocations[s] += give
                        pulled += give

    # Build below-floor report
    below_floor = []
    for s in sorted(class_shards):
        alloc = allocations.get(s, 0)
        avail = availability.get(s, 0)
        if alloc == 0:
            continue
        est_test = int(alloc * holdout)
        if est_test < test_floor:
            below_floor.append({
                "shard": s,
                "allocated": alloc,
                "available": avail,
                "est_test": est_test,
                "flag": "below_statistical_floor",
            })

    return allocations, below_floor


# ═══════════════════════════════════════════════════════════════
# Sampling helpers
# ═══════════════════════════════════════════════════════════════

def _sample_flat_shard(rows: List[dict], n: int, seed: int) -> List[dict]:
    """Random sampling without replacement (FIX 1.3: never filesystem order)."""
    if n >= len(rows):
        return list(rows)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(rows), size=n, replace=False)
    return [rows[i] for i in sorted(indices)]


def _sample_deepfake_videos(
    rows: List[dict],
    n: int,
    max_frames_per_video: int,
    seed: int,
) -> List[dict]:
    """
    FIX 3.2: Sample videos, take up to max_frames_per_video from each,
    spreading frames evenly (not consecutive).
    """
    rng = np.random.RandomState(seed)

    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)

    video_ids = list(groups.keys())
    rng.shuffle(video_ids)

    sampled = []
    for vid in video_ids:
        if len(sampled) >= n:
            break
        frames = groups[vid]
        if len(frames) <= max_frames_per_video:
            sampled.extend(frames)
        else:
            # Spread evenly through the video
            step = max(1, len(frames) // max_frames_per_video)
            selected = frames[::step][:max_frames_per_video]
            sampled.extend(selected)

    return sampled[:n]


# ═══════════════════════════════════════════════════════════════
# Splitting
# ═══════════════════════════════════════════════════════════════

def _stratified_split(rows: List[dict], seed: int) -> Dict[str, List[dict]]:
    """Stratified train/val/test split by label_int."""
    from sklearn.model_selection import train_test_split

    if not rows:
        return {"train": [], "val": [], "test": []}

    labels = [r["label_int"] for r in rows]
    holdout = VAL_SPLIT + TEST_SPLIT

    train_val_idx, test_idx = train_test_split(
        list(range(len(rows))), test_size=holdout,
        stratify=labels, random_state=seed,
    )
    rv_labels = [labels[i] for i in train_val_idx]
    val_ratio = VAL_SPLIT / holdout
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio,
        stratify=rv_labels, random_state=seed,
    )

    return {
        "train": [rows[i] for i in train_idx],
        "val": [rows[i] for i in val_idx],
        "test": [rows[i] for i in test_idx],
    }


def _group_aware_split(rows: List[dict], seed: int) -> Dict[str, List[dict]]:
    """
    FIX 3.3: Split by group_id (video/identity) so no group appears in
    more than one of train/val/test.
    """
    from sklearn.model_selection import GroupShuffleSplit

    if not rows:
        return {"train": [], "val": [], "test": []}

    groups = [r["group_id"] for r in rows]
    labels = [r["label_int"] for r in rows]

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT + TEST_SPLIT, random_state=seed)
    train_val_idx, test_idx = next(gss_outer.split(rows, labels, groups))

    train_val_rows = [rows[i] for i in train_val_idx]
    train_val_groups = [groups[i] for i in train_val_idx]
    train_val_labels = [labels[i] for i in train_val_idx]

    val_size_relative = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=seed)
    train_idx_local, val_idx_local = next(
        gss_inner.split(train_val_rows, train_val_labels, train_val_groups)
    )

    return {
        "train": [train_val_rows[i] for i in train_idx_local],
        "val": [train_val_rows[i] for i in val_idx_local],
        "test": [rows[i] for i in test_idx],
    }


def _verify_no_group_leakage(splits: Dict[str, List[dict]], shard_name: str):
    """FIX 3.5: Hard assertion — no group_id appears in more than one split."""
    seen: Dict[str, Set[str]] = defaultdict(set)
    for split_name, rows in splits.items():
        for r in rows:
            gid = r["group_id"]
            seen[gid].add(split_name)

    for gid, split_names in seen.items():
        if len(split_names) > 1:
            raise AssertionError(
                f"GROUP LEAKAGE in {shard_name}: group_id '{gid}' appears in "
                f"splits {sorted(split_names)}"
            )


# ═══════════════════════════════════════════════════════════════
# Print helpers
# ═══════════════════════════════════════════════════════════════

def _print_allocation_table(
    availability: Dict[str, int],
    allocations: Dict[str, int],
    shard_class: Dict[str, str],
):
    print(f"\n{'='*72}")
    print(f"  PER-SHARD ALLOCATION TABLE (FIX 1)")
    print(f"{'='*72}")
    print(f"  {'Shard':<40s} {'Class':<14s} {'Avail':>8s} {'Alloc':>8s} {'%':>6s}")
    print(f"  {'-'*40} {'-'*14} {'-'*8} {'-'*8} {'-'*6}")
    for s in sorted(availability.keys()):
        a = allocations.get(s, 0)
        av = availability[s]
        pct = a / av * 100 if av > 0 else 0
        print(f"  {s:<40s} {shard_class[s]:<14s} {av:>8d} {a:>8d} {pct:>5.1f}%")
    total_a = sum(allocations.values())
    total_v = sum(availability.values())
    print(f"  {'-'*40} {'-'*14} {'-'*8} {'-'*8} {'-'*6}")
    print(f"  {'TOTAL':<40s} {'':14s} {total_v:>8d} {total_a:>8d} {total_a/total_v*100:>5.1f}%")
    print(f"{'='*72}\n")


def _print_test_floor_report(below_floor: List[dict]):
    if not below_floor:
        print(f"\n  [FIX 2] All shards meet TEST_MIN_PER_SHARD = {TEST_MIN_PER_SHARD}")
        return
    print(f"\n{'='*72}")
    print(f"  BELOW TEST FLOOR (FIX 2) -- use with caution / note as limitation")
    print(f"{'='*72}")
    print(f"  {'Shard':<40s} {'Alloc':>8s} {'Avail':>8s} {'EstTest':>8s}  Flag")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")
    for item in sorted(below_floor, key=lambda x: x["est_test"]):
        print(f"  {item['shard']:<40s} {item['allocated']:>8d} "
              f"{item['available']:>8d} {item['est_test']:>8d}  {item['flag']}")
    print(f"{'='*72}")


def _print_deepfake_video_stats(
    shard_name: str,
    all_rows: List[dict],
    sampled_rows: List[dict],
):
    all_groups = defaultdict(int)
    for r in all_rows:
        all_groups[r["group_id"]] += 1
    sampled_groups = defaultdict(list)
    for r in sampled_rows:
        sampled_groups[r["group_id"]].append(r)

    frames_per = [len(v) for v in sampled_groups.values()]
    print(f"\n  [FIX 3] {shard_name}:")
    print(f"    Distinct videos available:   {len(all_groups)}")
    print(f"    Videos selected:             {len(sampled_groups)}")
    if frames_per:
        print(f"    Frames per selected video:   min={min(frames_per)}, "
              f"max={max(frames_per)}, mean={sum(frames_per)/len(frames_per):.1f}")
    print(f"    Total frames sampled:        {len(sampled_rows)}")


# ═══════════════════════════════════════════════════════════════
# Main build logic
# ═══════════════════════════════════════════════════════════════

def build_rebuilt_manifest(
    input_root: Path,
    run_id: str = "",
    force: bool = False,
) -> Tuple[dict, str]:
    """
    Full rebuild pipeline:
      1. Discover + scan all mounted shards
      2. FIX 1: Proportional allocation per class
      3. FIX 2: Enforce per-shard test floor
      4. Sample images (FIX 3: video-aware for deepfake shards)
      5. Split (FIX 3: group-aware for deepfake shards)
      6. Build + return manifest
    """
    print(f"\n{'#'*72}")
    print(f"  MANIFEST REBUILD -- target {TARGET_TOTAL_IMAGES:,} images")
    print(f"  Class ratios: {TARGET_CLASS_RATIOS}")
    print(f"  Test floor: {TEST_MIN_PER_SHARD}/shard, Max frames/video: {MAX_FRAMES_PER_VIDEO}")
    print(f"{'#'*72}\n")

    # ── 1. Discover and scan ──
    shard_rows = discover_and_scan_mounts(input_root)
    if not shard_rows:
        raise RuntimeError("No images found in any mounted dataset. See diagnostics above.")

    # ── 2. Build availability by class and shard ──
    shard_class: Dict[str, str] = {}
    availability: Dict[str, int] = {}
    class_shards: Dict[str, List[str]] = defaultdict(list)

    for slug, (label, _) in KAGGLE_DATASETS.items():
        if slug in shard_rows:
            shard_class[slug] = label
            availability[slug] = len(shard_rows[slug])
            class_shards[label].append(slug)

    # ── FIX 1: Per-class proportional allocation ──
    print(f"\n  Computing per-class targets from {TARGET_TOTAL_IMAGES:,} total...")
    class_targets = {}
    for cls, ratio in TARGET_CLASS_RATIOS.items():
        target = round(TARGET_TOTAL_IMAGES * ratio)
        class_targets[cls] = target
        print(f"    {cls}: {ratio:.1%} -> {target:,}")

    allocations: Dict[str, int] = {}
    for cls in CLASS_NAMES:
        shards = class_shards.get(cls, [])
        if not shards:
            continue
        cls_avail = {s: availability[s] for s in shards}
        cls_target = class_targets[cls]
        alloc = _allocate_proportional(cls_avail, cls_target)
        allocations.update(alloc)

    _print_allocation_table(availability, allocations, shard_class)

    # ── FIX 2: Enforce test floor (iterate until stable) ──
    holdout = VAL_SPLIT + TEST_SPLIT
    all_below_floor = []

    for cls in CLASS_NAMES:
        shards = class_shards.get(cls, [])
        if not shards:
            continue
        allocations, below = _redistribute_test_floor_boosts(
            allocations, availability, shards, TEST_MIN_PER_SHARD, holdout,
        )
        all_below_floor.extend(below)

    _print_test_floor_report(all_below_floor)

    # ── 4. Sample + Split ──
    all_sampled: List[dict] = []

    for cls in CLASS_NAMES:
        shards = class_shards.get(cls, [])
        for shard in sorted(shards):
            alloc = allocations.get(shard, 0)
            if alloc == 0:
                continue

            rows = shard_rows[shard]
            is_video_shard = shard in DEEPFAKE_VIDEO_SHARDS

            if is_video_shard:
                sampled = _sample_deepfake_videos(rows, alloc, MAX_FRAMES_PER_VIDEO, SEED)
                _print_deepfake_video_stats(shard, rows, sampled)
            else:
                sampled = _sample_flat_shard(rows, alloc, SEED)

            all_sampled.extend(sampled)

    print(f"\n  Sampled total: {len(all_sampled):,} images")

    # ── 5. Split: group-aware for video shards, stratified for the rest ──
    video_rows = [r for r in all_sampled if r["shard"] in DEEPFAKE_VIDEO_SHARDS]
    non_video_rows = [r for r in all_sampled if r["shard"] not in DEEPFAKE_VIDEO_SHARDS]

    print(f"  Splitting {len(non_video_rows):,} non-video images (stratified)...")
    non_video_splits = _stratified_split(non_video_rows, SEED)

    video_splits_by_shard: Dict[str, Dict[str, List[dict]]] = {}
    for shard in sorted(DEEPFAKE_VIDEO_SHARDS):
        shard_vid_rows = [r for r in video_rows if r["shard"] == shard]
        if not shard_vid_rows:
            continue
        print(f"  Splitting {shard} ({len(shard_vid_rows):,} frames, group-aware)...")
        splits = _group_aware_split(shard_vid_rows, SEED)
        _verify_no_group_leakage(splits, shard)
        print(f"    PASS: no group_id leakage for {shard}")
        video_splits_by_shard[shard] = splits

    # Merge splits
    final_splits = {"train": [], "val": [], "test": []}
    for split_name in ["train", "val", "test"]:
        final_splits[split_name].extend(non_video_splits[split_name])
        for shard_splits in video_splits_by_shard.values():
            final_splits[split_name].extend(shard_splits[split_name])

    # ── 6. Build manifest ──
    images_dict = {}
    for split_name, rows in final_splits.items():
        for r in rows:
            images_dict[r["image_id"]] = {
                "shard": r["shard"],
                "label": r["label"],
                "label_int": r["label_int"],
                "split": split_name,
            }

    total = len(images_dict)
    class_dist = {}
    for lbl_name, lbl_int in LABEL_MAP.items():
        cnt = sum(1 for v in images_dict.values() if v["label_int"] == lbl_int)
        class_dist[lbl_name] = round(cnt / total, 4) if total > 0 else 0

    split_sizes = {s: len(final_splits[s]) for s in ["train", "val", "test"]}
    shards_used = sorted(set(r["shard"] for r in all_sampled))

    manifest = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "created_by_run": run_id or "rebuild_manifest",
        "seed": SEED,
        "target_total": TARGET_TOTAL_IMAGES,
        "target_class_ratios": TARGET_CLASS_RATIOS,
        "test_min_per_shard": TEST_MIN_PER_SHARD,
        "max_frames_per_video": MAX_FRAMES_PER_VIDEO,
        "shards_used": shards_used,
        "total_images": total,
        "class_distribution": class_dist,
        "split_sizes": split_sizes,
        "images": images_dict,
        "per_shard_allocation": {s: allocations.get(s, 0) for s in shards_used},
        "below_floor_shards": [item["shard"] for item in all_below_floor],
    }

    sha256 = compute_manifest_sha256(manifest)
    manifest["manifest_sha256"] = sha256

    return manifest, sha256


# ═══════════════════════════════════════════════════════════════
# Verification + upload
# ═══════════════════════════════════════════════════════════════

def _verify_manifest(manifest: dict):
    """Run all verification checks per the spec."""
    total = manifest["total_images"]
    dist = manifest["class_distribution"]
    splits = manifest["split_sizes"]

    print(f"\n{'#'*72}")
    print(f"  VERIFICATION")
    print(f"{'#'*72}")

    # 1. Total
    total_avail = sum(manifest.get("per_shard_allocation", {}).values())
    print(f"\n  [1] Total images: {total:,} (target: {TARGET_TOTAL_IMAGES:,}, available: {total_avail:,})")
    if TARGET_TOTAL_IMAGES >= total_avail:
        print(f"  [1] Using ALL available images (target >= available)")
    else:
        assert abs(total - TARGET_TOTAL_IMAGES) / TARGET_TOTAL_IMAGES < 0.02, \
            f"Total {total:,} deviates >2% from target {TARGET_TOTAL_IMAGES:,}"

    # 2. Class ratios
    for cls, expected_ratio in TARGET_CLASS_RATIOS.items():
        actual = dist.get(cls, 0)
        diff = abs(actual - expected_ratio)
        status = "OK" if diff <= 0.01 else "WARN"
        print(f"  [2] {cls}: {actual:.1%} (target: {expected_ratio:.1%}) diff={diff:.3f} [{status}]")
        assert diff <= 0.02, f"{cls} ratio {actual:.1%} deviates >2% from {expected_ratio:.1%}"

    # 3. Test floor
    below_floor = manifest.get("below_floor_shards", [])
    if below_floor:
        print(f"\n  [3] Below-floor shards: {below_floor} (flagged in report)")
    else:
        print(f"\n  [3] All shards meet TEST_MIN_PER_SHARD = {TEST_MIN_PER_SHARD}")

    # 4. Split sizes
    print(f"\n  [4] Split sizes:")
    for s, n in splits.items():
        pct = n / total * 100 if total > 0 else 0
        print(f"      {s}: {n:,} ({pct:.1f}%)")

    # 5. Epoch time estimate
    train_size = splits["train"]
    images_per_sec = 50  # conservative CPU estimate
    epoch_min = train_size / images_per_sec / 60
    print(f"\n  [5] Epoch time estimate:")
    print(f"      Train split: {train_size:,} images")
    print(f"      Est. ~{epoch_min:.0f} min/epoch @ {images_per_sec} img/s (CPU)")
    print(f"      40 epochs = ~{epoch_min * 40 / 60:.0f} hours")
    if epoch_min * 40 / 60 > 10:
        print(f"      WARNING: exceeds 10-hour Kaggle limit.")

    print(f"\n{'#'*72}")
    print(f"  ALL VERIFICATION CHECKS PASSED")
    print(f"{'#'*72}")


def _upload_to_hf(manifest: dict, hf_token: str, hf_repo: str):
    """Upload rebuilt manifest to HF (overwrites existing, version stays 1)."""
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(hf_repo, repo_type="model", exist_ok=True)

    tmp = Path("/kaggle/working/split_manifest_rebuild.json")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    api.upload_file(
        path_or_fileobj=str(tmp),
        path_in_repo=MANIFEST_PATH_IN_REPO,
        repo_id=hf_repo,
        repo_type="model",
    )
    print(f"\n  Uploaded: {MANIFEST_PATH_IN_REPO} -> {hf_repo}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Rebuild the frozen split manifest (FIX 1-4)")
    parser.add_argument("--input-root", type=str, default=str(KAGGLE_INPUT_ROOT),
                        help="Root of Kaggle input mounts")
    parser.add_argument("--hf-repo", type=str, default=HF_MANIFEST_REPO,
                        help="HF repo for manifest upload")
    parser.add_argument("--run-id", type=str, default="rebuild_v1",
                        help="Run identifier for manifest metadata")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Force rebuild even if manifest exists on HF")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt (non-interactive)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build manifest but don't upload")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for sampling")
    parser.add_argument("--target-total", type=int, default=TARGET_TOTAL_IMAGES,
                        help=f"Target total images (default: {TARGET_TOTAL_IMAGES})")
    args = parser.parse_args()

    # Override globals for this run
    import rebuild_manifest as _self
    _self.SEED = args.seed
    _self.TARGET_TOTAL_IMAGES = args.target_total

    input_root = Path(args.input_root)
    print(f"  Input root: {input_root}")
    print(f"  HF repo: {args.hf_repo}")

    manifest, sha256 = build_rebuilt_manifest(
        input_root=input_root,
        run_id=args.run_id,
        force=args.force_rebuild,
    )

    _verify_manifest(manifest)

    # Confirmation
    if not args.yes:
        print(f"\n  Manifest ready for upload. sha256={sha256[:16]}...")
        resp = input("  Upload to HF? [y/N] ").strip().lower()
        if resp != "y":
            print("  Aborted.")
            return

    if not args.dry_run:
        try:
            from split_manifest_manager import _find_hf_token
            hf_token = _find_hf_token()
        except Exception:
            hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("  WARNING: No HF token found. Saving locally only.")
        else:
            _upload_to_hf(manifest, hf_token, args.hf_repo)

    # Always save locally
    local_path = Path("split_manifest_rebuilt.json")
    with open(local_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Saved locally: {local_path}")
    print(f"\n  DONE. Manifest sha256={sha256[:16]}..., total={manifest['total_images']:,}")


if __name__ == "__main__":
    main()
