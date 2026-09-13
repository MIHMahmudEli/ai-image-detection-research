"""
Split Manifest Manager
=======================
Download-or-bootstrap-on-first-run logic for the frozen split manifest.

Every training session calls this at startup. It either:
  1. Downloads the existing manifest from HF (all runs after the first), or
  2. Bootstraps it by scanning mounted Kaggle datasets (first-ever run only)

The manifest is then frozen at manifest/split_manifest.json on HF and
never modified — all parallel runs use the same frozen split.

Usage (in any Kaggle notebook):
    from split_manifest_manager import SplitManifestManager

    mgr = SplitManifestManager(hf_token=hf_token, run_id="mfft-base_acct1_20250601")
    manifest, sha256 = mgr.get_or_bootstrap()
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

from pipeline_config import (
    HF_MANIFEST_REPO, MANIFEST_PATH_IN_REPO, KAGGLE_INPUT_ROOT,
    SEED, VAL_SPLIT, TEST_SPLIT, LABEL_MAP, CLASS_NAMES,
    IMG_EXTENSIONS, KAGGLE_DATASETS,
)

logger = logging.getLogger(__name__)


def stable_image_id(filename: str, shard: str, label: str) -> str:
    """Deterministic image ID from filename+shard+label (not filesystem path)."""
    return hashlib.sha1(f"{filename}|{shard}|{label}".encode()).hexdigest()[:20]


def compute_manifest_sha256(manifest: dict) -> str:
    """SHA-256 of the manifest JSON content (excluding the hash field itself)."""
    m = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()


def _find_hf_token() -> str:
    """Auto-discover HF token from Kaggle Secrets, env, or .env."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    for p in [Path("/kaggle/working/.env"), Path(__file__).parent / ".env"]:
        if p.exists():
            with open(p) as f:
                for line in f:
                    if line.startswith("hf="):
                        return line.strip().split("=", 1)[1]
    raise ValueError("HF_TOKEN not found. Set as Kaggle Secret or in .env")


def build_mount_index(mount_path: Path, shard: str, label: str) -> Dict[str, Path]:
    """
    Walk a mounted shard directory ONCE, building {image_id: path}.
    This is O(files_in_shard) instead of O(files_in_shard) per lookup.
    """
    index = {}
    for img_path in mount_path.rglob("*"):
        if img_path.suffix.lower() in IMG_EXTENSIONS and img_path.is_file():
            if img_path.stat().st_size == 0:
                continue
            img_id = stable_image_id(img_path.name, shard, label)
            index[img_id] = img_path
    return index


class SplitManifestManager:
    """
    Manages the frozen split manifest: download-or-bootstrap, validation,
    and path resolution for all training sessions.
    """

    def __init__(
        self,
        hf_token: str = None,
        hf_repo: str = None,
        run_id: str = "",
        input_root: str = None,
    ):
        self.hf_token = hf_token or _find_hf_token()
        self.hf_repo = hf_repo or HF_MANIFEST_REPO
        self.run_id = run_id
        self.input_root = Path(input_root) if input_root else KAGGLE_INPUT_ROOT
        self._manifest: Optional[dict] = None
        self._sha256: Optional[str] = None
        self._mount_indices: Dict[str, Dict[str, Path]] = {}

    def get_or_bootstrap(self) -> Tuple[dict, str]:
        """
        Main entry point. Returns (manifest_dict, manifest_sha256).

        If the manifest exists on HF -> download and return it.
        If it does NOT exist -> bootstrap from currently mounted datasets,
        upload to HF, and return it.
        """
        # Try download first
        try:
            manifest, sha256 = self._download_manifest()
            print(f"[SplitManifest] Found existing manifest on HF (sha256={sha256[:12]}...)")
            print(f"  Version: {manifest.get('version', '?')}, "
                  f"Images: {manifest.get('total_images', '?')}, "
                  f"Shards: {manifest.get('shards_used', [])}")
            self._manifest = manifest
            self._sha256 = sha256
            return manifest, sha256
        except Exception:
            pass

        # No manifest on HF — bootstrap
        print(f"[SplitManifest] No manifest on HF — bootstrapping from mounted datasets...")
        print(f"  Run ID: {self.run_id}")

        manifest, sha256 = self._bootstrap_manifest()
        self._upload_manifest(manifest)
        self._manifest = manifest
        self._sha256 = sha256

        print(f"[SplitManifest] Bootstrapped and uploaded! sha256={sha256[:12]}...")
        return manifest, sha256

    def download(self) -> Tuple[dict, str]:
        """Download existing manifest. Raises if not found."""
        return self._download_manifest()

    def _download_manifest(self) -> Tuple[dict, str]:
        """Download manifest from HF and verify hash."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            os.system("pip install huggingface_hub -q")
            from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=self.hf_repo,
            filename=MANIFEST_PATH_IN_REPO,
            repo_type="model",
            token=self.hf_token,
        )

        with open(path) as f:
            manifest = json.load(f)

        sha256 = compute_manifest_sha256(manifest)
        stored = manifest.get("manifest_sha256", "")
        if stored and stored != sha256:
            raise ValueError(
                f"Manifest hash mismatch! stored={stored[:12]}, computed={sha256[:12]}. "
                "The manifest file may be corrupted."
            )

        return manifest, sha256

    def _bootstrap_manifest(self) -> Tuple[dict, str]:
        """Scan mounted datasets, build stratified split, return manifest."""
        from sklearn.model_selection import train_test_split

        # 0. Discover actual mount points and match to expected slugs
        input_root = self.input_root
        print(f"  Scanning {input_root} for mounted datasets...")
        if input_root.exists():
            actual_mounts = [d.name for d in input_root.iterdir() if d.is_dir()]
            print(f"  Found {len(actual_mounts)} mount(s): {actual_mounts[:15]}{'...' if len(actual_mounts) > 15 else ''}")
            # Kaggle often nests attached datasets under /kaggle/input/datasets/
            datasets_dir = input_root / "datasets"
            if datasets_dir.exists():
                nested = [d.name for d in datasets_dir.iterdir() if d.is_dir()]
                print(f"  Found {len(nested)} dataset(s) under {datasets_dir}: {nested[:15]}{'...' if len(nested) > 15 else ''}")
                actual_mounts.extend(nested)
        else:
            actual_mounts = []
            print(f"  {input_root} does not exist")

        # Build a mapping: expected_slug -> actual_mount_name
        # Kaggle mount names can be "owner-slug", "slug", "datasets/slug", or just "slug"
        datasets_dir = input_root / "datasets"
        slug_map = {}  # expected_slug -> actual_mount_dir
        for slug in KAGGLE_DATASETS:
            # Try exact match at top level
            if slug in [m for m in actual_mounts if not m.startswith("datasets")]:
                # Could be at top level or inside datasets/
                top = input_root / slug
                nested = datasets_dir / slug
                slug_map[slug] = top if top.exists() else nested
                continue
            # Try exact match inside datasets/
            if datasets_dir.exists():
                if slug in [d.name for d in datasets_dir.iterdir() if d.is_dir()]:
                    slug_map[slug] = datasets_dir / slug
                    continue
            # Try partial match (e.g. "mihmahmud-stable-diffusion" matches "stable-diffusion")
            matches = [m for m in actual_mounts if slug in m or m in slug]
            if len(matches) == 1:
                # Resolve: could be top-level or nested
                candidate = input_root / matches[0]
                if not candidate.exists() and datasets_dir.exists():
                    candidate = datasets_dir / matches[0]
                slug_map[slug] = candidate
                print(f"  Mapped {slug} -> {matches[0]}")
            elif len(matches) > 1:
                # Pick shortest match (most specific)
                best = min(matches, key=len)
                candidate = input_root / best
                if not candidate.exists() and datasets_dir.exists():
                    candidate = datasets_dir / best
                slug_map[slug] = candidate
                print(f"  Mapped {slug} -> {best} (from {matches})")
            # else: not mounted

        # 1. Scan all mounted datasets
        rows = []
        shards_used = []
        for slug, (label, subdir) in KAGGLE_DATASETS.items():
            if slug not in slug_map:
                print(f"  SKIP {slug}: not mounted")
                continue

            mount_dir = slug_map[slug]
            shards_used.append(slug)

            # Try configured subdir, then fallback to mount root
            image_root = mount_dir / subdir
            if not image_root.exists():
                # Try auto-discovery: look for dirs containing images
                image_root = mount_dir

            count = 0
            for img_path in image_root.rglob("*"):
                if img_path.suffix.lower() in IMG_EXTENSIONS and img_path.is_file():
                    if img_path.stat().st_size == 0:
                        continue
                    img_id = stable_image_id(img_path.name, slug, label)
                    rows.append({
                        "image_id": img_id,
                        "shard": slug,
                        "label": label,
                        "label_int": LABEL_MAP[label],
                    })
                    count += 1
            print(f"  {slug}: {count} images ({label})")

        if not rows:
            # Detailed diagnostics
            print("\n  DIAGNOSTICS:")
            for slug, (label, subdir) in KAGGLE_DATASETS.items():
                if slug in slug_map:
                    mount_dir = slug_map[slug]
                    contents = list(mount_dir.iterdir())[:8]
                    print(f"  {slug} -> {mount_dir}")
                    for c in contents:
                        kind = "dir" if c.is_dir() else "file"
                        print(f"    [{kind}] {c.name}")
                    target = mount_dir / subdir
                    if target.exists():
                        inner = list(target.iterdir())[:5]
                        print(f"    /{subdir}/ -> {[x.name for x in inner]}")
                    else:
                        print(f"    /{subdir}/ DOES NOT EXIST")
                else:
                    print(f"  {slug}: NOT MOUNTED")
            raise RuntimeError("No images found in any mounted dataset. See diagnostics above.")

        print(f"  Total: {len(rows)} images across {len(shards_used)} shards")

        # 2. Stratified train/val/test split
        labels = [r["label_int"] for r in rows]
        holdout = VAL_SPLIT + TEST_SPLIT

        train_val_idx, test_idx = train_test_split(
            list(range(len(rows))), test_size=holdout,
            stratify=labels, random_state=SEED,
        )
        rv_labels = [labels[i] for i in train_val_idx]
        val_ratio = VAL_SPLIT / holdout
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_ratio,
            stratify=rv_labels, random_state=SEED,
        )

        split_map = {}
        for i in train_idx:
            split_map[i] = "train"
        for i in val_idx:
            split_map[i] = "val"
        for i in test_idx:
            split_map[i] = "test"

        for i, row in enumerate(rows):
            row["split"] = split_map[i]

        # 3. Compute class distribution (ratios across all data)
        total = len(rows)
        class_dist = {}
        for lbl_name, lbl_int in LABEL_MAP.items():
            class_dist[lbl_name] = round(sum(1 for r in rows if r["label_int"] == lbl_int) / total, 4)

        # 4. Build images dict
        images_dict = {}
        for row in rows:
            images_dict[row["image_id"]] = {
                "shard": row["shard"],
                "label": row["label"],
                "label_int": row["label_int"],
                "split": row["split"],
            }

        # 5. Assemble manifest
        manifest = {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "created_by_run": self.run_id,
            "seed": SEED,
            "shards_used": shards_used,
            "total_images": total,
            "class_distribution": class_dist,
            "split_sizes": {
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
            },
            "images": images_dict,
        }

        sha256 = compute_manifest_sha256(manifest)
        manifest["manifest_sha256"] = sha256

        # Print summary
        for s in ["train", "val", "test"]:
            n = manifest["split_sizes"][s]
            dist = {}
            for row in rows:
                if row["split"] == s:
                    dist[row["label"]] = dist.get(row["label"], 0) + 1
            print(f"  {s}: {n} | {dist}")

        return manifest, sha256

    def _upload_manifest(self, manifest: dict):
        """Upload manifest to the FIXED shared path on HF."""
        from huggingface_hub import HfApi

        api = HfApi(token=self.hf_token)
        api.create_repo(self.hf_repo, repo_type="model", exist_ok=True)

        # Write to temp, upload
        tmp = Path("/kaggle/working/split_manifest_upload.json")
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        api.upload_file(
            path_or_fileobj=str(tmp),
            path_in_repo=MANIFEST_PATH_IN_REPO,
            repo_id=self.hf_repo,
            repo_type="model",
        )
        print(f"  Uploaded: {MANIFEST_PATH_IN_REPO} -> {self.hf_repo}")

    # ------------------------------------------------------------------
    # Mount index building (BUG 1 fix)
    # ------------------------------------------------------------------

    def build_all_mount_indices(self, manifest: dict):
        """
        Build per-shard lookup indices ONCE. Call this before resolve_path().
        Each index maps image_id -> Path, built by walking the mount dir once.
        """
        self._mount_indices = {}
        shards_needed = set(info["shard"] for info in manifest["images"].values())

        for shard in shards_needed:
            if shard not in KAGGLE_DATASETS:
                continue
            label, subdir = KAGGLE_DATASETS[shard]
            mount_dir = self.input_root / shard
            if not mount_dir.exists():
                # Try inside datasets/ subdirectory
                datasets_dir = self.input_root / "datasets"
                if datasets_dir.exists():
                    mount_dir = datasets_dir / shard
                if not mount_dir.exists():
                    # Fuzzy match: find mount containing shard name
                    if self.input_root.exists():
                        actual = [d.name for d in self.input_root.iterdir() if d.is_dir()]
                        if datasets_dir.exists():
                            actual += [d.name for d in datasets_dir.iterdir() if d.is_dir()]
                        matches = [m for m in actual if shard in m or m in shard]
                        if matches:
                            best = min(matches, key=len)
                            candidate = self.input_root / best
                            if not candidate.exists() and datasets_dir.exists():
                                candidate = datasets_dir / best
                            mount_dir = candidate
                        else:
                            continue
                    else:
                        continue

            image_root = mount_dir / subdir
            if not image_root.exists():
                image_root = mount_dir

            idx = build_mount_index(image_root, shard, label)
            self._mount_indices[shard] = idx
            print(f"  Index built: {shard} -> {len(idx)} images")

    def resolve_path(self, image_id: str, info: dict) -> Optional[Path]:
        """
        O(1) lookup via pre-built mount index.
        build_all_mount_indices() must be called first.
        """
        shard = info["shard"]
        idx = self._mount_indices.get(shard)
        if idx is not None:
            return idx.get(image_id)
        return None

    def verify_against_manifest(
        self, resolved_count: int, split_name: str, manifest: dict
    ):
        """Sanity check that resolved data roughly matches manifest expectations."""
        expected = manifest["split_sizes"].get(split_name, 0)
        if expected == 0:
            return
        ratio = resolved_count / expected
        if ratio < 0.8:
            logger.warning(
                f"  {split_name}: only {resolved_count}/{expected} images resolved "
                f"({ratio:.0%}). Some datasets may not be mounted."
            )
