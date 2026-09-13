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
import re
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Set

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


# ─────────────────────────────────────────────────────────────
# Mount overrides (FIX 3)
# ─────────────────────────────────────────────────────────────

def _load_mount_overrides() -> Dict[str, str]:
    """
    Load MOUNT_OVERRIDES mapping from Kaggle Secret, env var, or local file.
    Priority: Kaggle Secret > env var > /kaggle/working/mount_overrides.json
    Returns dict mapping expected_slug -> actual_mount_name.
    """
    raw = None

    # 1. Kaggle Secret
    try:
        from kaggle_secrets import UserSecretsClient
        raw = UserSecretsClient().get_secret("MOUNT_OVERRIDES_JSON")
    except Exception:
        pass

    # 2. Env var
    if raw is None:
        raw = os.environ.get("MOUNT_OVERRIDES_JSON")

    # 3. Local file
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
            print(f"  [MountOverrides] Loaded {len(overrides)} override(s): {overrides}")
        return overrides
    except json.JSONDecodeError as e:
        print(f"  [MountOverrides] WARNING: Invalid JSON: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# Layered slug matching (FIX 2)
# ─────────────────────────────────────────────────────────────

_SLUG_NORMALIZE_RE = re.compile(r"[-_\s]+")

def _normalize_slug(s: str) -> str:
    """Lowercase, strip accents, replace [-_ ] with single hyphen."""
    return _SLUG_NORMALIZE_RE.sub("-", s.lower().strip())


def _tokenize(s: str) -> Set[str]:
    """Split a slug into alphanumeric tokens."""
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _token_overlap_score(a: str, b: str) -> float:
    """Fraction of tokens in `a` that also appear in `b` (0.0 – 1.0)."""
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def _find_mount_path(input_root: Path, mount_name: str) -> Optional[Path]:
    """Resolve a mount name to a real Path, checking top-level and datasets/."""
    candidates = [
        input_root / mount_name,
        input_root / "datasets" / mount_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def match_mount(
    slug: str,
    actual_mounts: List[str],
    input_root: Path,
) -> Tuple[Optional[Path], str, List[str]]:
    """
    Layered matching for a single expected slug against actual mount names.

    Returns:
        (matched_path, match_method, did_you_mean_candidates)
        matched_path is None if no match found.
    """
    datasets_dir = input_root / "datasets"

    # Collect all candidate names (top-level + nested)
    all_candidates = list(set(actual_mounts))

    # Layer 1: Exact match
    for cand in all_candidates:
        if slug == cand:
            path = _find_mount_path(input_root, cand)
            if path:
                return path, "exact", []

    # Layer 2: Case-insensitive match
    slug_lower = slug.lower()
    for cand in all_candidates:
        if slug_lower == cand.lower():
            path = _find_mount_path(input_root, cand)
            if path:
                return path, "case-insensitive", []

    # Layer 3: Normalized match (replace [-_ ] with single separator)
    slug_norm = _normalize_slug(slug)
    for cand in all_candidates:
        if slug_norm == _normalize_slug(cand):
            path = _find_mount_path(input_root, cand)
            if path:
                return path, "normalized", []

    # Layer 4: Contiguous substring match (original behavior)
    for cand in all_candidates:
        if slug in cand or cand in slug:
            path = _find_mount_path(input_root, cand)
            if path:
                return path, "substring", []

    # Layer 5: Token-overlap match (all tokens of slug appear in cand)
    best_token_match = None
    best_score = 0.0
    did_you_mean = []
    for cand in all_candidates:
        score = _token_overlap_score(slug, cand)
        if score > 0:
            did_you_mean.append(f"{cand} (overlap={score:.0%})")
        if score >= 0.5 and score > best_score:
            best_score = score
            best_token_match = cand

    if best_token_match:
        path = _find_mount_path(input_root, best_token_match)
        if path:
            return path, f"token-overlap({best_score:.0%})", did_you_mean

    return None, "no-match", did_you_mean


def _print_mount_tree(input_root: Path, max_depth: int = 3):
    """FIX 1: Print complete recursive listing of input_root, unconditionally."""
    print(f"\n{'='*60}")
    print(f"  MOUNT TREE: {input_root}")
    print(f"{'='*60}")

    if not input_root.exists():
        print(f"  (directory does not exist)")
        return

    def _count_filesRecursive(d: Path) -> int:
        try:
            return sum(1 for _ in d.rglob("*") if _.is_file())
        except (PermissionError, OSError):
            return -1

    def _print_tree(d: Path, prefix: str, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except (PermissionError, OSError):
            return

        for entry in entries[:50]:  # Cap at 50 entries per level
            if entry.is_dir():
                try:
                    n_children = len(list(entry.iterdir()))
                except (PermissionError, OSError):
                    n_children = -1
                print(f"  {prefix}{entry.name}/ ({n_children} entries)")
                _print_tree(entry, prefix + "  ", depth + 1)
            else:
                try:
                    size = entry.stat().st_size
                    if size > 1_000_000:
                        size_str = f"{size/1e6:.1f}MB"
                    elif size > 1_000:
                        size_str = f"{size/1e3:.0f}KB"
                    else:
                        size_str = f"{size}B"
                except OSError:
                    size_str = "?"
                print(f"  {prefix}{entry.name} ({size_str})")

        if len(list(d.iterdir())) > 50:
            print(f"  {prefix}... (truncated)")

    _print_tree(input_root, "", 0)
    print(f"{'='*60}\n")


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

        # ── FIX 1: Print mount tree unconditionally ──
        _print_mount_tree(self.input_root)

        # ── Load overrides (FIX 3) ──
        overrides = _load_mount_overrides()

        # ── Discover actual mount names ──
        input_root = self.input_root
        all_mount_names: List[str] = []
        if input_root.exists():
            all_mount_names = [d.name for d in input_root.iterdir() if d.is_dir()]
            datasets_dir = input_root / "datasets"
            if datasets_dir.exists():
                all_mount_names += [d.name for d in datasets_dir.iterdir() if d.is_dir()]
            all_mount_names = list(set(all_mount_names))

        print(f"  Actual mounts found: {all_mount_names}")

        # ── Match each expected slug (FIX 2 + FIX 3) ──
        slug_map: Dict[str, Path] = {}
        match_info: Dict[str, dict] = {}

        for slug in KAGGLE_DATASETS:
            # FIX 3: Override takes priority
            if slug in overrides:
                actual_name = overrides[slug]
                path = _find_mount_path(input_root, actual_name)
                if path:
                    slug_map[slug] = path
                    match_info[slug] = {"method": "override", "path": str(path), "did_you_mean": []}
                    print(f"  {slug}: OVERRIDDEN -> {actual_name} ({path})")
                    continue
                else:
                    print(f"  {slug}: OVERRIDDEN -> {actual_name} but path not found!")

            # FIX 2: Layered matching
            matched_path, method, did_you_mean = match_mount(slug, all_mount_names, input_root)
            match_info[slug] = {"method": method, "path": str(matched_path) if matched_path else None, "did_you_mean": did_you_mean}
            if matched_path:
                slug_map[slug] = matched_path
                print(f"  {slug}: MATCHED ({method}) -> {matched_path.name}")
            else:
                candidates_str = f" (did you mean: {', '.join(did_you_mean[:3])})" if did_you_mean else ""
                print(f"  {slug}: NOT MATCHED{candidates_str}")

        # ── Scan all matched datasets ──
        rows = []
        shards_used = []
        for slug, (label, subdir) in KAGGLE_DATASETS.items():
            if slug not in slug_map:
                continue

            mount_dir = slug_map[slug]
            shards_used.append(slug)

            # Try configured subdir, then fallback to mount root
            image_root = mount_dir / subdir
            if not image_root.exists():
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
            # Detailed failure diagnostics
            print(f"\n{'='*60}")
            print("  MOUNT MATCHING SUMMARY")
            print(f"{'='*60}")
            for slug in KAGGLE_DATASETS:
                info = match_info.get(slug, {})
                method = info.get("method", "N/A")
                path = info.get("path", "N/A")
                dym = info.get("did_you_mean", [])
                status = f"MATCHED ({method})" if path and path != "None" else "NOT MATCHED"
                print(f"  {slug:35s} {status}")
                if path and path != "None":
                    print(f"    -> {path}")
                if dym:
                    print(f"    Did you mean: {', '.join(dym[:3])}")
            print(f"{'='*60}")
            print("\n  If auto-matching fails, create a MOUNT_OVERRIDES_JSON Kaggle Secret:")
            print('  {"expected-slug": "actual-mount-folder-name"}')
            print(f"{'='*60}\n")
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
    # Mount index building
    # ------------------------------------------------------------------

    def _resolve_shard_path(self, shard: str) -> Optional[Path]:
        """Resolve a shard name to its mount path using the same layered matching."""
        overrides = _load_mount_overrides()

        # Override first
        if shard in overrides:
            path = _find_mount_path(self.input_root, overrides[shard])
            if path:
                return path

        # Discover mounts
        all_mount_names: List[str] = []
        if self.input_root.exists():
            all_mount_names = [d.name for d in self.input_root.iterdir() if d.is_dir()]
            datasets_dir = self.input_root / "datasets"
            if datasets_dir.exists():
                all_mount_names += [d.name for d in datasets_dir.iterdir() if d.is_dir()]
            all_mount_names = list(set(all_mount_names))

        matched_path, _, _ = match_mount(shard, all_mount_names, self.input_root)
        return matched_path

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

            mount_dir = self._resolve_shard_path(shard)
            if mount_dir is None:
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
