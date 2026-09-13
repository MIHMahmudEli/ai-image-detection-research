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

logger = logging.getLogger(__name__)

HF_MANIFEST_REPO = "studyhub991/mfft-master-manifest"
MANIFEST_PATH = "manifest/split_manifest.json"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")

SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

LABEL_MAP = {"real": 0, "ai_generated": 1, "deepfake": 2}
CLASS_NAMES = ["real", "ai_generated", "deepfake"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Kaggle mount slug -> (label, image_subdir inside the mount)
KAGGLE_DATASETS = {
    "stable-diffusion":                 ("ai_generated", "Stable Diffusion/images"),
    "places365":                        ("real",          "train"),
    "open-images-v7-dataset":           ("real",          "Open-Images-V7-Dataset/open-images-v7/train/images"),
    "ntire2026":                        ("real",          "NTIRE2026"),
    "midjourney":                       ("ai_generated",  "Midjourney/Datasetfordream"),
    "mfft-real":                        ("real",          "Mfft_real"),
    "genimage-ai":                      ("ai_generated",  "genimage_ai"),
    "faceforensics":                    ("deepfake",      "cropped_images"),
    "dfdc-faces-of-the-train-sample":   ("deepfake",      "train/fake"),
    "dall-e3":                          ("ai_generated",  "DALL-E3"),
    "celebdf-v2image-dataset":          ("deepfake",      "Celeb_V2"),
}


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


class SplitManifestManager:
    """
    Manages the frozen split manifest: download-or-bootstrap, validation,
    and path resolution for all training sessions.
    """

    def __init__(
        self,
        hf_token: str = None,
        hf_repo: str = HF_MANIFEST_REPO,
        run_id: str = "",
        input_root: str = None,
    ):
        self.hf_token = hf_token or _find_hf_token()
        self.hf_repo = hf_repo
        self.run_id = run_id
        self.input_root = Path(input_root) if input_root else KAGGLE_INPUT_ROOT
        self._manifest: Optional[dict] = None
        self._sha256: Optional[str] = None

    def get_or_bootstrap(self) -> Tuple[dict, str]:
        """
        Main entry point. Returns (manifest_dict, manifest_sha256).

        If the manifest exists on HF → download and return it.
        If it does NOT exist → bootstrap from currently mounted datasets,
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
            filename=MANIFEST_PATH,
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

        # 1. Scan all mounted datasets
        rows = []
        shards_used = []
        for slug, (label, subdir) in KAGGLE_DATASETS.items():
            mount_dir = self.input_root / slug
            if not mount_dir.exists():
                print(f"  SKIP {slug}: not mounted")
                continue

            shards_used.append(slug)
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
            raise RuntimeError("No images found in any mounted dataset")

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
            path_in_repo=MANIFEST_PATH,
            repo_id=self.hf_repo,
            repo_type="model",
        )
        print(f"  Uploaded: {MANIFEST_PATH} -> {self.hf_repo}")

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def discover_mounts(self) -> Dict[str, Path]:
        """Discover all mounted Kaggle datasets."""
        mounts = {}
        if not self.input_root.exists():
            return mounts
        for entry in sorted(self.input_root.iterdir()):
            if entry.is_dir():
                mounts[entry.name] = entry
        return mounts

    def resolve_path(self, image_id: str, manifest: dict) -> Optional[Path]:
        """
        Resolve an image path from the manifest against mounted datasets.
        Uses path_hint as a guide but does fuzzy search as fallback.
        """
        info = manifest["images"].get(image_id)
        if info is None:
            return None

        mounts = self.discover_mounts()
        shard = info["shard"]

        if shard in mounts:
            mount = mounts[shard]
            # Try rglob by filename (fast enough for most shards)
            for img in mount.rglob("*"):
                if img.suffix.lower() in IMG_EXTENSIONS and img.is_file():
                    if stable_image_id(img.name, shard, info["label"]) == image_id:
                        return img

        # Fallback: search all mounts
        for mount in mounts.values():
            for img in mount.rglob("*"):
                if img.suffix.lower() in IMG_EXTENSIONS and img.is_file():
                    if stable_image_id(img.name, shard, info["label"]) == image_id:
                        return img

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
