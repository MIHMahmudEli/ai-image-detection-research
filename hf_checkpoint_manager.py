"""
HF Checkpoint Manager
======================
Per-run namespaced checkpoint management on HuggingFace Hub.

All paths are prefixed with runs/<run_id>/ to prevent collisions
between parallel training sessions. The shared manifest stays
at the top level.

Path layout:
    runs/<run_id>/checkpoints/last.pt
    runs/<run_id>/checkpoints/best.pt
    runs/<run_id>/checkpoints/epoch_<N>.pt
    runs/<run_id>/logs/training_state.json
    runs/<run_id>/logs/metrics.csv
    runs/<run_id>/logs/session_log.csv
    runs/<run_id>/README.md

    manifest/split_manifest.json (shared, top-level)
"""

import os
import csv
import json
import time
import shutil
import hashlib
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

HF_REPO = "studyhub991/mfft-checkpoints"
MANIFEST_REPO = "studyhub991/mfft-master-manifest"
MANIFEST_PATH = "manifest/split_manifest.json"


def _find_hf_token() -> str:
    """Auto-discover HF token from env or .env."""
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
    raise ValueError("HF_TOKEN not found")


def compute_manifest_sha256(manifest: dict) -> str:
    """SHA-256 of manifest content (excluding hash field)."""
    m = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class TrainingState:
    """Persistent training state, saved to HF after every epoch."""
    run_id: str = ""
    model_variant: str = ""
    manifest_sha256: str = ""
    manifest_version: int = 1
    current_epoch: int = 0
    total_epochs: int = 40
    best_val_macro_f1: float = 0.0
    best_val_auc: float = 0.0
    best_val_accuracy: float = 0.0
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    patience_limit: int = 8
    early_stop: bool = False
    best_model_epoch: int = 0
    total_train_time: float = 0.0
    status: str = "initialized"
    created_at: str = ""
    last_updated: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_updated:
            self.last_updated = now


@dataclass
class MetricRecord:
    """Single epoch metrics."""
    run_id: str = ""
    epoch: int = 0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    val_macro_f1: float = 0.0
    val_auc: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_specificity: float = 0.0
    learning_rate: float = 0.0
    elapsed_time: float = 0.0


class HFCheckpointManager:
    """
    Manages all HF interactions for a single training run.

    Args:
        hf_token: HuggingFace token (auto-discovered if empty)
        hf_repo: Checkpoint repo ID
        run_id: Unique run identifier (e.g. mfft-base_acct1_20250601)
        manifest_sha256: Hash of the split manifest this run uses
    """

    def __init__(
        self,
        hf_token: str = "",
        hf_repo: str = HF_REPO,
        run_id: str = "",
        manifest_sha256: str = "",
    ):
        self.hf_token = hf_token or _find_hf_token()
        self.hf_repo = hf_repo
        self.run_id = run_id
        self.manifest_sha256 = manifest_sha256

        self._ensure_repo()
        self.local_cache = Path(tempfile.mkdtemp(prefix="hf_cache_"))
        self._init_csv_log()

    # ------------------------------------------------------------------
    # Path helpers (run_id-namespaced)
    # ------------------------------------------------------------------

    @property
    def run_prefix(self) -> str:
        return f"runs/{self.run_id}"

    def _path(self, suffix: str) -> str:
        return f"{self.run_prefix}/{suffix}"

    # ------------------------------------------------------------------
    # HF repo management
    # ------------------------------------------------------------------

    def _ensure_repo(self):
        """Create HF repo if it doesn't exist."""
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.hf_token)
            api.create_repo(self.hf_repo, repo_type="model", exist_ok=True)
        except Exception as e:
            logger.warning(f"Repo check failed: {e}")

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self, state: TrainingState):
        """Save training_state.json to the run's namespace on HF."""
        state.run_id = self.run_id
        state.manifest_sha256 = self.manifest_sha256
        state.last_updated = datetime.now().isoformat()

        path = Path(self.local_cache / "training_state.json")
        with open(path, "w") as f:
            json.dump(asdict(state), f, indent=2, default=str)

        self._upload(path, "logs/training_state.json")

    def load_state(self) -> Optional[TrainingState]:
        """Download and deserialize training_state.json for this run."""
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=self._path("logs/training_state.json"),
                repo_type="model",
                token=self.hf_token,
            )
            with open(path) as f:
                data = json.load(f)
            state = TrainingState(**{
                k: v for k, v in data.items()
                if k in TrainingState.__dataclass_fields__
            })

            # Verify manifest hash matches
            if state.manifest_sha256 and state.manifest_sha256 != self.manifest_sha256:
                raise ValueError(
                    f"Manifest hash mismatch on resume! "
                    f"run expects {state.manifest_sha256[:12]}, "
                    f"current manifest is {self.manifest_sha256[:12]}. "
                    f"The manifest changed since this run started."
                )

            return state
        except Exception as e:
            if "404" in str(e) or "EntryNotFound" in str(e):
                return None
            raise

    # ------------------------------------------------------------------
    # Checkpoint upload/download
    # ------------------------------------------------------------------

    def upload_checkpoint(self, checkpoint_path: str, tag: str) -> bool:
        """Upload a checkpoint file to runs/<run_id>/checkpoints/."""
        p = Path(checkpoint_path)
        if not p.exists():
            return False

        hf_path = f"checkpoints/{tag}.pt"
        self._upload(p, hf_path)
        return True

    def download_checkpoint(self, tag: str = "last") -> Optional[Path]:
        """Download a checkpoint from this run's namespace."""
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=self._path(f"checkpoints/{tag}.pt"),
                repo_type="model",
                token=self.hf_token,
            )
            local_path = self.local_cache / f"{tag}.pt"
            shutil.copy(path, local_path)
            return local_path
        except Exception as e:
            if "404" in str(e) or "EntryNotFound" in str(e):
                return None
            raise

    # ------------------------------------------------------------------
    # Metrics logging
    # ------------------------------------------------------------------

    def _init_csv_log(self):
        """Initialize the local CSV metrics log."""
        self.csv_path = self.local_cache / "metrics.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(MetricRecord.__dataclass_fields__.keys()))
                writer.writeheader()

    def log_metrics(self, record: MetricRecord):
        """Append a metric record and upload to HF."""
        record.run_id = self.run_id
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(MetricRecord.__dataclass_fields__.keys()))
            writer.writerow(asdict(record))
        self._upload(self.csv_path, "logs/metrics.csv")

    def log_session(self, message: str):
        """Append a line to the session log."""
        log_path = self.local_cache / "session_log.csv"
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), message])
        self._upload(log_path, "logs/session_log.csv")

    # ------------------------------------------------------------------
    # Model card (per-run README)
    # ------------------------------------------------------------------

    def update_model_card(self, state: TrainingState, metrics: dict = None):
        """Write a README.md for this specific run."""
        card = f"""# {state.run_id}

## Summary

| Field | Value |
|-------|-------|
| Run ID | `{state.run_id}` |
| Model | {state.model_variant} |
| Manifest SHA-256 | `{state.manifest_sha256[:16]}...` |
| Manifest Version | {state.manifest_version} |
| Status | {state.status} |
| Current Epoch | {state.current_epoch}/{state.total_epochs} |
| Best Val Macro-F1 | {state.best_val_macro_f1:.4f} |
| Best Val AUC | {state.best_val_auc:.4f} |
| Best Val Accuracy | {state.best_val_accuracy:.4f} |
| Early Stop | {state.early_stop} |
| Patience | {state.patience_counter}/{state.patience_limit} |
| Total Train Time | {state.total_train_time:.1f}s |
| Created | {state.created_at} |
| Last Updated | {state.last_updated} |
"""
        if metrics:
            card += "\n## Latest Epoch Metrics\n\n"
            for k, v in metrics.items():
                card += f"- {k}: {v}\n"

        card += f"\n---\n*Generated by MFFT training pipeline*"

        tmp = Path(self.local_cache / "README.md")
        with open(tmp, "w") as f:
            f.write(card)
        self._upload(tmp, "README.md")

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def upload_onnx(self, onnx_path: str) -> bool:
        """Upload ONNX model to this run's namespace."""
        p = Path(onnx_path)
        if not p.exists():
            return False
        self._upload(p, "results/model.onnx")
        return True

    # ------------------------------------------------------------------
    # Internal upload helper
    # ------------------------------------------------------------------

    def _upload(self, local_path: Path, suffix: str, retries: int = 3):
        """Upload a file to runs/<run_id>/<suffix> with retry."""
        from huggingface_hub import HfApi
        api = HfApi(token=self.hf_token)
        hf_path = self._path(suffix)

        for attempt in range(retries):
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=hf_path,
                    repo_id=self.hf_repo,
                    repo_type="model",
                )
                return
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Upload failed after {retries} attempts: {e}")
                    self.log_session(f"UPLOAD_FAILED: {suffix} - {e}")
                else:
                    time.sleep(2 ** attempt * 2)

    # ------------------------------------------------------------------
    # Manifest download (for resume verification)
    # ------------------------------------------------------------------

    def download_manifest(self) -> Optional[dict]:
        """Download the shared split manifest (top-level, not namespaced)."""
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=MANIFEST_REPO,
                filename=MANIFEST_PATH,
                repo_type="model",
                token=self.hf_token,
            )
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            if "404" in str(e) or "EntryNotFound" in str(e):
                return None
            raise
