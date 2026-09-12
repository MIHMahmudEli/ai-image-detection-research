"""
HuggingFace Hub Checkpoint Manager
====================================
Resumable, multi-session training persistence layer using HuggingFace Hub.

Handles:
- Save/load local checkpoints
- Upload/download to HF repo with retry + exponential backoff
- Atomic writes (never leave training_state.json half-written)
- Resume from last epoch across Kaggle session disconnects
- Session logging (GPU-hours, timestamps)
- 4 tracked metrics: val_macro_f1, val_auc, val_loss, val_accuracy

HF Repo Structure:
    /checkpoints/last.pt          # overwritten every epoch
    /checkpoints/best.pt          # overwritten on improvement
    /checkpoints/epoch_<N>.pt     # periodic snapshots (never overwritten)
    /logs/training_state.json     # single source of truth for resume
    /logs/metrics.csv             # per-epoch metrics history
    /logs/session_log.csv         # session start/end + GPU-hours
    /results/                     # final artifacts

Usage:
    from hf_checkpoint_manager import HFCheckpointManager

    mgr = HFCheckpointManager(
        repo_id="username/mfft-checkpoints",
        hf_token=os.environ["HF_TOKEN"],
    )

    # Resume
    state = mgr.resume()
    if state:
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])

    # After each epoch
    mgr.save_and_upload(
        epoch=5,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={...},
        is_best=True,
    )
"""

import os
import io
import json
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Single source of truth for training progress."""
    epoch: int = 0
    best_epoch: int = 0
    best_val_macro_f1: float = 0.0
    best_val_auc: float = 0.0
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0
    epochs_since_improvement: int = 0
    patience: int = 8
    status: str = "running"  # running | completed | early_stopped
    reason: str = ""
    git_sha: str = ""
    timestamp: str = ""
    total_gpu_minutes: float = 0.0

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "best_epoch": self.best_epoch,
            "best_val_macro_f1": self.best_val_macro_f1,
            "best_val_auc": self.best_val_auc,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "epochs_since_improvement": self.epochs_since_improvement,
            "patience": self.patience,
            "status": self.status,
            "reason": self.reason,
            "git_sha": self.git_sha,
            "timestamp": self.timestamp,
            "total_gpu_minutes": self.total_gpu_minutes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class HFCheckpointManager:
    """
    Manages checkpoint save/load/upload/download for resumable multi-session
    Kaggle training via HuggingFace Hub.
    """

    def __init__(
        self,
        repo_id: str,
        hf_token: str,
        local_dir: str = "/kaggle/working/mfft_training",
        git_sha: str = "",
        max_retries: int = 5,
        base_retry_delay: float = 2.0,
    ):
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.git_sha = git_sha
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay

        self.checkpoints_dir = self.local_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.logs_dir = self.local_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.results_dir = self.local_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

        self._api = None
        self._session_start = time.time()
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def api(self):
        """Lazy-initialize HfApi."""
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.hf_token)
        return self._api

    # ------------------------------------------------------------------
    # Local save/load
    # ------------------------------------------------------------------

    def save_checkpoint_local(
        self,
        path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        metrics: dict,
        state: TrainingState,
        scaler=None,
        rng_state=None,
    ) -> Path:
        """Save full checkpoint locally."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "metrics": metrics,
            "state": state.to_dict(),
            "git_sha": self.git_sha,
            "timestamp": datetime.now().isoformat(),
        }
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        if rng_state is not None:
            checkpoint["rng_state"] = rng_state

        torch.save(checkpoint, path)
        return path

    def load_checkpoint_local(self, path: Path) -> Optional[dict]:
        """Load checkpoint from local path."""
        if not path.exists():
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {path}: {e}")
            return None

    def save_training_state_local(self, state: TrainingState) -> Path:
        """Atomically save training_state.json locally."""
        state.timestamp = datetime.now().isoformat()
        state_dict = state.to_dict()

        # Write to temp file, then rename (atomic on most filesystems)
        tmp_path = self.logs_dir / "training_state.json.tmp"
        final_path = self.logs_dir / "training_state.json"

        with open(tmp_path, "w") as f:
            json.dump(state_dict, f, indent=2)

        # Verify write
        with open(tmp_path, "r") as f:
            verified = json.load(f)
        assert verified == state_dict, "Verification failed"

        tmp_path.rename(final_path)
        return final_path

    def load_training_state_local(self) -> Optional[TrainingState]:
        """Load training state from local file."""
        path = self.logs_dir / "training_state.json"
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                d = json.load(f)
            return TrainingState.from_dict(d)
        except Exception as e:
            logger.error(f"Failed to load training state: {e}")
            return None

    def append_metrics_row(self, metrics: dict):
        """Append a row to local metrics.csv."""
        csv_path = self.logs_dir / "metrics.csv"
        import csv
        write_header = not csv_path.exists()

        fieldnames = [
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_accuracy", "val_precision", "val_recall",
            "val_f1", "val_macro_f1", "val_auc",
            "learning_rate", "timestamp",
        ]

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

    # ------------------------------------------------------------------
    # HuggingFace Hub upload/download (with retry)
    # ------------------------------------------------------------------

    def _retry_operation(self, operation, description: str):
        """Execute operation with exponential backoff retry."""
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return operation()
            except Exception as e:
                last_error = e
                delay = self.base_retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed for {description}: {e}"
                )
                if attempt < self.max_retries:
                    logger.info(f"  Retrying in {delay:.1f}s...")
                    time.sleep(delay)

        raise RuntimeError(
            f"Failed {description} after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def upload_file(self, local_path: Path, repo_path: str) -> Any:
        """Upload a single file to HF repo with retry."""
        def _upload():
            return self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=self.repo_id,
                repo_type="model",
            )
        return self._retry_operation(_upload, f"upload {repo_path}")

    def upload_folder(self, local_dir: Path, repo_path: str) -> Any:
        """Upload a folder to HF repo with retry."""
        def _upload():
            return self.api.upload_folder(
                folder_path=str(local_dir),
                path_in_repo=repo_path,
                repo_id=self.repo_id,
                repo_type="model",
            )
        return self._retry_operation(_upload, f"upload folder {repo_path}")

    def download_file(self, repo_path: str, local_path: Path) -> bool:
        """Download a single file from HF repo with retry."""
        def _download():
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=self.repo_id,
                filename=repo_path,
                local_dir=str(self.local_dir.parent),
                repo_type="model",
                token=self.hf_token,
            )
            # Move from repo cache to our local_dir
            cached = self.local_dir.parent / repo_path
            if cached.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                cached.rename(local_path)
                return True
            return False
        try:
            return self._retry_operation(_download, f"download {repo_path}")
        except Exception as e:
            logger.debug(f"File not found on HF: {repo_path}")
            return False

    # ------------------------------------------------------------------
    # Resume logic (idempotent)
    # ------------------------------------------------------------------

    def resume(self) -> Optional[Tuple[dict, TrainingState]]:
        """
        Attempt to resume from HF hub.
        Returns (checkpoint_dict, training_state) or None if fresh start.
        This is idempotent — re-running always picks up exactly where left off.
        """
        logger.info("Attempting to resume from HuggingFace Hub...")

        # 1. Download training_state.json
        state_path = self.logs_dir / "training_state.json"
        state_downloaded = self.download_file("logs/training_state.json", state_path)

        if not state_downloaded:
            logger.info("No previous training state found. Starting fresh run.")
            return None

        # 2. Load state
        state = self.load_training_state_local()
        if state is None:
            logger.warning("Corrupt training state. Starting fresh run.")
            return None

        if state.status in ("completed", "early_stopped"):
            logger.info(
                f"Previous run already {state.status} at epoch {state.epoch}. "
                "Starting fresh run."
            )
            return None

        # 3. Download last checkpoint
        last_path = self.checkpoints_dir / "last.pt"
        checkpoint_downloaded = self.download_file("checkpoints/last.pt", last_path)

        if not checkpoint_downloaded:
            logger.warning(
                f"Training state exists (epoch {state.epoch}) but checkpoint missing. "
                "Starting fresh run."
            )
            return None

        # 4. Load checkpoint
        checkpoint = self.load_checkpoint_local(last_path)
        if checkpoint is None:
            logger.warning("Corrupt checkpoint. Starting fresh run.")
            return None

        logger.info(
            f"Resumed from epoch {state.epoch} | "
            f"best_val_macro_f1={state.best_val_macro_f1:.4f} | "
            f"epochs_since_improvement={state.epochs_since_improvement}"
        )

        return checkpoint, state

    # ------------------------------------------------------------------
    # Per-epoch save + upload sequence
    # ------------------------------------------------------------------

    def save_and_upload(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        metrics: dict,
        state: TrainingState,
        scaler=None,
    ) -> TrainingState:
        """
        Full per-epoch save + upload sequence:
        1. Save local checkpoint
        2. Upload as checkpoints/last.pt (overwrite)
        3. If epoch % 10 == 0: upload as checkpoints/epoch_<N>.pt
        4. If improved: upload as checkpoints/best.pt
        5. Append to metrics.csv
        6. Update + upload training_state.json
        """
        rng_state = {
            "python": None,  # captured externally if needed
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        # 1. Save local last.pt
        last_path = self.checkpoints_dir / "last.pt"
        self.save_checkpoint_local(
            last_path, model, optimizer, scheduler, epoch, metrics, state, scaler, rng_state,
        )
        logger.info(f"  Saved local: {last_path}")

        # 2. Upload last.pt (overwrite)
        self.upload_file(last_path, "checkpoints/last.pt")
        logger.info("  Uploaded: checkpoints/last.pt")

        # 3. Periodic snapshot
        if epoch % 10 == 0:
            snap_path = self.checkpoints_dir / f"epoch_{epoch}.pt"
            self.save_checkpoint_local(
                snap_path, model, optimizer, scheduler, epoch, metrics, state, scaler, rng_state,
            )
            self.upload_file(snap_path, f"checkpoints/epoch_{epoch}.pt")
            logger.info(f"  Uploaded: checkpoints/epoch_{epoch}.pt")

        # 4. Best checkpoint
        is_better = (
            metrics.get("val_macro_f1", 0) > state.best_val_macro_f1
            or (
                metrics.get("val_macro_f1", 0) == state.best_val_macro_f1
                and metrics.get("val_auc", 0) > state.best_val_auc
            )
        )
        if is_better:
            state.best_epoch = epoch
            state.best_val_macro_f1 = metrics["val_macro_f1"]
            state.best_val_auc = metrics.get("val_auc", 0)
            state.best_val_loss = metrics.get("val_loss", 0)
            state.best_val_accuracy = metrics.get("val_accuracy", 0)
            state.epochs_since_improvement = 0

            best_path = self.checkpoints_dir / "best.pt"
            self.save_checkpoint_local(
                best_path, model, optimizer, scheduler, epoch, metrics, state, scaler, rng_state,
            )
            self.upload_file(best_path, "checkpoints/best.pt")
            logger.info(
                f"  New best (macro_f1={state.best_val_macro_f1:.4f}). "
                "Uploaded: checkpoints/best.pt"
            )
        else:
            state.epochs_since_improvement += 1

        # 5. Append metrics
        metrics_row = {
            "epoch": epoch,
            "train_loss": metrics.get("train_loss", 0),
            "train_acc": metrics.get("train_acc", 0),
            "val_loss": metrics.get("val_loss", 0),
            "val_accuracy": metrics.get("val_accuracy", 0),
            "val_precision": metrics.get("val_precision", 0),
            "val_recall": metrics.get("val_recall", 0),
            "val_f1": metrics.get("val_f1", 0),
            "val_macro_f1": metrics.get("val_macro_f1", 0),
            "val_auc": metrics.get("val_auc", 0),
            "learning_rate": metrics.get("learning_rate", 0),
            "timestamp": datetime.now().isoformat(),
        }
        self.append_metrics_row(metrics_row)
        self.upload_file(self.logs_dir / "metrics.csv", "logs/metrics.csv")

        # 6. Update training state
        state.epoch = epoch
        self.save_training_state_local(state)
        self.upload_file(self.logs_dir / "training_state.json", "logs/training_state.json")
        logger.info(f"  Uploaded: logs/training_state.json")

        return state

    # ------------------------------------------------------------------
    # Final upload
    # ------------------------------------------------------------------

    def upload_final_artifacts(
        self,
        model: torch.nn.Module,
        state: TrainingState,
        final_metrics: dict,
        figures_dir: Optional[Path] = None,
        manuscript_metrics: Optional[dict] = None,
    ):
        """
        Upload all final artifacts after training completes.
        """
        logger.info("\nUploading final artifacts to HuggingFace Hub...")

        # Best model weights
        best_path = self.checkpoints_dir / "best.pt"
        if best_path.exists():
            self.upload_file(best_path, "checkpoints/best.pt")

        # Final model weights
        final_path = self.results_dir / "final_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "state": state.to_dict(),
            "final_metrics": final_metrics,
        }, final_path)
        self.upload_file(final_path, "results/final_model.pt")

        # ONNX export (if feasible)
        try:
            self._export_onnx(model, self.results_dir / "model.onnx")
            self.upload_file(self.results_dir / "model.onnx", "results/model.onnx")
            logger.info("  Uploaded ONNX export")
        except Exception as e:
            logger.warning(f"  ONNX export skipped: {e}")

        # Metrics JSON
        metrics_path = self.results_dir / "final_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)
        self.upload_file(metrics_path, "results/final_metrics.json")

        # Manuscript-ready metrics
        if manuscript_metrics:
            mr_path = self.results_dir / "manuscript_ready_metrics.json"
            with open(mr_path, "w") as f:
                json.dump(manuscript_metrics, f, indent=2)
            self.upload_file(mr_path, "results/manuscript_ready_metrics.json")

        # Figures
        if figures_dir and figures_dir.exists():
            for fig_path in figures_dir.glob("*.png"):
                self.upload_file(fig_path, f"results/figures/{fig_path.name}")

        # Session log
        self._log_session_end(state)

        # Update model card (README.md)
        self._update_model_card(state, final_metrics)

        logger.info("All final artifacts uploaded.")

    def _export_onnx(self, model: torch.nn.Module, path: Path):
        """Export model to ONNX format."""
        model.eval()
        dummy = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
        torch.onnx.export(
            model, dummy, str(path),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )

    def _update_model_card(self, state: TrainingState, final_metrics: dict):
        """Update the HF repo README.md with training summary."""
        readme_content = f"""---
tags:
- mfft
- ai-image-detection
- frequency-analysis
---

# MFFT Checkpoints

## Training Status: {state.status.upper()}

| Metric | Value |
|---|---|
| Best Epoch | {state.best_epoch} |
| Best val_macro_f1 | {state.best_val_macro_f1:.4f} |
| Best val_auc | {state.best_val_auc:.4f} |
| Best val_loss | {state.best_val_loss:.4f} |
| Best val_accuracy | {state.best_val_accuracy:.2f}% |
| Total Epochs | {state.epoch} |
| Git SHA | `{state.git_sha[:8] if state.git_sha else 'N/A'}` |

## Checkpoints

- `checkpoints/last.pt` — most recent epoch (overwritten each epoch)
- `checkpoints/best.pt` — best val_macro_f1 (overwritten on improvement)
- `checkpoints/epoch_<N>.pt` — periodic snapshots (every 10 epochs)

## Early Stopping

Patience: {state.epochs_since_improvement}/{state.patience}
{f"- Stopped at epoch {state.epoch}: {state.reason}" if state.status == "early_stopped" else ""}

## HF Repo Structure

```
checkpoints/
    last.pt
    best.pt
    epoch_10.pt
    epoch_20.pt
    ...
logs/
    training_state.json
    metrics.csv
    session_log.csv
results/
    final_model.pt
    model.onnx
    final_metrics.json
    manuscript_ready_metrics.json
    figures/
```
"""
        readme_path = self.results_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)
        self.upload_file(readme_path, "README.md")

    # ------------------------------------------------------------------
    # Session logging
    # ------------------------------------------------------------------

    def log_session_start(self):
        """Log the start of a Kaggle session."""
        csv_path = self.logs_dir / "session_log.csv"
        import csv
        write_header = not csv_path.exists()

        row = {
            "session_id": self._session_id,
            "event": "start",
            "timestamp": datetime.now().isoformat(),
            "gpu_name": self._get_gpu_name(),
            "gpu_memory_gb": self._get_gpu_memory(),
        }

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self.upload_file(csv_path, "logs/session_log.csv")

    def _log_session_end(self, state: TrainingState):
        """Log the end of a Kaggle session."""
        elapsed_minutes = (time.time() - self._session_start) / 60
        state.total_gpu_minutes += elapsed_minutes

        csv_path = self.logs_dir / "session_log.csv"
        import csv

        row = {
            "session_id": self._session_id,
            "event": "end",
            "timestamp": datetime.now().isoformat(),
            "gpu_name": self._get_gpu_name(),
            "gpu_memory_gb": self._get_gpu_memory(),
            "elapsed_minutes": round(elapsed_minutes, 2),
            "epoch": state.epoch,
            "best_val_macro_f1": state.best_val_macro_f1,
            "total_gpu_minutes": round(state.total_gpu_minutes, 2),
        }

        # Re-read existing to get headers
        fieldnames = list(row.keys())
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerow(row)

        self.upload_file(csv_path, "logs/session_log.csv")

        # Also update training state with accumulated GPU time
        self.save_training_state_local(state)
        self.upload_file(self.logs_dir / "training_state.json", "logs/training_state.json")

    @staticmethod
    def _get_gpu_name() -> str:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip().split("\n")[0] if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def _get_gpu_memory() -> float:
        try:
            if torch.cuda.is_available():
                return round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
        except Exception:
            pass
        return 0.0
