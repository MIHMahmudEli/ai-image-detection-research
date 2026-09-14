"""
Kaggle Utilities
================
Shared helpers for all Kaggle notebooks:
  - Environment detection (Kaggle vs local vs DGX)
  - Dataset path resolution across mounted inputs
  - HuggingFace token discovery + upload (checkpoints, results, models)
  - Full-state checkpoint save/resume (survives session crashes)
  - Working directory setup

Usage in any notebook cell:
    from src.kaggle_utils import KaggleEnv
    env = KaggleEnv(project_root_search=True)
    # env.is_kaggle, env.working_dir, env.figures_dir, etc.
"""

import os
import sys
import json
import hashlib
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, List, Any


class KaggleEnv:
    """
    Detects the runtime environment and resolves all paths.

    Attributes:
        is_kaggle (bool): True if running on Kaggle
        is_gpu (bool): True if CUDA available
        smoke_test (bool): True if no GPU (quick verification mode)
        project_root (Path): Root of the ai-image-detection-research repo
        working_dir (Path): Kaggle working dir or project root
        images_dir (Path): Resolved path to dataset/images
        checkpoints_dir (Path): model/checkpoints/
        results_dir (Path): paper/result/<mode>/
        figures_dir (Path): paper/result/<mode>/fig/
        tables_dir (Path): paper/result/<mode>/table/
        hf_token (str): HuggingFace token or empty
        hf_checkpoint_repo (str): HF repo for checkpoints
        hf_results_repo (str): HF repo for results/paper outputs
        hf_model_repo (str): HF repo for deployment-ready models
    """

    def __init__(
        self,
        project_root_search: bool = True,
        variant: str = "base",
        mode: str = "full_scale",
    ):
        import torch

        # ── Environment detection ──
        self.is_kaggle = Path("/kaggle/input").exists()
        self.is_gpu = torch.cuda.is_available()
        self.smoke_test = not self.is_gpu

        # ── Project root ──
        if project_root_search:
            self.project_root = self._find_project_root()
        else:
            self.project_root = Path.cwd().resolve()

        # ── Working directory ──
        if self.is_kaggle:
            self.working_dir = Path("/kaggle/working")
        else:
            self.working_dir = self.project_root

        os.chdir(self.working_dir)
        if str(self.project_root / "model") not in sys.path:
            sys.path.insert(0, str(self.project_root / "model"))

        # ── Image data resolution ──
        self.images_dir = self._resolve_images_dir()

        # ── Checkpoints, results, figures ──
        self.checkpoints_dir = self.project_root / "model" / "checkpoints"
        mode_dir = "verify" if self.smoke_test else mode
        self.results_dir = self.project_root / "paper" / "result" / mode_dir
        self.figures_dir = self.results_dir / "fig"
        self.tables_dir = self.results_dir / "table"

        # ── HuggingFace config ──
        self.hf_token = self._find_hf_token()
        self.hf_checkpoint_repo = os.environ.get(
            "HF_CHECKPOINT_REPO", "MohsinElis/mfft-checkpoints"
        )
        self.hf_results_repo = os.environ.get(
            "HF_RESULTS_REPO", "MohsinElis/mfft-results"
        )
        self.hf_model_repo = os.environ.get(
            "HF_MODEL_REPO", "MohsinElis/mfft-model"
        )

        # ── Summary ──
        print(f"KaggleEnv initialized:")
        print(f"  is_kaggle={self.is_kaggle}, is_gpu={self.is_gpu}, smoke={self.smoke_test}")
        print(f"  project_root={self.project_root}")
        print(f"  working_dir={self.working_dir}")
        print(f"  images_dir={self.images_dir}")
        print(f"  hf_token={'***' if self.hf_token else 'NOT FOUND'}")

    def _find_project_root(self) -> Path:
        """Walk up from CWD to find AGENTS.md or .git."""
        p = Path.cwd().resolve()
        for _ in range(10):
            if (p / "AGENTS.md").exists() or (p / ".git").exists():
                return p
            parent = p.parent
            if parent == p:
                break
            p = parent
        # Fallback: try known Kaggle dataset paths
        for candidate in [
            Path("/kaggle/input/ai-image-detection-research"),
            Path("/kaggle/working"),
            Path.cwd().resolve(),
        ]:
            if (candidate / "model").exists() or (candidate / "AGENTS.md").exists():
                return candidate
        return Path.cwd().resolve()

    def _resolve_images_dir(self) -> Path:
        """Find dataset/images across Kaggle mounts or local."""
        # Check multiple possible locations
        candidates = [
            self.project_root / "dataset" / "images",
            Path("/kaggle/working/dataset/images"),
        ]
        # Check Kaggle input mounts
        if self.is_kaggle:
            input_root = Path("/kaggle/input")
            for mount in input_root.rglob("images"):
                if mount.is_dir() and any(
                    mount.glob("**/*.jpg") or mount.glob("**/*.png")
                ):
                    candidates.insert(0, mount)
        for c in candidates:
            if c.exists():
                return c
        return self.project_root / "dataset" / "images"

    def _find_hf_token(self) -> str:
        """Auto-discover HF token from Kaggle Secrets, env, or .env."""
        # 1. Kaggle Secret
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
        # 2. Environment variable
        token = os.environ.get("HF_TOKEN")
        if token:
            return token
        # 3. .env file
        for p in [
            Path("/kaggle/working/.env"),
            self.project_root / ".env",
            Path.home() / ".env",
        ]:
            if p.exists():
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("hf="):
                            return line.split("=", 1)[1]
                        if line.startswith("HF_TOKEN="):
                            return line.split("=", 1)[1]
        return ""

    # ═══════════════════════════════════════════════════════════
    # Checkpoint save/resume (full state)
    # ═══════════════════════════════════════════════════════════

    def save_checkpoint(
        self,
        path: Path,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch: int,
        best_acc: float,
        history: dict,
        extra: Optional[dict] = None,
    ) -> Path:
        """
        Save full training state for resumption after crash/session limit.
        Returns the path saved to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_acc": best_acc,
            "history": history,
            "saved_at": datetime.now().isoformat(),
        }
        if extra:
            state["extra"] = extra

        torch.save(state, path)
        size_mb = path.stat().st_size / 1e6
        print(f"  Checkpoint saved: {path.name} ({size_mb:.1f} MB) @ epoch {epoch}")
        return path

    def load_checkpoint(
        self,
        path: Path,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device="cpu",
    ) -> Optional[dict]:
        """
        Load full training state. Returns extra info dict or None if not found.
        If optimizer/scheduler/scaler are provided, their states are restored.
        """
        path = Path(path)
        if not path.exists():
            return None

        try:
            state = torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"  WARNING: Could not load checkpoint {path.name}: {e}")
            return None

        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])

        epoch = state.get("epoch", 0)
        best_acc = state.get("best_acc", 0.0)
        history = state.get("history", {})
        extra = state.get("extra", {})

        print(f"  Resumed from {path.name}: epoch={epoch}, best_acc={best_acc:.2f}%")
        return {
            "epoch": epoch,
            "best_acc": best_acc,
            "history": history,
            "extra": extra,
        }

    def find_resume_checkpoint(self, ckpt_dir: Path) -> Optional[Path]:
        """Find the latest checkpoint in a directory for resumption."""
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            return None
        candidates = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
        # Fallback to best.pt
        best = ckpt_dir / "best.pt"
        return best if best.exists() else None

    # ═══════════════════════════════════════════════════════════
    # HuggingFace upload
    # ═══════════════════════════════════════════════════════════

    def upload_to_hf(
        self,
        local_path: Path,
        repo_id: str,
        path_in_repo: str,
        repo_type: str = "model",
        commit_message: str = "",
    ) -> bool:
        """Upload a file or directory to HuggingFace Hub."""
        if not self.hf_token:
            print(f"  WARNING: No HF token — skipping upload to {repo_id}")
            return False

        try:
            from huggingface_hub import HfApi
        except ImportError:
            os.system("pip install huggingface_hub -q")
            from huggingface_hub import HfApi

        api = HfApi(token=self.hf_token)
        api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)

        local_path = Path(local_path)
        msg = commit_message or f"Upload {local_path.name} at {datetime.now().isoformat()}"

        if local_path.is_dir():
            api.upload_folder(
                folder_path=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=msg,
            )
        else:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=msg,
            )
        print(f"  Uploaded: {local_path.name} -> {repo_id}/{path_in_repo}")
        return True

    def upload_checkpoint(self, local_ckpt: Path, model_name: str) -> bool:
        """Upload a checkpoint file to the checkpoints HF repo."""
        return self.upload_to_hf(
            local_ckpt,
            self.hf_checkpoint_repo,
            f"checkpoints/{model_name}/{local_ckpt.name}",
        )

    def upload_results_dir(self, local_dir: Path, tag: str = "") -> bool:
        """Upload an entire results directory to the results HF repo."""
        return self.upload_to_hf(
            local_dir,
            self.hf_results_repo,
            f"results/{tag}/{local_dir.name}" if tag else f"results/{local_dir.name}",
        )

    def upload_model_for_deployment(self, local_ckpt: Path, model_name: str) -> bool:
        """Upload a model checkpoint to the deployment HF repo."""
        return self.upload_to_hf(
            local_ckpt,
            self.hf_model_repo,
            f"models/{model_name}/{local_ckpt.name}",
            commit_message=f"Deploy {model_name} checkpoint",
        )


def setup_notebook_env(
    variant: str = "base",
    mode: str = "full_scale",
) -> KaggleEnv:
    """
    One-call setup for notebook Cell 1.
    Returns a fully initialized KaggleEnv.
    """
    env = KaggleEnv(project_root_search=True, variant=variant, mode=mode)
    return env
