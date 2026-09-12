"""
MFFT Resumable Kaggle Training Script
========================================
Multi-session training with HuggingFace Hub persistence.

Features:
- Resumes automatically after Kaggle session disconnect
- 40-epoch schedule with 8-epoch early stopping patience
- Macro-F1 as primary metric (3-class: real/ai_generated/deepfake)
- Periodic checkpoint snapshots at epochs 10/20/30/40
- Session logging for GPU-hour tracking

SETUP:
1. Clone repo: git clone <repo_url> && cd <repo_dir>
2. pip install -r model/requirements.txt && pip install huggingface_hub
3. Attach all dataset shards via Kaggle Input panel
4. Set HF_TOKEN as Kaggle Secret
5. Run this script

CONFIG: Edit the CONFIG section below before running.
"""

# ============================================================
# CONFIG — CHANGE THESE
# ============================================================
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/ai-image-detection-research.git"
GITHUB_COMMIT_SHA = "main"  # pin to specific commit/tag for reproducibility

HF_REPO_ID = "YOUR_USERNAME/mfft-checkpoints"  # HF repo for checkpoints
EXPECTED_SHARDS = ["mfft-shard-0", "mfft-shard-1"]  # Kaggle dataset slugs

MAX_EPOCHS = 40
PATIENCE = 8
IMAGE_SIZE = 224
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
MODEL_VARIANT = "base"
SEED = 42
SNAPSHOT_EPOCHS = [10, 20, 30, 40]
MIN_IMAGES_PER_SHARD = 500
# ============================================================

import os
import sys
import time
import json
import random
import subprocess
from pathlib import Path
from datetime import datetime

# Step 1: Clone repo (if not already present)
REPO_DIR = Path("/kaggle/working/mfft_repo")
if not REPO_DIR.exists():
    print(f"Cloning MFFT repo from {GITHUB_REPO_URL}...")
    subprocess.run(
        ["git", "clone", GITHUB_REPO_URL, str(REPO_DIR)],
        check=True,
    )
    if GITHUB_COMMIT_SHA != "main":
        subprocess.run(
            ["git", "-C", str(REPO_DIR), "checkout", GITHUB_COMMIT_SHA],
            check=True,
        )
    print(f"Cloned to {REPO_DIR} (commit: {GITHUB_COMMIT_SHA})")
else:
    print(f"Repo already exists at {REPO_DIR}")

# Add repo to Python path
sys.path.insert(0, str(REPO_DIR))

# Step 2: Install dependencies
subprocess.run(
    ["pip", "install", "-q", "huggingface_hub", "open_clip_torch", "scipy"],
    check=False,
)

# Step 3: Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)
from collections import Counter

# Local imports
from kaggle_dataset_loader import KaggleDatasetLoader, create_pytorch_datasets, CLASS_NAMES
from hf_checkpoint_manager import HFCheckpointManager, TrainingState

# Import model from cloned repo
from model.src.model import build_mfft, count_parameters

# ============================================================
# SEED
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================================================
# GIT SHA
# ============================================================
try:
    git_sha = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()
except Exception:
    git_sha = "unknown"
print(f"Git SHA: {git_sha}")

# ============================================================
# HF TOKEN (from Kaggle Secret)
# ============================================================
def get_hf_token():
    """Retrieve HF token from Kaggle Secret or environment."""
    # Try Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        return secrets.get_secret("HF_TOKEN")
    except Exception:
        pass

    # Try environment variable
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    raise ValueError(
        "HF_TOKEN not found. Set it as a Kaggle Secret or environment variable.\n"
        "Kaggle: Add via Add Data > Add a secret > Name: HF_TOKEN, Value: <your_token>"
    )

hf_token = get_hf_token()
print(f"HF Token loaded: {hf_token[:8]}...")

# ============================================================
# DATASET
# ============================================================
print("\n=== Loading Dataset ===")
loader = KaggleDatasetLoader(
    expected_shards=EXPECTED_SHARDS,
    min_images_per_shard=MIN_IMAGES_PER_SHARD,
    seed=SEED,
)

try:
    df, samples = loader.load()
except FileNotFoundError as e:
    print(f"\nERROR: {e}")
    print("\nAvailable shards under /kaggle/input/:")
    for entry in sorted(Path("/kaggle/input").iterdir()):
        if entry.is_dir():
            print(f"  - {entry.name}")
    raise

print(loader.summary())

# Create dataloaders
train_loader, val_loader, train_dataset, val_dataset = create_pytorch_datasets(
    df, image_size=IMAGE_SIZE, val_split=0.15, seed=SEED,
)

print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")
print(f"Batches: train={len(train_loader)}, val={len(val_loader)}")

# ============================================================
# MODEL
# ============================================================
print("\n=== Building MFFT Model ===")
model = build_mfft(MODEL_VARIANT).to(device)
n_params = count_parameters(model)
print(f"Parameters: {n_params:,}")

# ============================================================
# TRAINING SETUP
# ============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

warmup_steps = min(500, len(train_loader))
warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
cosine = CosineAnnealingWarmRestarts(optimizer, T_0=MAX_EPOCHS * len(train_loader), T_mult=2, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

scaler = GradScaler(enabled=torch.cuda.is_available())

# ============================================================
# CHECKPOINT MANAGER
# ============================================================
hf_mgr = HFCheckpointManager(
    repo_id=HF_REPO_ID,
    hf_token=hf_token,
    local_dir="/kaggle/working/mfft_training",
    git_sha=git_sha,
)

# ============================================================
# RESUME
# ============================================================
state = TrainingState(
    patience=PATIENCE,
    git_sha=git_sha,
)

start_epoch = 1
resumed = hf_mgr.resume()
if resumed is not None:
    checkpoint, restored_state = resumed
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("scaler_state_dict") and torch.cuda.is_available():
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    state = restored_state
    start_epoch = state.epoch + 1
    print(f"\nResumed from epoch {state.epoch}, continuing at epoch {start_epoch}")
else:
    print("\nStarting fresh training run.")

# Log session start
hf_mgr.log_session_start()

# ============================================================
# TRAINING LOOP
# ============================================================
print(f"\n{'='*60}")
print(f"Training MFFT-{MODEL_VARIANT}")
print(f"Epochs: {start_epoch} -> {MAX_EPOCHS}")
print(f"Batch: {BATCH_SIZE}, Image: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Patience: {PATIENCE} (macro-F1)")
print(f"{'='*60}\n")

start_time = time.time()
history = []

for epoch in range(start_epoch, MAX_EPOCHS + 1):
    epoch_start = time.time()

    # ── Train ──
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        preds = logits.argmax(dim=-1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total * 100
    avg_train_loss = train_loss / len(train_loader)

    # ── Validate ──
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            val_loss += loss.item()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_acc = val_correct / val_total * 100
    avg_val_loss = val_loss / len(val_loader)

    # ── Metrics ──
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    f1_binary = f1_score(y_true, y_pred, average="binary", zero_division=0) * 100
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

    try:
        if y_prob.ndim == 2 and y_prob.shape[1] > 2:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        else:
            auc = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        auc = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time

    # ── Print ──
    print(
        f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
        f"Train: {avg_train_loss:.4f}/{train_acc:.2f}% | "
        f"Val: {avg_val_loss:.4f}/{val_acc:.2f}% | "
        f"MacroF1: {macro_f1:.2f} | AUC: {auc:.4f} | "
        f"F1-bin: {f1_binary:.2f} | Spec: {specificity:.2f}% | "
        f"{epoch_time:.0f}s"
    )

    # ── Metrics dict ──
    metrics = {
        "epoch": epoch,
        "train_loss": round(avg_train_loss, 6),
        "train_acc": round(train_acc, 4),
        "val_loss": round(avg_val_loss, 6),
        "val_accuracy": round(val_acc, 4),
        "val_precision": round(precision, 4),
        "val_recall": round(recall, 4),
        "val_f1": round(f1_binary, 4),
        "val_macro_f1": round(macro_f1, 4),
        "val_auc": round(auc, 6),
        "learning_rate": round(scheduler.get_last_lr()[0], 8),
        "timestamp": datetime.now().isoformat(),
    }
    history.append(metrics)

    # ── Save + Upload ──
    state = hf_mgr.save_and_upload(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        state=state,
        scaler=scaler,
    )

    # ── Early Stopping ──
    if state.epochs_since_improvement >= state.patience:
        state.status = "early_stopped"
        state.reason = (
            f"No improvement in val_macro_f1 for {state.patience} consecutive epochs. "
            f"Best epoch: {state.best_epoch} (macro_f1={state.best_val_macro_f1:.4f})"
        )
        hf_mgr.save_training_state_local(state)
        hf_mgr.upload_file(hf_mgr.logs_dir / "training_state.json", "logs/training_state.json")
        print(f"\n*** EARLY STOP at epoch {epoch}: {state.reason} ***")
        break

    # ── Natural completion ──
    if epoch == MAX_EPOCHS:
        state.status = "completed"
        state.reason = f"Completed all {MAX_EPOCHS} epochs."
        hf_mgr.save_training_state_local(state)
        hf_mgr.upload_file(hf_mgr.logs_dir / "training_state.json", "logs/training_state.json")

# ============================================================
# FINAL RESULTS
# ============================================================
print(f"\n{'='*60}")
print(f"TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Status: {state.status}")
print(f"Best epoch: {state.best_epoch}")
print(f"Best val_macro_f1: {state.best_val_macro_f1:.4f}")
print(f"Best val_auc: {state.best_val_auc:.4f}")
print(f"Best val_accuracy: {state.best_val_accuracy:.2f}%")
print(f"Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
print(f"{'='*60}")

# ── Final metrics ──
final_metrics = {
    "status": state.status,
    "best_epoch": state.best_epoch,
    "best_val_macro_f1": state.best_val_macro_f1,
    "best_val_auc": state.best_val_auc,
    "best_val_loss": state.best_val_loss,
    "best_val_accuracy": state.best_val_accuracy,
    "total_epochs": epoch,
    "total_time_seconds": round(elapsed, 2),
    "params": n_params,
    "model_variant": MODEL_VARIANT,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "git_sha": git_sha,
}

# ── Manuscript-ready metrics ──
# Find the best epoch's full metrics
best_metrics = min(history, key=lambda m: abs(m["val_macro_f1"] - state.best_val_macro_f1))
manuscript_metrics = {
    "mfft_variant": MODEL_VARIANT,
    "accuracy": best_metrics["val_accuracy"],
    "precision_macro": best_metrics["val_precision"],
    "recall_macro": best_metrics["val_recall"],
    "f1_binary": best_metrics["val_f1"],
    "f1_macro": best_metrics["val_macro_f1"],
    "auc_roc": best_metrics["val_auc"],
    "specificity": specificity,
    "confusion_matrix": cm.tolist(),
    "best_epoch": state.best_epoch,
    "total_training_time_hours": round(elapsed / 3600, 2),
}

# ── Upload final artifacts ──
hf_mgr.upload_final_artifacts(
    model=model,
    state=state,
    final_metrics=final_metrics,
    figures_dir=Path("/kaggle/working/mfft_training/results"),
    manuscript_metrics=manuscript_metrics,
)

# ── Save locally ──
results_path = Path("/kaggle/working/mfft_results.json")
with open(results_path, "w") as f:
    json.dump({
        "final_metrics": final_metrics,
        "manuscript_metrics": manuscript_metrics,
        "history": history,
    }, f, indent=2)

print(f"\nResults saved to {results_path}")
print(f"Download checkpoints from: https://huggingface.co/{HF_REPO_ID}")
