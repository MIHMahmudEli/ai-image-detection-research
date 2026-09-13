"""
MFFT Resumable Kaggle Training Script
========================================
Multi-session training with HuggingFace Hub persistence.

FIRST RUN (with all 11 datasets attached):
  1. Clones repo, installs deps
  2. Builds master manifest by scanning mounted datasets
  3. Uploads manifest to HuggingFace
  4. Trains model, uploads checkpoints to HF

LATER RUNS (any account, any datasets attached):
  1. Clones repo, installs deps
  2. Downloads manifest from HF (same split as first run)
  3. Resolves paths from currently mounted datasets
  4. Resumes training from last checkpoint on HF

All models evaluate on the EXACT same data because the manifest
defines deterministic splits (seed=42) stored on HuggingFace.

SETUP:
1. Create Kaggle Notebook with GPU
2. Attach ALL 11 datasets:
   - stable-diffusion, places365, open-images-v7-dataset
   - ntire2026, midjourney, mfft-real, genimage-ai
   - faceforensics, dfdc-faces-of-the-train-sample
   - dall-e3, celebdf-v2image-dataset
3. Set HF_TOKEN as Kaggle Secret
4. Run this script
"""

# ============================================================
# CONFIG
# ============================================================
GITHUB_REPO_URL = "https://github.com/MIHMahmudEli/ai-image-detection-research.git"
GITHUB_COMMIT_SHA = "main"

HF_REPO_ID = "studyhub991/mfft-checkpoints"
HF_MANIFEST_REPO = "studyhub991/mfft-master-manifest"

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
# ============================================================

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# ── Step 1: Clone repo ──
REPO_DIR = Path("/kaggle/working/mfft_repo")
if not REPO_DIR.exists():
    print(f"Cloning MFFT repo from {GITHUB_REPO_URL}...")
    subprocess.run(["git", "clone", GITHUB_REPO_URL, str(REPO_DIR)], check=True)
    if GITHUB_COMMIT_SHA != "main":
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", GITHUB_COMMIT_SHA], check=True)
    print(f"Cloned (commit: {GITHUB_COMMIT_SHA})")
else:
    print(f"Repo exists at {REPO_DIR}")

sys.path.insert(0, str(REPO_DIR))

# ── Step 2: Install deps ──
subprocess.run(["pip", "install", "-q", "huggingface_hub", "open_clip_torch", "scipy"], check=False)

# ── Step 3: Imports ──
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from collections import Counter

from hf_checkpoint_manager import HFCheckpointManager, TrainingState
from model.src.model import build_mfft, count_parameters

# ── Seed ──
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ── Device ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ── Git SHA ──
try:
    git_sha = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    git_sha = "unknown"
print(f"Git SHA: {git_sha}")

# ── HF Token ──
def get_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = Path("/kaggle/working/.env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("hf="):
                    return line.strip().split("=", 1)[1]
    raise ValueError("HF_TOKEN not found. Set as Kaggle Secret.")

hf_token = get_hf_token()
print(f"HF Token: {hf_token[:8]}...")

# ============================================================
# PHASE 1: BUILD OR DOWNLOAD MANIFEST
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: Dataset Manifest")
print("=" * 60)

# Try to download existing manifest from HF
manifest_available = False
try:
    from huggingface_hub import hf_hub_download
    manifest_path = hf_hub_download(
        repo_id=HF_MANIFEST_REPO,
        filename="master_manifest.json",
        repo_type="model",
        token=hf_token,
    )
    with open(manifest_path) as f:
        existing_manifest = json.load(f)
    print(f"Existing manifest found on HF: {existing_manifest.get('manifest_hash', 'N/A')}")
    print(f"  Samples: {len(existing_manifest.get('samples', []))}")
    manifest_available = True
except Exception:
    print("No manifest found on HuggingFace Hub")

if not manifest_available:
    # FIRST RUN: Build manifest by scanning mounted datasets
    print("\n*** FIRST RUN: Building master manifest from mounted datasets ***")
    print("Attaching all 11 datasets...")

    # Import and run the manifest builder
    sys.path.insert(0, str(REPO_DIR))
    from build_master_manifest import build_manifest

    manifest = build_manifest(input_root="/kaggle/input", upload=True)
    print(f"\nManifest uploaded to HF: {manifest.get('manifest_hash', 'N/A')}")
else:
    print("Using existing manifest from HF (all models will use this same split)")

# ============================================================
# PHASE 2: LOAD DATASET
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Loading Dataset")
print("=" * 60)

from kaggle_dataset_loader import KaggleDatasetLoader

loader = KaggleDatasetLoader(
    hf_token=hf_token,
    hf_manifest_repo=HF_MANIFEST_REPO,
    image_size=IMAGE_SIZE,
)

train_df, val_df, test_df = loader.load()

if len(train_df) == 0:
    raise RuntimeError(
        "No training images found. Ensure all 11 Kaggle datasets are attached.\n"
        "Required: stable-diffusion, places365, open-images-v7-dataset,\n"
        "ntire2026, midjourney, mfft-real, genimage-ai, faceforensics,\n"
        "dfdc-faces-of-the-train-sample, dall-e3, celebdf-v2image-dataset"
    )

split_info = loader.get_split_info()
print(f"\nSplit verification (all models match this):")
print(f"  Manifest hash: {split_info['manifest_hash']}")
print(f"  Seed: {split_info['seed']}")
print(f"  Train: {split_info['train_count']} (manifest) -> {len(train_df)} (resolved)")
print(f"  Val: {split_info['val_count']} (manifest) -> {len(val_df)} (resolved)")

# Create dataloaders
train_loader, val_loader = loader.to_dataloaders(
    train_df, val_df, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
)

# ============================================================
# PHASE 3: MODEL
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Building Model")
print("=" * 60)

model = build_mfft(MODEL_VARIANT).to(device)
n_params = count_parameters(model)
print(f"MFFT-{MODEL_VARIANT}: {n_params:,} parameters")

# ============================================================
# PHASE 4: TRAINING SETUP
# ============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

warmup_steps = min(500, len(train_loader))
warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
cosine = CosineAnnealingWarmRestarts(optimizer, T_0=MAX_EPOCHS * len(train_loader), T_mult=2, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
scaler = GradScaler(enabled=torch.cuda.is_available())

# ── Checkpoint Manager ──
hf_mgr = HFCheckpointManager(
    repo_id=HF_REPO_ID,
    hf_token=hf_token,
    local_dir="/kaggle/working/mfft_training",
    git_sha=git_sha,
)

# ── Resume ──
state = TrainingState(patience=PATIENCE, git_sha=git_sha)
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
    print(f"\nResumed from epoch {state.epoch} -> epoch {start_epoch}")
else:
    print("\nStarting fresh training run.")

hf_mgr.log_session_start()

# ============================================================
# PHASE 5: TRAINING LOOP
# ============================================================
print(f"\n{'='*60}")
print(f"Training MFFT-{MODEL_VARIANT}")
print(f"Epochs: {start_epoch} -> {MAX_EPOCHS}")
print(f"Batch: {BATCH_SIZE}, Image: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Patience: {PATIENCE} (macro-F1)")
print(f"Manifest: {split_info['manifest_hash']}")
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

    for images, labels in train_loader:
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
        train_correct += (logits.argmax(dim=-1) == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total * 100
    avg_train_loss = train_loss / len(train_loader)

    # ── Validate ──
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, dim=-1)

            val_loss += loss.item()
            val_correct += (logits.argmax(dim=-1) == labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_acc = val_correct / val_total * 100
    avg_val_loss = val_loss / len(val_loader)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    f1_binary = f1_score(y_true, y_pred, average="binary", zero_division=0) * 100
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100

    try:
        if y_prob.ndim == 2 and y_prob.shape[1] > 2:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        else:
            auc = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) * 100 if (cm[0, 0] + cm[0, 1]) > 0 else 0

    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
        f"Train: {avg_train_loss:.4f}/{train_acc:.2f}% | "
        f"Val: {avg_val_loss:.4f}/{val_acc:.2f}% | "
        f"MacroF1: {macro_f1:.2f} | AUC: {auc:.4f} | "
        f"Spec: {specificity:.2f}% | {epoch_time:.0f}s"
    )

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
        "manifest_hash": split_info["manifest_hash"],
        "split_train_count": len(train_df),
        "split_val_count": len(val_df),
    }
    history.append(metrics)

    # ── Save + Upload ──
    state = hf_mgr.save_and_upload(
        epoch=epoch, model=model, optimizer=optimizer,
        scheduler=scheduler, metrics=metrics, state=state, scaler=scaler,
    )

    # ── Early Stopping ──
    if state.epochs_since_improvement >= state.patience:
        state.status = "early_stopped"
        state.reason = f"No improvement for {state.patience} epochs. Best: epoch {state.best_epoch} (macro_f1={state.best_val_macro_f1:.4f})"
        hf_mgr.save_training_state_local(state)
        hf_mgr.upload_file(hf_mgr.logs_dir / "training_state.json", "logs/training_state.json")
        print(f"\n*** EARLY STOP at epoch {epoch}: {state.reason} ***")
        break

    if epoch == MAX_EPOCHS:
        state.status = "completed"
        state.reason = f"Completed {MAX_EPOCHS} epochs."
        hf_mgr.save_training_state_local(state)
        hf_mgr.upload_file(hf_mgr.logs_dir / "training_state.json", "logs/training_state.json")

# ============================================================
# FINAL
# ============================================================
print(f"\n{'='*60}")
print(f"TRAINING COMPLETE")
print(f"Status: {state.status}")
print(f"Best epoch: {state.best_epoch} | macro_f1={state.best_val_macro_f1:.4f} | auc={state.best_val_auc:.4f}")
print(f"Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
print(f"Manifest: {split_info['manifest_hash']}")
print(f"{'='*60}")

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
    "manifest_hash": split_info["manifest_hash"],
}

best_m = min(history, key=lambda m: abs(m["val_macro_f1"] - state.best_val_macro_f1))
manuscript_metrics = {
    "mfft_variant": MODEL_VARIANT,
    "accuracy": best_m["val_accuracy"],
    "precision_macro": best_m["val_precision"],
    "recall_macro": best_m["val_recall"],
    "f1_binary": best_m["val_f1"],
    "f1_macro": best_m["val_macro_f1"],
    "auc_roc": best_m["val_auc"],
    "confusion_matrix": cm.tolist(),
    "best_epoch": state.best_epoch,
    "manifest_hash": split_info["manifest_hash"],
}

hf_mgr.upload_final_artifacts(model=model, state=state, final_metrics=final_metrics, manuscript_metrics=manuscript_metrics)

results_path = Path("/kaggle/working/mfft_results.json")
with open(results_path, "w") as f:
    json.dump({"final_metrics": final_metrics, "manuscript_metrics": manuscript_metrics, "history": history}, f, indent=2)

print(f"\nResults: {results_path}")
print(f"Checkpoints: https://huggingface.co/{HF_REPO_ID}")
