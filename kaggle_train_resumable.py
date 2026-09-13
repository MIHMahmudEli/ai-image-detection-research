"""
MFFT Resumable Kaggle Training Script
=======================================
Multi-session, multi-account training with HuggingFace Hub persistence.

All parallel runs evaluate on the EXACT same data because:
  1. A frozen split manifest is built on first run and shared via HF
  2. All subsequent runs download and use that same manifest
  3. Per-run namespacing prevents checkpoint collisions on HF

FIRST RUN (MFFT-base, all 11 datasets attached):
  1. Clones repo, installs deps
  2. SplitManifestManager bootstraps manifest from mounted datasets
  3. Uploads manifest to HF at manifest/split_manifest.json (frozen)
  4. Trains model, uploads checkpoints to runs/<run_id>/...

LATER RUNS (any account, any datasets attached):
  1. Clones repo, installs deps
  2. SplitManifestManager downloads existing manifest from HF
  3. KaggleDatasetLoader resolves paths from mounted datasets
  4. Resumes training from last checkpoint on HF

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
import hashlib
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
subprocess.run(["pip", "install", "-q", "huggingface_hub", "open_clip_torch", "scipy", "python-dotenv"], check=False)

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

from hf_checkpoint_manager import HFCheckpointManager, TrainingState, MetricRecord, compute_manifest_sha256
from split_manifest_manager import SplitManifestManager
from kaggle_dataset_loader import KaggleDatasetLoader
from model.src.model import build_mfft, count_parameters

CLASS_NAMES = ["real", "ai_generated", "deepfake"]

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

# ── Run ID ──
date_str = datetime.now().strftime("%Y%m%d")
account_str = "kaggle"
try:
    from kaggle_secrets import UserSecretsClient
    account_str = UserSecretsClient().get_secret("KAGGLE_USERNAME")[:8]
except Exception:
    pass
run_id = f"{MODEL_VARIANT}_{account_str}_{date_str}"
print(f"Run ID: {run_id}")

# ============================================================
# PHASE 1: SPLIT MANIFEST (download-or-bootstrap)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: Split Manifest")
print("=" * 60)

manifest_mgr = SplitManifestManager(
    hf_token=hf_token,
    hf_repo=HF_MANIFEST_REPO,
    run_id=run_id,
)

manifest, manifest_sha256 = manifest_mgr.get_or_bootstrap()
manifest_version = manifest.get("version", 1)

print(f"\nManifest details:")
print(f"  Version: {manifest_version}")
print(f"  SHA-256: {manifest_sha256[:16]}...")
print(f"  Total images: {manifest['total_images']}")
print(f"  Shards used: {manifest['shards_used']}")
print(f"  Split sizes: {manifest['split_sizes']}")
print(f"  Class distribution: {manifest['class_distribution']}")

# ============================================================
# PHASE 2: LOAD DATASET
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Loading Dataset")
print("=" * 60)

loader = KaggleDatasetLoader(
    manifest=manifest,
    manifest_sha256=manifest_sha256,
    input_root="/kaggle/input",
    image_size=IMAGE_SIZE,
)

train_loader, val_loader, test_loader = loader.create_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True,
)

split_summary = loader.get_split_summary()
print(f"\nSplit verification (all models match this):")
print(f"  Manifest: v{split_summary['manifest_version']} ({split_summary['manifest_sha256']})")
print(f"  Train: {split_summary['resolved']['train']}")
print(f"  Val: {split_summary['resolved']['val']}")
print(f"  Test: {split_summary['resolved']['test']}")

if len(loader.train_data) == 0:
    raise RuntimeError(
        "No training images resolved. Ensure all 11 Kaggle datasets are attached."
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

# ── Checkpoint Manager (run_id-namespaced) ──
hf_mgr = HFCheckpointManager(
    hf_token=hf_token,
    hf_repo=HF_REPO_ID,
    run_id=run_id,
    manifest_sha256=manifest_sha256,
)

# ── Resume ──
state = TrainingState(
    run_id=run_id,
    model_variant=MODEL_VARIANT,
    manifest_sha256=manifest_sha256,
    manifest_version=manifest_version,
    total_epochs=MAX_EPOCHS,
    patience_limit=PATIENCE,
)
start_epoch = 1

existing_state = hf_mgr.load_state()
if existing_state is not None:
    # Resume from checkpoint
    ckpt_path = hf_mgr.download_checkpoint("last")
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") and torch.cuda.is_available():
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    state = existing_state
    start_epoch = state.current_epoch + 1
    print(f"\nResumed from epoch {state.current_epoch} -> epoch {start_epoch}")
    print(f"  Best val macro-F1: {state.best_val_macro_f1:.4f}")
    print(f"  Manifest hash: {state.manifest_sha256[:16]}...")
else:
    print("\nStarting fresh training run.")

hf_mgr.log_session(f"START run_id={run_id} manifest_sha256={manifest_sha256[:16]} epoch={start_epoch}")

# ============================================================
# PHASE 5: TRAINING LOOP
# ============================================================
print(f"\n{'='*60}")
print(f"Training MFFT-{MODEL_VARIANT}")
print(f"Run ID: {run_id}")
print(f"Epochs: {start_epoch} -> {MAX_EPOCHS}")
print(f"Batch: {BATCH_SIZE}, Image: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Patience: {PATIENCE} (macro-F1)")
print(f"Manifest: v{manifest_version} ({manifest_sha256[:16]}...)")
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

    # ── Update state ──
    state.current_epoch = epoch
    state.train_accuracy = train_acc
    state.val_accuracy = val_acc
    state.val_macro_f1 = macro_f1 / 100
    state.val_auc = auc
    state.val_loss = avg_val_loss
    state.total_train_time = elapsed

    # ── Early stopping logic ──
    improved = False
    if macro_f1 / 100 > state.best_val_macro_f1:
        state.best_val_macro_f1 = macro_f1 / 100
        state.best_val_auc = auc
        state.best_val_accuracy = val_acc
        state.best_val_loss = avg_val_loss
        state.best_model_epoch = epoch
        state.patience_counter = 0
        improved = True
    else:
        state.patience_counter += 1

    state.status = "training"

    # ── Save state + best checkpoint ──
    state_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if torch.cuda.is_available() else None,
        "epoch": epoch,
    }

    # Always save last
    tmp_last = Path("/kaggle/working/last.pt")
    torch.save(state_dict, tmp_last)
    hf_mgr.upload_checkpoint(str(tmp_last), "last")

    # Save best if improved
    if improved:
        tmp_best = Path("/kaggle/working/best.pt")
        torch.save(state_dict, tmp_best)
        hf_mgr.upload_checkpoint(str(tmp_best), "best")

    # Snapshot epochs
    if epoch in SNAPSHOT_EPOCHS:
        tmp_snap = Path(f"/kaggle/working/epoch_{epoch}.pt")
        torch.save(state_dict, tmp_snap)
        hf_mgr.upload_checkpoint(str(tmp_snap), f"epoch_{epoch}")

    # Save state
    hf_mgr.save_state(state)

    # Log metrics
    metric_record = MetricRecord(
        run_id=run_id,
        epoch=epoch,
        train_loss=round(avg_train_loss, 6),
        train_accuracy=round(train_acc, 4),
        val_loss=round(avg_val_loss, 6),
        val_accuracy=round(val_acc, 4),
        val_macro_f1=round(macro_f1, 4),
        val_auc=round(auc, 6),
        val_precision=round(precision, 4),
        val_recall=round(recall, 4),
        val_specificity=round(specificity, 4),
        learning_rate=round(scheduler.get_last_lr()[0], 8),
        elapsed_time=round(epoch_time, 2),
    )
    hf_mgr.log_metrics(metric_record)

    # ── Early stopping ──
    if state.patience_counter >= state.patience_limit:
        state.status = "early_stopped"
        hf_mgr.save_state(state)
        hf_mgr.log_session(f"EARLY_STOP epoch={epoch} best_epoch={state.best_model_epoch} best_f1={state.best_val_macro_f1:.4f}")
        print(f"\n*** EARLY STOP at epoch {epoch}: no improvement for {state.patience_limit} epochs ***")
        print(f"    Best: epoch {state.best_model_epoch} (macro_f1={state.best_val_macro_f1:.4f})")
        break

    if epoch == MAX_EPOCHS:
        state.status = "completed"
        hf_mgr.save_state(state)
        hf_mgr.log_session(f"COMPLETED epoch={epoch} best_f1={state.best_val_macro_f1:.4f}")

# ============================================================
# FINAL
# ============================================================
print(f"\n{'='*60}")
print(f"TRAINING COMPLETE — {run_id}")
print(f"Status: {state.status}")
print(f"Best epoch: {state.best_model_epoch} | macro_f1={state.best_val_macro_f1:.4f} | auc={state.best_val_auc:.4f}")
print(f"Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
print(f"Manifest: v{manifest_version} ({manifest_sha256[:16]}...)")
print(f"{'='*60}")

# ── ONNX export ──
try:
    model.eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    onnx_path = Path("/kaggle/working/model.onnx")
    torch.onnx.export(model, dummy, str(onnx_path), opset_version=14)
    hf_mgr.upload_onnx(str(onnx_path))
    print(f"ONNX exported: {onnx_path}")
except Exception as e:
    print(f"ONNX export failed: {e}")

# ── Final model card ──
hf_mgr.update_model_card(state, metrics={
    "best_epoch": state.best_model_epoch,
    "best_val_macro_f1": state.best_val_macro_f1,
    "best_val_auc": state.best_val_auc,
    "total_epochs": state.current_epoch,
    "total_time_hours": round(elapsed / 3600, 2),
})

# ── Results JSON ──
results = {
    "run_id": run_id,
    "model_variant": MODEL_VARIANT,
    "manifest_sha256": manifest_sha256,
    "manifest_version": manifest_version,
    "status": state.status,
    "best_epoch": state.best_model_epoch,
    "best_val_macro_f1": state.best_val_macro_f1,
    "best_val_auc": state.best_val_auc,
    "total_time_seconds": round(elapsed, 2),
}
results_path = Path("/kaggle/working/results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults: {results_path}")
print(f"Checkpoints: https://huggingface.co/{HF_REPO_ID}/tree/main/runs/{run_id}")
