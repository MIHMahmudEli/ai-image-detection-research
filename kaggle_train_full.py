"""
MFFT Training — Kaggle Notebook (Self-Contained)
=================================================
Multi-Frequency Fusion Transformer for AI Image Detection

SETUP:
  1. Upload the kaggle_dataset/ as a Kaggle Dataset
  2. Create a new Kaggle Notebook
  3. Add your dataset as Input
  4. Enable GPU: Settings → Accelerator → GPU T4 x2
  5. Enable Internet: Settings → Internet → On
  6. Copy this entire script into a single cell and run

OR use the step-by-step notebook version.
"""

# ============================================================
# CELL 1: Configuration
# ============================================================
import os
from pathlib import Path

# ── CHANGE THESE to match your Kaggle dataset name ──
KAGGLE_DATASET = "mfft-dataset"  # e.g., "yourusername/mfft-dataset"
# ─────────────────────────────────────────────────────

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
MODEL_VARIANT = "base"
SEED = 42

# Paths (Kaggle standard)
INPUT_DIR = Path(f"/kaggle/input/{KAGGLE_DATASET}")
WORK_DIR = Path("/kaggle/working")
OUTPUT_DIR = WORK_DIR / "mfft_checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CELL 2: Environment Check
# ============================================================
import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# CELL 3: Install Dependencies (if needed)
# ============================================================
import subprocess
subprocess.run(["pip", "install", "-q", "open_clip_torch", "datasets"], check=False)

# ============================================================
# CELL 4: Load Dataset
# ============================================================
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import Counter

# Find metadata
metadata_path = INPUT_DIR / "metadata.csv"
if not metadata_path.exists():
    # Try alternatives
    for alt in ["train_metadata.csv", "clean_metadata.csv"]:
        candidate = INPUT_DIR / alt
        if candidate.exists():
            metadata_path = candidate
            break

print(f"Metadata: {metadata_path}")
df = pd.read_csv(metadata_path)
print(f"Total rows: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Find images directory
images_dir = INPUT_DIR / "images"
if not images_dir.exists():
    # Try parent or subdirectories
    for candidate in [INPUT_DIR, INPUT_DIR / "data"]:
        if any(candidate.rglob("*.jpg")):
            images_dir = candidate
            break

print(f"Images dir: {images_dir}")

# Verify image files exist
def check_image_exists(filename):
    """Check if image file exists in any subdirectory."""
    candidates = [
        images_dir / filename,
        images_dir / "images" / filename,
        images_dir / str(filename).replace("\\", "/"),
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return str(c)
    return None

# Build sample list
print("\nBuilding sample list...")
samples = []
missing = 0
for _, row in df.iterrows():
    img_path = check_image_exists(row["filename"])
    if img_path:
        label = 0 if row["label"] == "real" else 1
        samples.append((img_path, label))
    else:
        missing += 1

print(f"Found: {len(samples)} images, Missing: {missing}")
label_counts = Counter(label for _, label in samples)
print(f"Real: {label_counts[0]}, AI: {label_counts[1]}")

# Train/Val/Test split (80/10/10)
paths = [s[0] for s in samples]
labels = [s[1] for s in samples]

train_idx, temp_idx = train_test_split(range(len(samples)), test_size=0.2, stratify=labels, random_state=SEED)
temp_labels = [labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=SEED)

print(f"\nSplits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

# ============================================================
# CELL 5: Dataset & Transforms
# ============================================================
class AIDetectionDataset(Dataset):
    def __init__(self, indices, samples, size=224, augment=True):
        self.samples = [(samples[i][0], samples[i][1]) for i in indices]
        self.size = size

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(size + 16),
                transforms.RandomResizedCrop(size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10, fill=128),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size + 16),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), label
        except Exception:
            # Retry with a different sample
            for offset in range(1, min(10, len(self.samples))):
                retry_idx = (idx + offset) % len(self.samples)
                try:
                    img = Image.open(self.samples[retry_idx][0]).convert("RGB")
                    return self.transform(img), self.samples[retry_idx][1]
                except:
                    continue
            # Fallback blank image
            blank = torch.zeros(3, self.size, self.size)
            return blank, label

train_dataset = AIDetectionDataset(train_idx, samples, size=IMAGE_SIZE, augment=True)
val_dataset = AIDetectionDataset(val_idx, samples, size=IMAGE_SIZE, augment=False)
test_dataset = AIDetectionDataset(test_idx, samples, size=IMAGE_SIZE, augment=False)

# Weighted sampler for class balance
train_labels = [s[1] for s in train_dataset.samples]
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[l] for l in train_labels]
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

NUM_WORKERS = 2 if torch.cuda.is_available() else 0
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
print(f"Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

# ============================================================
# CELL 6: MFFT Model (Full Architecture from model/src/model.py)
# ============================================================
print("\n=== Building MFFT Model ===")
import torch.nn as nn
import torch.nn.functional as F

BAND_CONFIGS = {
    3: [(0.0, 0.15), (0.15, 0.45), (0.45, 1.0)],
    4: [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)],
}

class FrequencyDecomposition(nn.Module):
    def __init__(self, num_bands=3):
        super().__init__()
        self.bands = BAND_CONFIGS.get(num_bands, BAND_CONFIGS[3])
        self.num_bands = len(self.bands)
        self._mask_cache = {}

    def _get_masks(self, H, W, device):
        key = (H, W, str(device))
        if key in self._mask_cache:
            return self._mask_cache[key]

        ny, nx = H // 2, W // 2
        r_max = min(ny, nx)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        dist = torch.sqrt((y_grid - ny).float() ** 2 + (x_grid - nx).float() ** 2)

        masks = []
        for lo, hi in self.bands:
            r_low, r_high = r_max * lo, r_max * hi
            masks.append(((dist >= int(r_low)) & (dist < int(r_high))).float())
        stacked = torch.stack(masks).unsqueeze(1).unsqueeze(1)
        self._mask_cache[key] = stacked
        return stacked

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_shifted = torch.fft.fftshift(x_fft)
        masks = self._get_masks(H, W, x.device)
        outputs = []
        for i in range(self.num_bands):
            filtered = x_shifted * masks[i]
            band = torch.fft.ifft2(torch.fft.ifftshift(filtered), norm='ortho').real
            outputs.append(band)
        return outputs


class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feat_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU())
        self.blocks = nn.ModuleList([
            self._make_block(32, 64, 3, 2),
            self._make_block(64, 128, 3, 2),
            self._make_block(128, 256, 3, 2)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, feat_dim), nn.LayerNorm(feat_dim))

    def _make_block(self, cin, cout, k, s):
        return nn.Sequential(
            nn.Conv2d(cin, cin, k, stride=s, padding=k//2, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.GELU())

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))


class FrequencyGuidedAttention(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.freq_gate = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.ReLU(),
            nn.Linear(dim // 4, dim), nn.Sigmoid())

    def forward(self, x, freq_weights):
        gate = self.freq_gate(x)
        return x * gate * freq_weights


class MFFT(nn.Module):
    def __init__(self, in_channels=3, feat_dim=256, num_bands=3, num_heads=8,
                 num_classes=2):
        super().__init__()
        self.num_bands = num_bands
        self.feat_dim = feat_dim
        self.decomposer = FrequencyDecomposition(num_bands=num_bands)
        self.extractors = nn.ModuleList([
            FrequencyFeatureExtractor(in_channels, feat_dim) for _ in range(num_bands)])
        self.fusion = CrossAttentionFusion(feat_dim, num_heads)
        self.freq_guided_attn = FrequencyGuidedAttention(feat_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim * num_bands),
            nn.Linear(feat_dim * num_bands, feat_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(feat_dim, feat_dim // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_heatmap=False):
        bands = self.decomposer(x)
        features = [extractor(band) for band, extractor in zip(bands, self.extractors)]

        feat_stack = torch.stack(features, dim=1)
        fused = self.fusion(feat_stack)

        freq_magnitudes = torch.stack([
            torch.abs(band).mean(dim=(1, 2, 3)) for band in bands], dim=1)
        freq_weights = F.softmax(freq_magnitudes, dim=1).unsqueeze(-1)
        guided = self.freq_guided_attn(fused, freq_weights)

        B = x.shape[0]
        combined = guided.reshape(B, -1)
        logits = self.classifier(combined)

        if return_heatmap:
            heatmaps = []
            for band in bands:
                h = torch.abs(band).mean(dim=1, keepdim=True)
                h = F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=False)
                heatmaps.append(h)
            return logits, torch.cat(heatmaps, dim=1)
        return logits


def build_mfft(variant="base"):
    configs = {
        "tiny":  {"feat_dim": 128, "num_heads": 4, "num_bands": 3},
        "base":  {"feat_dim": 384, "num_heads": 6, "num_bands": 3},
        "large": {"feat_dim": 768, "num_heads": 12, "num_bands": 4},
    }
    cfg = configs.get(variant, configs["base"])
    return MFFT(**cfg)

model = build_mfft(MODEL_VARIANT).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params:,}")

# ============================================================
# CELL 7: Training Setup
# ============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=min(500, len(train_loader)))
cosine = CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS * len(train_loader), T_mult=2, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[min(500, len(train_loader))])

scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

# ============================================================
# CELL 8: Training Loop
# ============================================================
print(f"\n{'='*60}")
print(f"Training MFFT-{MODEL_VARIANT}")
print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Image: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"{'='*60}\n")

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_acc = 0.0
global_step = 0
start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

for epoch in range(1, EPOCHS + 1):
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
        global_step += 1

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
    all_preds, all_labels, all_probs = [], [], []

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
            all_probs.extend(probs[:, 1].cpu().numpy())

    val_acc = val_correct / val_total * 100
    avg_val_loss = val_loss / len(val_loader)

    # Metrics
    from sklearn.metrics import roc_auc_score
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    elapsed = time.time() - start_epoch if epoch > 1 else 0
    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Train: {avg_train_loss:.4f}/{train_acc:.2f}% | "
          f"Val: {avg_val_loss:.4f}/{val_acc:.2f}% | "
          f"F1: {f1:.2f} AUC: {auc:.4f} | "
          f"Spec: {specificity:.2f}%")

    is_best = val_acc > best_acc
    if is_best:
        best_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_f1": f1,
            "val_auc": auc,
            "history": history,
        }, OUTPUT_DIR / "best.pt")
        print(f"  *** New best (acc={val_acc:.2f}%) ***")

    if epoch % 5 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": {"val_acc": val_acc, "val_f1": f1, "val_auc": auc},
        }, OUTPUT_DIR / f"checkpoint_epoch_{epoch}.pt")

    start_epoch = time.time()

# ============================================================
# CELL 9: Final Test Evaluation
# ============================================================
print(f"\n{'='*60}")
print("FINAL TEST SET EVALUATION")
print(f"{'='*60}")

model.eval()
test_correct = 0
test_total = 0
all_test_preds, all_test_labels, all_test_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)
        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())
        all_test_probs.extend(probs[:, 1].cpu().numpy())

test_acc = test_correct / test_total * 100
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
y_true = np.array(all_test_labels)
y_pred = np.array(all_test_preds)
y_score = np.array(all_test_probs)

cm = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred) * 100
recall = recall_score(y_true, y_pred) * 100
f1 = f1_score(y_true, y_pred) * 100
auc = roc_auc_score(y_true, y_score)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

print(f"Accuracy:    {test_acc:.2f}%")
print(f"Precision:   {precision:.2f}%")
print(f"Recall:      {recall:.2f}%")
print(f"F1-Score:    {f1:.2f}%")
print(f"Specificity: {specificity:.2f}%")
print(f"AUC-ROC:     {auc:.4f}")
print(f"\nConfusion Matrix:")
print(f"              Pred Real  Pred AI")
print(f"Actual Real   {tn:>6d}     {fp:>6d}")
print(f"Actual AI     {fn:>6d}     {tp:>6d}")

# GPU memory stats
if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nPeak GPU Memory: {peak_mem:.2f} GB")

# ============================================================
# CELL 10: Save Results
# ============================================================
import json

results = {
    "model": f"MFFT-{MODEL_VARIANT}",
    "params": n_params,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "best_val_acc": round(best_acc, 2),
    "test_acc": round(test_acc, 2),
    "precision": round(precision, 2),
    "recall": round(recall, 2),
    "f1": round(f1, 2),
    "auc_roc": round(auc, 4),
    "specificity": round(specificity, 2),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "train_samples": len(train_dataset),
    "val_samples": len(val_dataset),
    "test_samples": len(test_dataset),
}

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

with open(OUTPUT_DIR / "history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}")
print(f"\nDownload checkpoints from: {OUTPUT_DIR}")
print(f"  - best.pt (best model)")
print(f"  - results.json (metrics)")
print(f"  - history.json (training curves)")
