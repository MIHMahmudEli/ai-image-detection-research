"""
Kaggle Notebook: MFFT AI-Generated Image Detection
====================================================
1. Upload real images as Kaggle Dataset first (see instructions below)
2. Create new Kaggle Notebook, add this Dataset as input
3. Enable GPU (Settings → Accelerator → GPU T4 x2)
4. Enable Internet access (Settings → Internet)
5. Copy this entire script into the first cell and run

INSTRUCTIONS - Upload real images as Kaggle Dataset:
  1. From this machine, zip real images:
     cd dataset/images/real && zip -r real_images.zip .
  2. Upload to Kaggle: kaggle.com/datasets/new
  3. Name it "mfft-real-images" (or whatever you choose)
  4. Set the DATASET variable below to match your dataset name
"""

# ============================================================
# CONFIG - CHANGE THESE
# ============================================================
DATASET = "mfft-real-images"  # Your Kaggle dataset name with real images
IMAGE_SIZE = 128  # bitmind AI images are 128x128; real images will resize to match
BATCH_SIZE = 64   # T4 has 16GB VRAM, can handle 64 at 128x128
EPOCHS = 50
LR = 3e-4
MODEL_VARIANT = "base"
HF_TOKEN = None  # Optional: set your HF token for faster downloads

# ============================================================
# IMPORTS
# ============================================================
import os, sys, time, json, math, random, io
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torchvision import transforms

# ============================================================
# SETUP PATHS
# ============================================================
REAL_DIR = Path(f"/kaggle/input/{DATASET}/images/real")
OUTPUT_DIR = Path("/kaggle/working/mfft_checkpoints")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# SEED
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# ============================================================
# DOWNLOAD AI IMAGES FROM HUGGINGFACE
# ============================================================
def download_hf_images():
    """
    Download BigGAN and Glide images from bitmind HuggingFace datasets.
    Returns: dict of {generator_name: list_of_image_paths}
    """
    from datasets import load_dataset
    
    ai_images = {}
    for gen_name, hf_repo in [
        ("BigGAN", "bitmind/GenImage_BigGAN"),
        ("glide", "bitmind/GenImage_glide"),
    ]:
        cache_dir = f"/kaggle/working/hf_cache/{gen_name}"
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"\nLoading {gen_name} from {hf_repo}...")
        t0 = time.time()
        
        ds = load_dataset(hf_repo, split="train", streaming=True, cache_dir=cache_dir)
        paths = []
        count = 0
        extract_dir = Path(f"/kaggle/working/ai_images/{gen_name}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        for i, sample in enumerate(ds):
            img = sample["image"]
            path = sample.get("path", f"{gen_name}_{i:06d}.png")
            out_path = extract_dir / os.path.basename(path)
            if not out_path.exists():
                img.save(out_path)
            paths.append(str(out_path))
            count += 1
            
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1} images ({count/(elapsed or 1):.0f} img/s)")
        
        elapsed = time.time() - t0
        print(f"  Loaded {count} images in {elapsed:.0f}s")
        ai_images[gen_name] = paths
    
    return ai_images

print("\n=== DOWNLOADING AI IMAGES ===")
try:
    ai_image_paths = download_hf_images()
except Exception as e:
    print(f"HF datasets download failed: {e}")
    print("Falling back to manual parquet download...")
    # Fallback: download parquet files and extract manually
    import requests
    import pyarrow.parquet as pq
    
    ai_image_paths = {}
    for gen_name, hf_repo, num_shards in [
        ("BigGAN", "bitmind/GenImage_BigGAN", 8),
        ("glide", "bitmind/GenImage_glide", 21),
    ]:
        extract_dir = Path(f"/kaggle/working/ai_images/{gen_name}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        paths = []
        for shard in range(num_shards):
            url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main/data/train-{shard:05d}-of-{num_shards:05d}.parquet"
            print(f"  Downloading {gen_name} shard {shard+1}/{num_shards}...")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            data = r.content
            
            pf = pq.ParquetFile(io.BytesIO(data))
            for rg in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg)
                struct_arr = table.column("image").combine_chunks()
                for i in range(len(struct_arr)):
                    row = struct_arr[i].as_py()
                    rel_path = row["path"]
                    out_path = extract_dir / os.path.basename(rel_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(row["bytes"])
                    paths.append(str(out_path))
            
            print(f"    -> {len(paths)} images so far")
        
        ai_image_paths[gen_name] = paths

total_ai = sum(len(v) for v in ai_image_paths.values())
print(f"\nTotal AI images: {total_ai}")

# ============================================================
# LOAD REAL IMAGES
# ============================================================
real_paths = sorted([str(p) for p in REAL_DIR.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
print(f"\nReal images: {len(real_paths)}")

# ============================================================
# BUILD DATASET
# ============================================================
class AIDataset(Dataset):
    def __init__(self, real_paths, ai_paths, size=128, augment=True):
        self.samples = [(p, 0) for p in real_paths] + [(p, 1) for p in ai_paths]
        
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5, fill=128),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

from PIL import Image

# Undersample to balance
from sklearn.model_selection import train_test_split

all_ai = []
for v in ai_image_paths.values():
    all_ai.extend(v)

n_real = len(real_paths)
n_ai = len(all_ai)
target = min(n_real, n_ai)
print(f"\nBalancing: {n_real} real, {n_ai} AI -> {target} each")

random.shuffle(real_paths)
random.shuffle(all_ai)
real_balanced = real_paths[:target]
ai_balanced = all_ai[:target]

# Train/val split (80/20 stratified)
real_train, real_val = train_test_split(real_balanced, test_size=0.2, random_state=42)
ai_train, ai_val = train_test_split(ai_balanced, test_size=0.2, random_state=42)

train_paths = real_train + ai_train
val_paths = real_val + ai_val
random.shuffle(train_paths)
random.shuffle(val_paths)

print(f"Train: {len(train_paths)} ({len(real_train)} real, {len(ai_train)} AI)")
print(f"Val:   {len(val_paths)} ({len(real_val)} real, {len(ai_val)} AI)")

train_dataset = AIDataset(real_train, ai_train, size=IMAGE_SIZE, augment=True)
val_dataset = AIDataset(real_val, ai_val, size=IMAGE_SIZE, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ============================================================
# MFFT MODEL
# ============================================================
print("\n=== BUILDING MFFT MODEL ===")

class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class FrequencyDecomp(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        # Learnable frequency filters
        self.low_weight = nn.Parameter(torch.ones(1, embed_dim, 1, 1))
        self.high_weight = nn.Parameter(torch.ones(1, embed_dim, 1, 1))
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # FFT
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft_shifted = torch.fft.fftshift(x_fft)
        
        # Create frequency masks
        cy, cx = H // 2, W // 2
        r = min(H, W) // 4  # low-frequency radius
        Y, X = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
        dist = torch.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        low_mask = (dist <= r).float().reshape(1, 1, H, W)
        high_mask = (dist > r).float().reshape(1, 1, H, W)
        
        # Apply learnable weights
        low = x_fft_shifted * low_mask * self.low_weight
        high = x_fft_shifted * high_mask * self.high_weight
        
        # Inverse FFT
        low_out = torch.fft.ifft2(torch.fft.ifftshift(low), norm='ortho').real
        high_out = torch.fft.ifft2(torch.fft.ifftshift(high), norm='ortho').real
        
        return low_out, high_out

class FrequencyGuidedAttention(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_low = nn.Linear(embed_dim, embed_dim * 3)
        self.qkv_high = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.freq_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        B, N, C = x.shape
        # Low and high frequency paths
        low_x = x
        high_x = x
        
        qkv_l = self.qkv_low(low_x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv_h = self.qkv_high(high_x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        q_l, k_l, v_l = qkv_l[0], qkv_l[1], qkv_l[2]
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        
        attn_l = (q_l @ k_l.transpose(-2, -1)) * self.scale
        attn_l = attn_l.softmax(dim=-1)
        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale
        attn_h = attn_h.softmax(dim=-1)
        
        out_l = (attn_l @ v_l).transpose(1, 2).reshape(B, N, C)
        out_h = (attn_h @ v_h).transpose(1, 2).reshape(B, N, C)
        
        # Frequency-guided fusion
        gate = self.freq_gate(torch.cat([out_l, out_h], dim=-1))
        out = gate * out_l + (1 - gate) * out_h
        
        return self.proj(out)

class MFFTBlock(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.freq_attn = FrequencyGuidedAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = x + self.freq_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MFFT(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=3, num_classes=2,
                 embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.freq_decomp = FrequencyDecomp(embed_dim)
        self.blocks = nn.ModuleList([MFFTBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Frequency decomposition (applied to spatial features)
        low_freq, high_freq = self.freq_decomp(x)
        
        # Patch embed on low + high
        x_low = self.patch_embed(low_freq)
        x_high = self.patch_embed(high_freq)
        x = x_low + x_high  # fused
        
        # Add CLS token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]  # CLS token
        x = self.head(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

model = MFFT(img_size=IMAGE_SIZE, patch_size=16, embed_dim=384, depth=6, num_heads=6)
model = model.to(device)
print(f"Model parameters: {count_parameters(model):,}")

# ============================================================
# TRAINING SETUP
# ============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.05)

# LR scheduler
warmup_steps = 500
warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
cosine = CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS * 100, T_mult=2, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

scaler = GradScaler(enabled=torch.cuda.is_available())

# ============================================================
# TRAINING LOOP
# ============================================================
print(f"\n{'='*60}")
print(f"Training MFFT-{MODEL_VARIANT}")
print(f"{'='*60}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Learning rate: {LR}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"{'='*60}\n")

best_val_acc = 0.0
global_step = 0
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    # TRAIN
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % 1 == 0:
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
    
    # VALIDATE
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
            all_probs.extend(probs.cpu().numpy())
    
    val_acc = val_correct / val_total * 100
    
    # Metrics
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    
    # AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
    except:
        auc = 0.0
    
    elapsed = time.time() - start_time
    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | "
          f"Val Acc: {val_acc:.2f}% | "
          f"F1: {f1:.2f} | AUC: {auc:.4f} | "
          f"Time: {elapsed:.0f}s")
    
    # Save best model
    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "val_f1": f1,
            "val_auc": auc,
        }, OUTPUT_DIR / "best.pt")
        print(f"  *** New best model saved (acc={val_acc:.2f}%) ***")
    
    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": {"val_acc": val_acc, "val_f1": f1, "val_auc": auc},
        }, OUTPUT_DIR / f"checkpoint_epoch_{epoch}.pt")

# ============================================================
# FINAL RESULTS
# ============================================================
print(f"\n{'='*60}")
print(f"TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Best val acc: {best_val_acc:.2f}%")
print(f"Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
print(f"Checkpoints: {OUTPUT_DIR}")
print(f"{'='*60}")

# Save summary
summary = {
    "model": "MFFT",
    "variant": MODEL_VARIANT,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "best_val_acc": best_val_acc,
    "params": count_parameters(model),
    "train_samples": len(train_dataset),
    "val_samples": len(val_dataset),
    "real_images": n_real,
    "ai_images": total_ai,
    "generators": list(ai_image_paths.keys()),
}
with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")
print("\nDownload your checkpoints from /kaggle/working/mfft_checkpoints/")
