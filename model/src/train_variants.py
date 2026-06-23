"""
Train all MFFT variants (tiny, base, large) and collect comparison metrics.
Run overnight: python -m src.train_variants
"""
import os, sys, json, time, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'model'))
os.chdir(PROJECT_ROOT)

from src.dataset import AIDetectionDataset, ImageTransform, create_dataloaders
from src.config import Config
from src.model import build_mfft, count_parameters

cfg = Config()
cfg.dataset.undersample = True
cfg.dataset.test_split = 0.1

train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(cfg)
device = torch.device('cpu')
NUM_EPOCHS = 10  # fewer epochs for comparison; enough to see trends

variants = ['tiny', 'base', 'large']
results = {}

for variant in variants:
    print(f'\n{"="*60}')
    print(f'Training MFFT-{variant}...')
    print(f'{"="*60}')

    model = build_mfft(variant).to(device)
    n_params = count_parameters(model)
    print(f'Parameters: {n_params:,}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=min(500, len(train_loader)))
    cosine = CosineAnnealingWarmRestarts(optimizer, T_0=NUM_EPOCHS * len(train_loader),
                                         T_mult=2, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[min(500, len(train_loader))])

    best_acc = 0
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total * 100
        print(f'  Epoch {epoch+1}: train={train_acc:.2f}%, val={val_acc:.2f}%')
        if val_acc > best_acc:
            best_acc = val_acc

    elapsed = time.time() - start_time
    print(f'  Best val: {best_acc:.2f}%, Time: {elapsed/60:.1f} min')

    # Test evaluation
    test_correct = test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=-1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_acc = test_correct / test_total * 100

    results[variant] = {
        'params': n_params,
        'val_acc': round(best_acc, 2),
        'test_acc': round(test_acc, 2),
        'training_time_min': round(elapsed / 60, 1),
    }
    print(f'  Test acc: {test_acc:.2f}%')

out_path = PROJECT_ROOT / 'paper' / 'tables' / 'model_comparison.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {out_path}')
print(json.dumps(results, indent=2))
