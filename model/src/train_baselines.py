"""
Train baseline models (SimpleCNN, LightViT) for ablation comparison.
Run overnight: python -m src.train_baselines
"""
import os, sys, json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'model'))
os.chdir(PROJECT_ROOT)

from src.dataset import create_dataloaders
from src.config import Config
from src.baselines import SimpleCNN, LightViT, count_parameters

cfg = Config()
cfg.dataset.undersample = True
cfg.dataset.test_split = 0.1

train_loader, val_loader, test_loader, _, _, _ = create_dataloaders(cfg)
device = torch.device('cpu')
NUM_EPOCHS = 10

baselines = {
    'SimpleCNN': SimpleCNN,
    'LightViT': lambda: LightViT(depth=4, num_heads=4, embed_dim=192),
}
results = {}

for name, model_cls in baselines.items():
    print(f'\n{"="*60}')
    print(f'Training {name}...')
    print(f'{"="*60}')

    model = model_cls().to(device)
    n_params = count_parameters(model)
    print(f'Parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

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
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

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
    test_correct = test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=-1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_acc = test_correct / test_total * 100

    results[name] = {
        'params': n_params,
        'val_acc': round(best_acc, 2),
        'test_acc': round(test_acc, 2),
        'training_time_min': round(elapsed / 60, 1),
    }
    print(f'  Best val: {best_acc:.2f}%, Test: {test_acc:.2f}%, Time: {elapsed/60:.1f} min')

out_path = PROJECT_ROOT / 'paper' / 'tables' / 'ablation_results.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {out_path}')
print(json.dumps(results, indent=2))
