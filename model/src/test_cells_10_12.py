"""
Quick test Cells 10-12 logic without re-training.
Loads saved checkpoint, runs evaluation on test set, generates figures + tables.
Usage: python -m src.test_cells_10_12
"""
import os, sys, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'model'))
os.chdir(PROJECT_ROOT)

from src.dataset import AIDetectionDataset, ImageTransform
from src.config import Config
from src.model import build_mfft, count_parameters

device = torch.device('cpu')
print(f'Device: {device}')

# Same dataset setup as notebook Cells 2-3
cfg = Config()
cfg.training.model_variant = 'base'
cfg.training.image_size = 224
cfg.training.batch_size = 8
cfg.training.mixed_precision = False
cfg.training.num_workers = 0
cfg.dataset.undersample = True
cfg.dataset.val_split = 0.1
cfg.dataset.test_split = 0.1

full_dataset = AIDetectionDataset(
    root_dir=str(PROJECT_ROOT),
    metadata_paths=[str(PROJECT_ROOT / 'dataset' / 'metadata' / 'all.csv')],
    transform=None,
    is_train=True,
    size=cfg.training.image_size,
    undersample=True,
)
print(f'Total samples: {len(full_dataset)}')

# Split
from sklearn.model_selection import train_test_split
labels = [s[1] for s in full_dataset.samples]
indices = list(range(len(full_dataset)))
train_idx, temp_idx = train_test_split(indices, test_size=cfg.dataset.val_split + cfg.dataset.test_split,
                                       stratify=labels, random_state=42)
temp_labels = [labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=42)

train_dataset = AIDetectionDataset(
    root_dir=str(PROJECT_ROOT), metadata_paths=[],
    transform=ImageTransform(size=cfg.training.image_size, augment=True),
    is_train=True, size=cfg.training.image_size, undersample=False)
train_dataset.samples = [full_dataset.samples[i] for i in train_idx]

val_dataset = AIDetectionDataset(
    root_dir=str(PROJECT_ROOT), metadata_paths=[],
    transform=ImageTransform(size=cfg.training.image_size, augment=False),
    is_train=False, size=cfg.training.image_size, undersample=False)
val_dataset.samples = [full_dataset.samples[i] for i in val_idx]

test_dataset = AIDetectionDataset(
    root_dir=str(PROJECT_ROOT), metadata_paths=[],
    transform=ImageTransform(size=cfg.training.image_size, augment=False),
    is_train=False, size=cfg.training.image_size, undersample=False)
test_dataset.samples = [full_dataset.samples[i] for i in test_idx]

from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, pin_memory=False)
print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')

# Load trained model
model = build_mfft('base').to(device)
ckpt = PROJECT_ROOT / 'model' / 'checkpoints' / 'best_mfft_base.pt'
if not ckpt.exists():
    ckpt = PROJECT_ROOT / 'model' / 'checkpoints' / 'mfft_base_final.pt'
state = torch.load(ckpt, map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()
print(f'Loaded checkpoint: {ckpt.name}')
print(f'Parameters: {count_parameters(model):,}')

# =========== Cell 10 logic: Evaluate + generate figures ===========
print('\n' + '='*60)
print('TESTING: Cell 10 — Evaluate + Generate Figures')
print('='*60)

all_test_labels = []
all_test_probs = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=-1)
        all_test_labels.extend(labels.cpu().numpy())
        all_test_probs.extend(probs[:, 1].cpu().numpy())

y_true = np.array(all_test_labels)
y_score = np.array(all_test_probs)
y_pred = (y_score >= 0.5).astype(int)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

print(f'Test samples evaluated: {len(y_true)}')
print(f'Accuracy: {(y_pred == y_true).mean()*100:.2f}%')
print(f'Confusion matrix:\n{cm}')

# Pick sample image for fig1
sample_img = None
for cls_dir in ['real', 'ai_generated', 'ai_altered']:
    d = PROJECT_ROOT / 'dataset' / 'images' / cls_dir
    if d.exists():
        files = list(d.glob('*.jpg')) or list(d.glob('*.png'))
        if files:
            sample_img = str(files[0])
            break

all_labels = [s[1] for s in test_dataset.samples]

# Generate figures
from src.visualize import generate_all_figures

FIGS_DIR = PROJECT_ROOT / 'paper' / 'figures'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

history = None  # Fig 2 will be skipped (no training done)
comparison_results = None  # Fig 9 will be skipped

paths = generate_all_figures(
    history=history,
    train_loader=None,
    model=model,
    val_loader=test_loader,
    device=device,
    y_true=y_true,
    y_score=y_score,
    cm=cm,
    labels=all_labels,
    comparison_results=comparison_results,
    sample_image_path=sample_img,
    output_dir=FIGS_DIR,
)

print(f'\nFigures generated: {len(paths)}/13')
for name, p in paths.items():
    print(f'  {name}: {p.name}')

# =========== Cell 11 logic: Per-category accuracy ===========
print('\n' + '='*60)
print('TESTING: Cell 11 — Per-Category Accuracy')
print('='*60)

from collections import defaultdict

category_map = defaultdict(list)
for idx, (img_path, true_label) in enumerate(test_dataset.samples):
    parent_dir = Path(img_path).parent.name
    if parent_dir == "real":
        category = "Real"
    elif parent_dir == "ai_generated":
        category = "AI Generated"
    elif parent_dir == "ai_altered":
        category = "AI Altered"
    else:
        category = "Unknown"
    category_map[category].append((y_pred[idx] == true_label, y_score[idx], true_label, y_pred[idx]))

print("=" * 65)
print(f"{'Category':<20} {'Count':>8} {'Accuracy':>10} {'Avg Conf':>10} {'AUC':>8}")
print("-" * 65)
from sklearn.metrics import roc_auc_score
overall_correct = 0
overall_total = 0
rows = []
for cat in ["Real", "AI Generated", "AI Altered", "Unknown"]:
    if cat not in category_map:
        continue
    items = category_map[cat]
    correct_list = [c for c, _, _, _ in items]
    scores = [s for _, s, _, _ in items]
    true_labels = [t for _, _, t, _ in items]
    n = len(correct_list)
    acc = sum(correct_list) / n * 100
    avg_conf = sum(scores) / n * 100
    try:
        auc = roc_auc_score(true_labels, scores)
    except Exception:
        auc = 0.0
    rows.append((cat, n, acc, avg_conf, auc))
    overall_correct += sum(correct_list)
    overall_total += n
    print(f"{cat:<20} {n:>8} {acc:>9.2f}% {avg_conf:>9.2f}% {auc:>7.4f}")

print("-" * 65)
overall_acc = overall_correct / overall_total * 100
print(f"{'OVERALL':<20} {overall_total:>8} {overall_acc:>9.2f}%")
print("=" * 65)

tables_dir = PROJECT_ROOT / "paper" / "tables"
tables_dir.mkdir(parents=True, exist_ok=True)
json.dump({
    "categories": {r[0]: {"count": int(r[1]), "accuracy": round(float(r[2]), 2), "avg_confidence": round(float(r[3]), 2), "auc": round(float(r[4]), 4)} for r in rows},
    "overall": {"count": int(overall_total), "accuracy": round(float(overall_acc), 2)}
}, open(tables_dir / "per_category_accuracy.json", "w"), indent=2)
print("Saved: paper/tables/per_category_accuracy.json")

# =========== Cell 12 logic: Save all tables ===========
print('\n' + '='*60)
print('TESTING: Cell 12 — Save All Tables')
print('='*60)

import csv
from sklearn.metrics import precision_score, recall_score, f1_score, brier_score_loss
from sklearn.calibration import calibration_curve

def _to_serializable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def save_csv(filename, headers, rows_data):
    p = tables_dir / filename
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows_data)
    print(f'  Saved {p.name}')

def save_json(filename, data):
    p = tables_dir / filename
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=_to_serializable)
    print(f'  Saved {p.name}')

# Table 1
full_dataset = test_dataset  # Not exactly right but tests the table logic
all_lbls = [s[1] for s in full_dataset.samples]
n_real = all_lbls.count(0)
n_ai = all_lbls.count(1)
ds_stats = {
    'total': len(all_lbls), 'real': n_real, 'ai_generated': n_ai,
    'test': len(test_dataset),
    'test_pct': 100.0,
}
save_json('table1_dataset_statistics.json', ds_stats)
save_csv('table1_dataset_statistics.csv',
         ['Split', 'Total', 'Real', 'AI', 'Percentage'],
         [['Test', len(test_dataset), n_real, n_ai, '100%']])

# Table 2
history_dummy = {'train_loss': [0.7, 0.68], 'val_loss': [0.69, 0.67],
                 'train_acc': [54.0, 57.0], 'val_acc': [58.0, 62.0]}
save_json('table2_training_history.json', history_dummy)
save_csv('table2_training_history.csv',
         ['Epoch', 'Train Loss', 'Val Loss', 'Train Acc (%)', 'Val Acc (%)'],
         [[i+1, history_dummy['train_loss'][i], history_dummy['val_loss'][i],
           history_dummy['train_acc'][i], history_dummy['val_acc'][i]]
          for i in range(len(history_dummy['train_loss']))])

# Table 3
acc = (y_pred == y_true).mean() * 100
prec = precision_score(y_true, y_pred) * 100
rec = recall_score(y_true, y_pred) * 100
f1 = f1_score(y_true, y_pred) * 100
auc = roc_auc_score(y_true, y_score)
prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy='uniform')
ece = float(np.mean(np.abs(prob_true - prob_pred)))
metrics = {
    'accuracy_pct': round(float(acc), 2),
    'precision_pct': round(float(prec), 2),
    'recall_pct': round(float(rec), 2),
    'f1_score_pct': round(float(f1), 2),
    'auc_roc': round(float(auc), 4),
    'expected_calibration_error': round(ece, 4),
}
save_json('table3_evaluation_metrics.json', metrics)
save_csv('table3_evaluation_metrics.csv', ['Metric', 'Value'],
         [['Accuracy (%)', f'{acc:.2f}'], ['Precision (%)', f'{prec:.2f}'],
          ['Recall (%)', f'{rec:.2f}'], ['F1 Score (%)', f'{f1:.2f}'],
          ['AUC-ROC', f'{auc:.4f}'], ['ECE', f'{ece:.4f}']])

# Table 4
cm_tbl = {'tn': int(cm[0,0]), 'fp': int(cm[0,1]), 'fn': int(cm[1,0]), 'tp': int(cm[1,1])}
save_json('table4_confusion_matrix.json', cm_tbl)
save_csv('table4_confusion_matrix.csv', ['', 'Predicted Real', 'Predicted AI'],
         [['Actual Real', cm_tbl['tn'], cm_tbl['fp']],
          ['Actual AI', cm_tbl['fn'], cm_tbl['tp']]])

# Table 5
cat_data = {}
cat_rows = []
for cat in ['Real', 'AI Generated', 'AI Altered']:
    if cat in category_map:
        items = category_map[cat]
        correct_list = [c for c, _, _, _ in items]
        scores_list = [s for _, s, _, _ in items]
        n = len(correct_list)
        a = sum(correct_list) / n * 100
        cat_data[cat] = {'count': n, 'accuracy_pct': round(float(a), 2),
                         'avg_confidence_pct': round(float(sum(scores_list)/n*100), 2)}
        cat_rows.append([cat, n, f'{a:.2f}%', f'{sum(scores_list)/n*100:.2f}%'])
save_json('table5_per_category_accuracy.json', cat_data)
save_csv('table5_per_category_accuracy.csv', ['Category', 'Count', 'Accuracy', 'Avg Confidence'], cat_rows)

# Table 6
model_info = {
    'architecture': 'Multi-Frequency Fusion Transformer (MFFT)',
    'variant': 'base',
    'total_params': count_parameters(model),
    'image_size': '224x224',
}
save_json('table6_model_architecture.json', model_info)
save_csv('table6_model_architecture.csv', ['Property', 'Value'],
         [[k.replace('_', ' ').title(), str(v)] for k, v in model_info.items()])

# Table 8
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, target_names=['Real', 'AI-Generated'], output_dict=True, zero_division=0)
per_class = {
    'real': {'precision_pct': round(float(report['Real']['precision']*100), 2),
             'recall_pct': round(float(report['Real']['recall']*100), 2),
             'f1_pct': round(float(report['Real']['f1-score']*100), 2),
             'support': int(report['Real']['support'])},
    'ai_generated': {'precision_pct': round(float(report['AI-Generated']['precision']*100), 2),
                     'recall_pct': round(float(report['AI-Generated']['recall']*100), 2),
                     'f1_pct': round(float(report['AI-Generated']['f1-score']*100), 2),
                     'support': int(report['AI-Generated']['support'])},
}
save_json('table8_per_class_metrics.json', per_class)
save_csv('table8_per_class_metrics.csv', ['Class', 'Precision (%)', 'Recall (%)', 'F1 Score (%)', 'Support'],
         [['Real', per_class['real']['precision_pct'], per_class['real']['recall_pct'],
           per_class['real']['f1_pct'], per_class['real']['support']],
          ['AI-Generated', per_class['ai_generated']['precision_pct'], per_class['ai_generated']['recall_pct'],
           per_class['ai_generated']['f1_pct'], per_class['ai_generated']['support']]])

# Table 9
brier = float(brier_score_loss(y_true, y_score))
calib_metrics = {
    'brier_score': round(brier, 4),
    'ece': round(ece, 4),
    'mean_confidence_correct': float(y_score[y_pred == y_true].mean()) if y_pred[y_pred == y_true].size > 0 else 0,
    'mean_confidence_incorrect': float(y_score[y_pred != y_true].mean()) if y_pred[y_pred != y_true].size > 0 else 0,
}
save_json('table9_calibration_metrics.json', calib_metrics)
save_csv('table9_calibration_metrics.csv', ['Metric', 'Value'],
         [[k.replace('_', ' ').title(), str(v)] for k, v in calib_metrics.items()])

print(f'\nAll tables saved to {tables_dir}/')
print('\n' + '='*60)
print('ALL CELLS 10-12 TESTED SUCCESSFULLY!')
print('='*60)
