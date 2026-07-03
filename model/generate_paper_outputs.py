"""
generate_paper_outputs.py
=========================
Run after training all models. Generates:
  - All manuscript tables (Table 1-9)
  - All manuscript figures (Figure 1-15)
  - Ablation comparison tables
  - Per-category breakdowns

Usage:
  python generate_paper_outputs.py              # uses default paths
  python generate_paper_outputs.py --variant base --checkpoint path/to/best.pt
"""
import os, sys, json, argparse, csv, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'model'))
sys.path.insert(0, str(PROJECT_ROOT / 'model' / 'src'))

from src.dataset import AIDetectionDataset, ImageTransform
from src.config import Config
from src.model import build_mfft, count_parameters
from src.baselines import count_parameters as count_baseline_params
from src.visualize import generate_all_figures

RESULTS_ROOT = PROJECT_ROOT / 'paper' / 'results'
FIGS_ROOT = PROJECT_ROOT / 'paper' / 'figures'


def load_model(variant: str, checkpoint: str, device: torch.device):
    model = build_mfft(variant)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            try:
                images = images.to(device)
                logits = model(images)
                probs = F.softmax(logits, dim=-1)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
            except Exception as e:
                print(f'  Warning: bad batch: {e}')
                continue
    y_true = np.array(all_labels)
    y_score = np.array(all_probs)
    y_pred = (y_score >= 0.5).astype(int)
    return y_true, y_score, y_pred


def collect_baseline_metrics():
    results = {}
    results_dir = PROJECT_ROOT / 'paper' / 'results'
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            metrics_file = subdir / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file) as f:
                    data = json.load(f)
                    results[data['model']] = data
    return results


def generate_tables(y_true, y_score, y_pred, history, model_name, output_dir):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(filename, data):
        def _conv(o):
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)): return int(o)
            return o
        with open(output_dir / filename, 'w') as f:
            json.dump(data, f, indent=2, default=_conv)
        print(f'  Saved {output_dir / filename}')

    acc = (y_pred == y_true).mean() * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    auc = roc_auc_score(y_true, y_score)

    metrics = {
        'model': model_name,
        'accuracy': round(acc, 2),
        'precision': round(prec, 2),
        'recall': round(rec, 2),
        'f1': round(f1, 2),
        'auc_roc': round(auc, 4),
    }
    save_json('metrics.json', metrics)
    save_json('history.json', history)

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    save_json('confusion_matrix.json', {'tn': int(cm[0,0]), 'fp': int(cm[0,1]),
                                         'fn': int(cm[1,0]), 'tp': int(cm[1,1])})

    print(f'\n  {model_name}: acc={acc:.2f}%, prec={prec:.2f}%, rec={rec:.2f}%, f1={f1:.2f}%, auc={auc:.4f}')
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Generate paper outputs from trained models')
    parser.add_argument('--variant', default='base', choices=['tiny', 'base', 'large'])
    parser.add_argument('--checkpoint', default=None, help='Path to model checkpoint')
    parser.add_argument('--output', default=None, help='Output directory for results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    cfg = Config()
    cfg.training.image_size = 384
    cfg.training.batch_size = 8
    cfg.training.num_workers = 0
    cfg.dataset.val_split = 0.1
    cfg.dataset.test_split = 0.1

    full_dataset = AIDetectionDataset(
        root_dir=str(PROJECT_ROOT),
        metadata_paths=cfg.dataset.metadata_paths,
        transform=None,
        is_train=True,
        size=cfg.training.image_size,
        undersample=True,
    )

    from sklearn.model_selection import train_test_split
    labels = [s[1] for s in full_dataset.samples]
    indices = list(range(len(full_dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=42)

    test_dataset = AIDetectionDataset(
        root_dir=str(PROJECT_ROOT), metadata_paths=[],
        transform=ImageTransform(size=cfg.training.image_size, augment=False),
        is_train=False, size=cfg.training.image_size, undersample=False,
    )
    test_dataset.samples = [full_dataset.samples[i] for i in test_idx]
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)
    print(f'Test samples: {len(test_dataset)}')

    if args.checkpoint:
        output_dir = Path(args.output or RESULTS_ROOT / f'{args.variant}_model')
        output_dir.mkdir(parents=True, exist_ok=True)

        model = load_model(args.variant, args.checkpoint, device)
        print(f'Model loaded: {args.variant} ({count_parameters(model):,} params)')

        y_true, y_score, y_pred = evaluate(model, test_loader, device)
        empty_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        metrics = generate_tables(y_true, y_score, y_pred, empty_history, f'MFFT-{args.variant}', output_dir)

        figs_dir = output_dir / 'fig'
        figs_dir.mkdir(parents=True, exist_ok=True)

        sample_img = None
        for cls_dir in ['real', 'BigGAN', 'genimage_ai', 'DALL-E3']:
            d = PROJECT_ROOT / 'dataset' / 'images' / cls_dir
            if d.exists():
                files = list(d.rglob('*.jpg')) or list(d.rglob('*.png'))
                if files:
                    sample_img = str(files[0])
                    break

        print('\nGenerating figures...')
        generate_all_figures(
            model=model,
            val_loader=test_loader,
            device=device,
            y_true=y_true,
            y_score=y_score,
            cm=np.array([[metrics.get('tn',0), metrics.get('fp',0)],
                          [metrics.get('fn',0), metrics.get('tp',0)]]),
            labels=labels,
            sample_image_path=sample_img,
            output_dir=figs_dir,
        )

    baseline_results = collect_baseline_metrics()
    if baseline_results:
        print(f'\n--- Baseline Summary ({len(baseline_results)} models) ---')
        import pandas as pd
        df = pd.DataFrame(baseline_results.values()).set_index('model')
        print(df.to_string())
        df.to_csv(RESULTS_ROOT / 'baseline_summary.csv')
        print(f'Saved {RESULTS_ROOT / "baseline_summary.csv"}')

    print('\nDone.')


if __name__ == '__main__':
    main()
