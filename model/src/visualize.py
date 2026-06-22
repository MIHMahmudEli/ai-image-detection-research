import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import build_mfft, MFFT, FrequencyDecomposition
from .dataset import AIDetectionDataset

IEEE_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00']
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

FIGS_DIR = Path(__file__).resolve().parent.parent.parent / 'paper' / 'figures'


def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_frequency_decomposition(
    image_path: str,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    _ensure_dir(save_path or FIGS_DIR / 'fig1_frequency_decomposition.png')

    img = Image.open(image_path).convert('RGB')
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    decomposer = FrequencyDecomposition()
    decomposer.eval()
    with torch.no_grad():
        bands = decomposer(img_tensor)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    axes[0, 0].imshow(np.array(img))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    titles = ['Low Frequency\n(0 – 0.15ρ)', 'Mid Frequency\n(0.15 – 0.45ρ)', 'High Frequency\n(0.45 – 1.0ρ)']
    colors = ['Blues', 'Greens', 'Reds']

    for i in range(3):
        ax = axes[0, 1 + i]
        band_np = bands[i].squeeze().cpu().numpy()
        band_np = (band_np - band_np.min()) / (band_np.max() - band_np.min() + 1e-8)
        ax.imshow(band_np.transpose(1, 2, 0))
        ax.set_title(titles[i])
        ax.axis('off')

    axes[1, 0].axis('off')

    for i in range(3):
        ax = axes[1, 1 + i]
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(bands[i].squeeze().cpu().numpy().mean(0))))
        fft_mag = np.log1p(fft_mag)
        ax.imshow(fft_mag, cmap=colors[i])
        ax.set_title(f'{titles[i].split(chr(10))[0]} Spectrum')
        ax.axis('off')

    fig.suptitle('MFFT Frequency Decomposition', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = save_path or FIGS_DIR / 'fig1_frequency_decomposition.png'
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    print(f'Saved: {save_path}')
    return save_path


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    _ensure_dir(save_path or FIGS_DIR / 'fig2_training_history.png')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    epochs = range(1, len(history.get('train_loss', [])) + 1)

    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], 'o-', color=IEEE_COLORS[0], label='Train', linewidth=1.5, markersize=3)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 's-', color=IEEE_COLORS[1], label='Validation', linewidth=1.5, markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if 'train_acc' in history:
        ax.plot(epochs, history['train_acc'], 'o-', color=IEEE_COLORS[0], label='Train', linewidth=1.5, markersize=3)
    if 'val_acc' in history:
        ax.plot(epochs, history['val_acc'], 's-', color=IEEE_COLORS[1], label='Validation', linewidth=1.5, markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training & Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('MFFT Training Progress', fontsize=13, y=1.02)
    plt.tight_layout()

    save_path = save_path or FIGS_DIR / 'fig2_training_history.png'
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    print(f'Saved: {save_path}')
    return save_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    normalize: bool = False,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    _ensure_dir(save_path or FIGS_DIR / 'fig3_confusion_matrix.png')

    if class_names is None:
        class_names = ['Real', 'AI-Generated']

    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(5, 4.5))

    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                ax=ax, annot_kws={'fontsize': 12})

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix', fontsize=13)

    plt.tight_layout()

    save_path = save_path or FIGS_DIR / 'fig3_confusion_matrix.png'
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    print(f'Saved: {save_path}')
    return save_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    _ensure_dir(save_path or FIGS_DIR / 'fig4_roc_curve.png')

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 5))

    ax.plot(fpr, tpr, color=IEEE_COLORS[0], linewidth=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color=IEEE_COLORS[0])

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 − Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = save_path or FIGS_DIR / 'fig4_roc_curve.png'
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    print(f'Saved: {save_path}')
    return save_path


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    _ensure_dir(save_path or FIGS_DIR / 'fig5_pr_curve.png')

    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5.5, 5))

    ax.plot(recall, precision, color=IEEE_COLORS[1], linewidth=2,
            label=f'PR curve (AP = {ap:.3f})')
    ax.fill_between(recall, precision, alpha=0.1, color=IEEE_COLORS[1])

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision (Positive Predictive Value)')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = save_path or FIGS_DIR / 'fig5_pr_curve.png'
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    print(f'Saved: {save_path}')
    return save_path


@torch.no_grad()
def plot_sample_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_samples: int = 16,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    _ensure_dir(save_path or FIGS_DIR / 'fig6_sample_predictions.png')

    model.eval()
    images, labels = next(iter(loader))
    num_samples = min(num_samples, images.size(0))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)

    logits = model(images)
    probs = F.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)

    images_cpu = images.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    cols = 4
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    for i in range(num_samples):
        ax = axes.flat[i]
        img = images_cpu[i] * std + mean
        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)

        true_label = 'Real' if labels[i].item() == 0 else 'AI'
        pred_label = 'Real' if preds[i].item() == 0 else 'AI'
        confidence = probs[i].max().item()
        correct = preds[i].item() == labels[i].item()

        color = '#228B22' if correct else '#DC143C'
        ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.0%})',
                     color=color, fontsize=8)
        ax.axis('off')

    for i in range(num_samples, rows * cols):
        axes.flat[i].axis('off')

    fig.suptitle('Sample Predictions — Green = Correct, Red = Incorrect',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    save_path = save_path or FIGS_DIR / 'fig6_sample_predictions.png'
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    print(f'Saved: {save_path}')
    return save_path


@torch.no_grad()
def plot_anomaly_heatmaps(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_samples: int = 4,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    _ensure_dir(save_path or FIGS_DIR / 'fig7_anomaly_heatmaps.png')

    if not hasattr(model, 'decomposer'):
        print('Warning: model does not have decomposer, using MFFT directly')
        return save_path or FIGS_DIR / 'fig7_anomaly_heatmaps.png'

    model.eval()
    images, labels = next(iter(loader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)

    bands = model.decomposer(images)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images_denorm = torch.clamp(images * std + mean, 0, 1)

    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples))

    band_names = ['Low', 'Mid', 'High']
    band_cmaps = ['Blues', 'Greens', 'Reds']

    for i in range(num_samples):
        img_np = images_denorm[i].permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(img_np)
        label_str = 'Real' if labels[i].item() == 0 else 'AI'
        axes[i, 0].set_title(f'{label_str} Image', fontsize=9)
        axes[i, 0].axis('off')

        for j in range(3):
            heatmap = torch.abs(bands[j][i]).mean(dim=0).cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            axes[i, 1 + j].imshow(img_np, alpha=0.6)
            im = axes[i, 1 + j].imshow(heatmap, cmap=band_cmaps[j], alpha=0.4)
            axes[i, 1 + j].set_title(f'{band_names[j]} Freq Heatmap', fontsize=9)
            axes[i, 1 + j].axis('off')

    for j, name in enumerate(['Image'] + [f'{b} Band' for b in band_names]):
        axes[0, j].set_ylabel(name, fontsize=10, rotation=0, ha='right', va='center',
                              labelpad=15)

    fig.suptitle('Per-Frequency-Band Anomaly Heatmaps', fontsize=13, y=1.02)
    plt.tight_layout()

    save_path = save_path or FIGS_DIR / 'fig7_anomaly_heatmaps.png'
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    print(f'Saved: {save_path}')
    return save_path


def plot_class_distribution(
    labels: List[int],
    class_names: List[str] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    _ensure_dir(save_path or FIGS_DIR / 'fig8_class_distribution.png')

    if class_names is None:
        class_names = ['Real', 'AI-Generated']

    counts = [labels.count(0), labels.count(1)]

    fig, ax = plt.subplots(figsize=(5, 4))

    bars = ax.bar(class_names, counts, color=[IEEE_COLORS[2], IEEE_COLORS[1]],
                  edgecolor='white', linewidth=1.5, width=0.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(counts) * 0.01,
                f'{count:,} ({count / sum(counts) * 100:.1f}%)',
                ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Number of Samples')
    ax.set_title('Dataset Class Distribution')
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    save_path = save_path or FIGS_DIR / 'fig8_class_distribution.png'
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    print(f'Saved: {save_path}')
    return save_path


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    _ensure_dir(save_path or FIGS_DIR / 'fig9_model_comparison.png')

    variants = list(results.keys())
    metrics = list(list(results.values())[0].keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(variants)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (variant, vals) in enumerate(results.items()):
        offsets = x + (i - len(variants) / 2 + 0.5) * width
        ax.bar(offsets, [vals[m] for m in metrics], width,
               label=variant.capitalize(), color=IEEE_COLORS[i])

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylabel('Score (%)' if all(v > 1 for v in list(results.values())[0].values()) else 'Score')
    ax.set_title('Model Variant Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()

    save_path = save_path or FIGS_DIR / 'fig9_model_comparison.png'
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    print(f'Saved: {save_path}')
    return save_path


def plot_architecture_diagram(
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    _ensure_dir(save_path or FIGS_DIR / 'fig10_architecture.png')

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def draw_box(ax, x, y, w, h, text, color='#E8F0FE', text_color='black', fontsize=9):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight='bold')

    def draw_arrow(ax, x1, x2, y):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='#555555',
                                    linewidth=2, connectionstyle='arc3,rad=0'))

    # Input
    draw_box(ax, 0.3, 2.2, 1.2, 1.2, 'Input Image\n224×224×3', '#D4E6F1')

    # Frequency Decomposition
    draw_box(ax, 2.0, 0.2, 1.2, 1.2, 'Low Freq.\n(0–0.15ρ)', '#D5F5E3')
    draw_box(ax, 2.0, 2.2, 1.2, 1.2, 'Mid Freq.\n(0.15–0.45ρ)', '#D5F5E3')
    draw_box(ax, 2.0, 4.2, 1.2, 1.2, 'High Freq.\n(0.45–1.0ρ)', '#D5F5E3')
    ax.text(1.3, 3.6, 'Frequency\nDecomposition\n(FFT+DCT)', ha='center', va='center',
            fontsize=8, color='#555', fontstyle='italic')
    draw_arrow(ax, 1.5, 2.0, 2.8)

    # Feature Extractors
    draw_box(ax, 3.7, 0.2, 1.2, 1.2, 'CNN\nExtractor', '#FADBD8')
    draw_box(ax, 3.7, 2.2, 1.2, 1.2, 'CNN\nExtractor', '#FADBD8')
    draw_box(ax, 3.7, 4.2, 1.2, 1.2, 'CNN\nExtractor', '#FADBD8')
    for y in [0.8, 2.8, 4.8]:
        draw_arrow(ax, 3.2, 3.7, y)

    # Cross-Attention Fusion
    draw_box(ax, 5.4, 2.2, 1.5, 1.2, 'Cross-Attention\nFusion (8 heads)', '#F9E79F')
    for y in [0.8, 2.8, 4.8]:
        draw_arrow(ax, 4.9, 5.4, y)

    # Frequency-Guided Attention
    draw_box(ax, 7.4, 2.2, 1.5, 1.2, 'Frequency-Guided\nAttention', '#D2B4DE')
    draw_arrow(ax, 6.9, 7.4, 2.8)

    # Classifier
    draw_box(ax, 9.4, 2.2, 1.2, 1.2, 'MLP\nClassifier', '#F0B27A')
    draw_arrow(ax, 8.9, 9.4, 2.8)

    # Output
    draw_box(ax, 11.1, 1.8, 1.2, 1.8, 'Real\nvs\nAI-Generated', '#A9DFBF')
    draw_arrow(ax, 10.6, 11.1, 2.8)

    ax.set_title('MFFT Architecture Overview', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()

    save_path = save_path or FIGS_DIR / 'fig10_architecture.png'
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    print(f'Saved: {save_path}')
    return save_path


def generate_all_figures(
    model: Optional[torch.nn.Module] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
    history: Optional[Dict] = None,
    y_true: Optional[np.ndarray] = None,
    y_score: Optional[np.ndarray] = None,
    cm: Optional[np.ndarray] = None,
    labels: Optional[List[int]] = None,
    comparison_results: Optional[Dict] = None,
    sample_image_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    show: bool = False,
) -> Dict[str, Path]:
    paths = {}

    if output_dir:
        global FIGS_DIR
        FIGS_DIR = Path(output_dir)
        FIGS_DIR.mkdir(parents=True, exist_ok=True)

    print('Generating Figure 1: Frequency Decomposition...')
    if sample_image_path and Path(sample_image_path).exists():
        paths['fig1'] = plot_frequency_decomposition(sample_image_path, show=show)
    else:
        print('  Skipped: no sample image provided')

    print('Generating Figure 2: Training History...')
    if history:
        paths['fig2'] = plot_training_history(history, show=show)
    else:
        print('  Skipped: no training history provided')

    print('Generating Figure 3: Confusion Matrix...')
    if cm is not None:
        paths['fig3'] = plot_confusion_matrix(cm, show=show)
    else:
        print('  Skipped: no confusion matrix provided')

    print('Generating Figure 4: ROC Curve...')
    if y_true is not None and y_score is not None:
        paths['fig4'] = plot_roc_curve(y_true, y_score, show=show)
    else:
        print('  Skipped: no y_true/y_score provided')

    print('Generating Figure 5: Precision-Recall Curve...')
    if y_true is not None and y_score is not None:
        paths['fig5'] = plot_pr_curve(y_true, y_score, show=show)
    else:
        print('  Skipped: no y_true/y_score provided')

    print('Generating Figure 6: Sample Predictions...')
    if model is not None and val_loader is not None and device is not None:
        paths['fig6'] = plot_sample_predictions(model, val_loader, device, show=show)
    else:
        print('  Skipped: model/loader/device not provided')

    print('Generating Figure 7: Anomaly Heatmaps...')
    if model is not None and val_loader is not None and device is not None:
        paths['fig7'] = plot_anomaly_heatmaps(model, val_loader, device, show=show)
    else:
        print('  Skipped: model/loader/device not provided')

    print('Generating Figure 8: Class Distribution...')
    if labels is not None:
        paths['fig8'] = plot_class_distribution(labels, show=show)
    else:
        print('  Skipped: no labels provided')

    print('Generating Figure 9: Model Comparison...')
    if comparison_results:
        paths['fig9'] = plot_model_comparison(comparison_results, show=show)
    else:
        print('  Skipped: no comparison results provided')

    print('Generating Figure 10: Architecture Diagram...')
    paths['fig10'] = plot_architecture_diagram(show=show)

    print(f'\nAll figures saved to {FIGS_DIR}/')
    return paths


if __name__ == '__main__':
    plot_architecture_diagram(show=True)
