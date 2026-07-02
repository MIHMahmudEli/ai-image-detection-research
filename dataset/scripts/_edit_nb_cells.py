import json
from pathlib import Path

path = Path('model/train_mfft.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

# Remove all existing Cell 4b cells (duplicates)
nb['cells'] = [c for c in nb['cells'] if 'Cell 4b' not in ''.join(c['source'][:1])]

# Find index of Cell 4 (Build Model)
insert_at = None
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if 'Cell 4:' in src and 'Build Model' in src:
        insert_at = i + 1
        break

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Cell 4b: Baseline Models Overview\n",
        "from src.baselines import (\n",
        "    SimpleCNN, LightViT, count_parameters,\n",
        "    resnet18, resnet50, efficientnet_b0, vit_b_16, swin_t,\n",
        "    CLIPBaseline,\n",
        ")\n",
        "\n",
        "baseline_models = {\n",
        "    'SimpleCNN': lambda: SimpleCNN(),\n",
        "    'LightViT': lambda: LightViT(depth=4, num_heads=4, embed_dim=192),\n",
        "    'ResNet-18': lambda: resnet18(),\n",
        "    'ResNet-50': lambda: resnet50(),\n",
        "    'EfficientNet-B0': lambda: efficientnet_b0(),\n",
        "    'ViT-B/16': lambda: vit_b_16(img_size=cfg.training.image_size),\n",
        "    'Swin-T': lambda: swin_t(),\n",
        "    'CLIP': lambda: CLIPBaseline(),\n",
        "}\n",
        "\n",
        "x = torch.randn(2, 3, cfg.training.image_size, cfg.training.image_size)\n",
        "print(f\"{'Model':<20} {'Params':>10} {'Output':>10}\")\n",
        "print('-' * 42)\n",
        "for name, fn in baseline_models.items():\n",
        "    m = fn()\n",
        "    p = count_parameters(m)\n",
        "    o = list(m(x).shape)\n",
        "    print(f'{name:<20} {p:>10,}  {str(o):>10}')\n",
        "print()\n",
        "print('To train a specific baseline, replace model in Cell 4 with:')\n",
        "print(\"  model = resnet50().to(device)\")\n",
        "print('Then run Cells 5-9 as-is.')\n",
    ]
}

nb['cells'].insert(insert_at, new_cell)

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Inserted Cell 4b at index {insert_at}. Valid JSON: True')

# Verify
nb2 = json.loads(path.read_text(encoding='utf-8'))
for i, c in enumerate(nb2['cells']):
    if c['cell_type'] == 'code' and c['source']:
        print(f'Cell {i}: {c["source"][0].strip()[:60]}')
