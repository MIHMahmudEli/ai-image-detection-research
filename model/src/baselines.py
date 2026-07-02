"""
Standard baseline models for ablation comparison.
Includes custom (SimpleCNN, LightViT) and torchvision models (ResNet, EfficientNet, ViT, Swin).
"""
import torch
import torch.nn as nn
import torchvision.models as M


class SimpleCNN(nn.Module):
    """Lightweight CNN baseline for AI image detection."""
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class LightViT(nn.Module):
    """Lightweight Vision Transformer baseline."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=256, depth=6, num_heads=8, num_classes=2):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def _wrap_torchvision(model_fn, weights_cls, num_classes=2):
    model = model_fn(weights=weights_cls)
    in_features = model.classifier[1].in_features if hasattr(model, 'classifier') else \
                  model.fc.in_features if hasattr(model, 'fc') else \
                  model.head.in_features
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        model.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(in_features, num_classes))
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'head'):
        model.head = nn.Linear(in_features, num_classes)
    return model


def resnet18(num_classes=2):
    return _wrap_torchvision(M.resnet18, M.ResNet18_Weights.IMAGENET1K_V1, num_classes)


def resnet50(num_classes=2):
    return _wrap_torchvision(M.resnet50, M.ResNet50_Weights.IMAGENET1K_V2, num_classes)


def efficientnet_b0(num_classes=2):
    return _wrap_torchvision(M.efficientnet_b0, M.EfficientNet_B0_Weights.IMAGENET1K_V1, num_classes)


def vit_b_16(img_size=384, num_classes=2):
    model = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1)
    if hasattr(model, 'heads'):
        in_feat = model.heads.head.in_features
        model.heads = nn.Linear(in_feat, num_classes)
    else:
        in_feat = model.head.in_features
        model.head = nn.Linear(in_feat, num_classes)
    return model


def swin_t(num_classes=2):
    model = M.swin_t(weights=M.Swin_T_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


class CLIPBaseline(nn.Module):
    """CLIP ViT-B/32 with a classification head for fine-tuning."""
    def __init__(self, num_classes=2):
        super().__init__()
        import open_clip
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model = self.clip_model.visual
        in_features = self.clip_model.output_dim
        self.clip_model.output_dim = None
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        features = self.clip_model(x)
        return self.head(features)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    models = {
        'SimpleCNN': SimpleCNN(),
        'LightViT': LightViT(),
        'ResNet18': resnet18(),
        'ResNet50': resnet50(),
        'EfficientNet-B0': efficientnet_b0(),
        'ViT-B/16': vit_b_16(),
        'Swin-T': swin_t(),
        'CLIP': CLIPBaseline(),
    }
    x = torch.randn(2, 3, 384, 384)
    for name, model in models.items():
        print(f'{name:20s} params={count_parameters(model):>10,}  out={model(x).shape}')
