import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class FrequencyDecomposition(nn.Module):
    """
    Decomposes image into Low, Mid, and High frequency bands
    using DCT (Discrete Cosine Transform) decomposition.
    """
    def __init__(self, bands: List[Tuple[float, float]] = None):
        super().__init__()
        if bands is None:
            bands = [(0.0, 0.15), (0.15, 0.45), (0.45, 1.0)]
        self.bands = bands
        self.num_bands = len(bands)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, C, H, W = x.shape
        x_dct = torch.fft.fft2(x, norm='ortho')
        x_shifted = torch.fft.fftshift(x_dct)

        ny, nx = H // 2, W // 2
        outputs = []
        for lo, hi in self.bands:
            mask = torch.zeros((H, W), device=x.device)
            r_low = int(min(ny, nx) * lo)
            r_high = int(min(ny, nx) * hi)
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=x.device),
                torch.arange(W, device=x.device),
                indexing='ij'
            )
            dist = torch.sqrt((y_grid - ny)**2 + (x_grid - nx)**2)
            mask = ((dist >= r_low) & (dist < r_high)).float()
            mask = mask.unsqueeze(0).unsqueeze(0)
            filtered = x_shifted * mask
            filtered_ifft = torch.fft.ifftshift(filtered)
            band = torch.fft.ifft2(filtered_ifft, norm='ortho').real
            outputs.append(band)
        return outputs


class FrequencyFeatureExtractor(nn.Module):
    """
    CNN backbone for each frequency band.
    Uses depthwise separable convolutions for efficiency.
    """
    def __init__(self, in_channels: int = 3, feat_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            self._make_block(32, 64, 3, 2),
            self._make_block(64, 128, 3, 2),
            self._make_block(128, 256, 3, 2),
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def _make_block(self, cin: int, cout: int, k: int, s: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(cin, cin, k, stride=s, padding=k//2, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion across frequency bands.
    Each band attends to all others to produce a fused representation.
    """
    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FrequencyGuidedAttention(nn.Module):
    """
    Novel module: uses frequency information to guide spatial attention.
    High-freq regions (edges, textures) get higher attention weights.
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        self.freq_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, freq_weights: torch.Tensor) -> torch.Tensor:
        gate = self.freq_gate(x)
        return x * gate * freq_weights


class MFFT(nn.Module):
    """
    Multi-Frequency Fusion Transformer (MFFT)
    ===========================================
    A novel architecture for AI-generated image detection.

    Key innovations:
    1. Frequency decomposition into Low/Mid/High bands via DCT
    2. Per-band CNN feature extraction
    3. Cross-attention fusion across frequency bands
    4. Frequency-guided spatial attention
    5. Multi-scale feature aggregation

    Input:  (B, 3, H, W)  RGB image
    Output: (B, 2)         logits [real, ai_generated]

    Ablation config (dict):
        spatial_only : bool  — skip frequency decomposition, use raw image
        skip_bands   : list  — band names to exclude: ["low", "mid", "high"]
        fusion_mode  : str   — "attention" | "concat" | "avg" | "max"
        use_fga      : bool  — enable/disable FrequencyGuidedAttention
    """
    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 256,
        num_bands: int = 3,
        num_heads: int = 8,
        num_classes: int = 2,
        ablation: Optional[dict] = None,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.ablation = ablation or {}

        self.decomposer = FrequencyDecomposition()
        self.extractors = nn.ModuleList([
            FrequencyFeatureExtractor(in_channels, feat_dim)
            for _ in range(num_bands)
        ])

        if self.ablation.get("fusion_mode", "attention") == "attention":
            self.fusion = CrossAttentionFusion(feat_dim, num_heads)
        else:
            self.fusion = None

        self.use_fga = self.ablation.get("use_fga", True)
        if self.use_fga:
            self.freq_guided_attn = FrequencyGuidedAttention(feat_dim)
        else:
            self.freq_guided_attn = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim * num_bands),
            nn.Linear(feat_dim * num_bands, feat_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, num_classes),
        )

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

    def forward(self, x: torch.Tensor, return_heatmap: bool = False):
        spatial_only = self.ablation.get("spatial_only", False)
        skip_bands = self.ablation.get("skip_bands", [])
        fusion_mode = self.ablation.get("fusion_mode", "attention")
        band_names = ["low", "mid", "high"]

        if spatial_only:
            bands = [x]
            extractors = self.extractors[:1]
        else:
            all_bands = self.decomposer(x)
            keep_indices = [i for i, name in enumerate(band_names) if name not in skip_bands]
            bands = [all_bands[i] for i in keep_indices]
            extractors = [self.extractors[i] for i in keep_indices]

        features = []
        for band, extractor in zip(bands, extractors):
            feat = extractor(band)
            features.append(feat)

        B = x.shape[0]
        num_active = len(features)

        if num_active == 1 or fusion_mode == "concat":
            fused = torch.cat(features, dim=-1)
            N = num_active
            D = fused.shape[-1] // N
            fused = fused.unsqueeze(1)
        elif fusion_mode == "avg":
            fused = torch.stack(features, dim=0).mean(dim=0)
            N = 1
            D = fused.shape[-1]
            fused = fused.unsqueeze(1)
        elif fusion_mode == "max":
            fused = torch.stack(features, dim=0).max(dim=0).values
            N = 1
            D = fused.shape[-1]
            fused = fused.unsqueeze(1)
        elif fusion_mode == "attention" and self.fusion is not None and num_active > 1:
            feat_stack = torch.stack(features, dim=1)
            fused = self.fusion(feat_stack)
            N = fused.shape[1]
            D = fused.shape[2]
        else:
            fused = torch.cat(features, dim=-1)
            N = num_active
            D = fused.shape[-1] // N
            fused = fused.unsqueeze(1)

        if self.use_fga and self.freq_guided_attn is not None and not spatial_only:
            if not spatial_only:
                all_bands_for_fga = bands
            else:
                all_bands_for_fga = self.decomposer(x)
            freq_magnitudes = torch.stack([
                torch.abs(band).mean(dim=(1, 2, 3))
                for band in all_bands_for_fga
            ], dim=1)
            freq_weights = F.softmax(freq_magnitudes, dim=1).unsqueeze(-1)
            guided = self.freq_guided_attn(fused, freq_weights)
        else:
            guided = fused

        combined = guided.reshape(B, N * D)
        if combined.shape[-1] != self.classifier[0].normalized_shape[0]:
            pad = self.classifier[0].normalized_shape[0] - combined.shape[-1]
            if pad > 0:
                combined = F.pad(combined, (0, pad))

        logits = self.classifier(combined)

        if return_heatmap:
            heatmaps = []
            for band in bands:
                h = torch.abs(band).mean(dim=1, keepdim=True)
                h = F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=False)
                heatmaps.append(h)
            return logits, torch.cat(heatmaps, dim=1)
        return logits


class MFFTWithExplainability(MFFT):
    """
    Extension of MFFT that also produces:
    - Confidence score (0-1)
    - Per-region anomaly heatmap
    - Per-frequency-band contribution scores
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x / 255.0 - self.mean) / self.std
        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        x = self.preprocess(x)
        self.eval()
        logits, heatmaps = self.forward(x, return_heatmap=True)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        return {
            "prediction": preds,
            "real_prob": probs[:, 0],
            "ai_prob": probs[:, 1],
            "confidence": probs.max(dim=-1).values,
            "heatmaps": heatmaps,
            "logits": logits,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_mfft(variant: str = "base", ablation: Optional[dict] = None) -> MFFT:
    configs = {
        "tiny":  {"feat_dim": 128, "num_heads": 4, "num_bands": 3},
        "base":  {"feat_dim": 384, "num_heads": 6, "num_bands": 3},
        "large": {"feat_dim": 768, "num_heads": 12, "num_bands": 4},
    }
    cfg = configs.get(variant, configs["base"])
    model = MFFT(**cfg, ablation=ablation)
    return model
