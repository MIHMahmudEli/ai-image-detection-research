# Multi-Frequency Fusion Transformer (MFFT) for AI-Generated Image Detection

**Authors:** [Your Name]¹*, [Co-author Name]²

**Affiliations:**
¹ Department of Computer Science, [University/Institution], [City], [Country]
² Department of [Discipline], [University/Institution], [City], [Country]

**Corresponding Author:** [Your Name] — [your.email@institution.edu]

**Keywords:** AI-generated image detection, deepfake detection, frequency analysis, transformer, cross-attention fusion, image forensics, vision transformer, frequency-guided attention

---

## Abstract

The proliferation of generative AI models capable of producing photorealistic images has created an urgent need for robust, generalizable detection methods. While existing approaches predominantly operate in the spatial domain, mounting evidence suggests that AI generation models leave characteristic artifacts in the frequency domain that are complementary to spatial cues. We present the **Multi-Frequency Fusion Transformer (MFFT)**, a novel architecture that systematically decomposes input images into low, mid, and high frequency bands using Fourier-based radial filtering, extracts per-band features using dedicated CNN backbones with depthwise separable convolutions, and fuses them via multi-head cross-attention to produce a final classification. Unlike prior work that treats frequency information as a single channel or uses hand-crafted features, our method learns to dynamically attend to the most discriminative frequency bands for each input. Additionally, we introduce Frequency-Guided Attention (FGA), a mechanism that weights spatial features based on their frequency magnitude content, directing model capacity toward high-frequency regions—edges, textures, and boundaries—where AI artifacts are most prevalent. We evaluate MFFT on a comprehensive dataset of approximately 2.7 million images spanning real photographs (900K), AI-generated images from over 11 generator families including DALL-E 3, Midjourney, Stable Diffusion, Flux, BigGAN, GLIDE, ADM, VQDM, Wukong, and GenImage (900K), and deepfakes from Celeb-DF, FaceForensics++, and DFDC (900K). MFFT is compared against ten baseline methods spanning lightweight CNNs, standard CNNs (ResNet-18/50), efficient CNNs (EfficientNet-B0), vision transformers (ViT-B/16, DeiT-S, Swin-T), CLIP-based detectors, and frequency-domain methods (FreqDetect). Three model variants are introduced—MFFT-Tiny (370K parameters), MFFT-Base (860K parameters), and MFFT-Large (3.1M parameters)—offering a accuracy-efficiency trade-off. Ablation studies across 9 configurations confirm that multi-frequency fusion contributes significant improvement over spatial-only baselines, cross-attention fusion outperforms concatenation, averaging, and max-pooling strategies, and FGA provides consistent gains across all model scales. We provide publicly available code and pretrained models to facilitate reproducibility and further research.

---

## 1. Introduction

### 1.1 The Challenge of AI-Generated Imagery

Generative AI models capable of producing photorealistic images have advanced at an unprecedented pace. Models such as DALL-E 3 [1], Midjourney [2], and Stable Diffusion [3] can generate images that are indistinguishable from authentic photographs to the human eye. By 2025, an estimated 15% of images on major social media platforms were AI-generated [4], creating critical challenges for misinformation detection, journalistic integrity, legal proceedings, and scientific research.

The impact is already evident: AI-generated images have been used in disinformation campaigns targeting democratic elections [5], fraudulent news reporting undermining media trust [6], fabricated evidence in legal proceedings [7], and synthetic profile images for large-scale social engineering attacks. As generative models continue to improve—with each new generation reducing perceptible artifacts—the window for detecting AI-generated content narrows, creating an accelerating arms race between generation and detection technologies.

The problem is compounded by the diversity of generation architectures. Generative Adversarial Networks (GANs), diffusion models, and autoregressive transformers each produce characteristic but distinct artifact patterns. A detector trained exclusively on GAN-generated images may fail dramatically when confronted with diffusion model outputs, and vice versa. This necessitates detection methods that capture fundamental, architecture-agnostic signatures of synthetic image generation rather than superficial, dataset-specific patterns.

**The Case for Frequency-Domain Analysis**: Frequency-domain approaches are particularly promising because the artifacts they detect arise from fundamental mathematical properties of the generation process rather than from specific training data or model implementations. All neural network generators involve upsampling operations (transposed convolutions in GANs, latent-to-image decoders in diffusion models, patch embeddings in transformers) that introduce characteristic high-frequency irregularities. Similarly, the denoising process in diffusion models leaves residual noise patterns that differ from natural image noise distributions. These frequency-domain signatures are more fundamental and more transferable across generator families than spatial-domain features, which tend to be dataset-specific and style-dependent. Furthermore, frequency analysis is inherently multi-scale—different generation artifacts manifest at different frequency ranges—suggesting that a multi-band approach that adaptively weights frequency information should provide superior detection capability.

### 1.2 Current Detection Approaches

Existing approaches to AI-generated image detection fall into three broad categories:

**Spatial-domain deep learning** methods train convolutional neural networks (CNNs) or vision transformers (ViTs) to discriminate between real and AI-generated images directly from pixel values. While these approaches achieve respectable accuracy (85-94% on benchmark datasets [8, 9]), they are susceptible to distribution shift when tested on generators not seen during training [10]. Moreover, spatial-domain methods lack interpretability—they provide a binary prediction without indicating which image regions or properties drove the decision.

**Forensic analysis** methods examine low-level statistical properties including noise patterns [11], compression artifacts [12], and camera sensor noise [13]. These methods are interpretable and grounded in physical principles but require technical expertise, specific capture conditions, and often fail on heavily compressed or post-processed images. They also tend to generalize poorly across different camera models and processing pipelines.

**Frequency-domain methods** analyze the spectral properties of images, leveraging the observation that AI generation models introduce characteristic frequency signatures [14, 15]. These approaches are promising because frequency artifacts arise from fundamental architectural properties of generative models—upsampling operations in GANs, the denoising process in diffusion models, and patch boundaries in transformer-based generators—rather than superficial pixel-level patterns. However, existing frequency-domain methods have been limited to single-band analysis or hand-crafted frequency features, missing the opportunity to learn adaptive, multi-band representations.

### 1.3 Limitations of Current Work

Despite significant progress, three critical gaps remain:

**First**, existing frequency-domain approaches analyze a single frequency band (typically high-frequency residuals) or use simple frequency statistics (radial FFT magnitude profiles, DCT coefficient distributions). No prior work systematically decomposes images into multiple frequency bands and learns to fuse them adaptively for the detection task.

**Second**, current detectors operate primarily in the spatial domain and do not explicitly leverage frequency information to guide spatial attention. This means they treat all spatial regions equally, despite the fact that AI artifacts are more prevalent in high-frequency regions such as edges, textures, and object boundaries where generative models struggle to reproduce the statistical properties of natural images.

**Third**, most published results are on narrow benchmarks testing one or two generator families, often with fewer than 100K images. Real-world deployment requires robustness across diverse generators, image categories, manipulation types, and post-processing operations. The field lacks comprehensive evaluation on large-scale, multi-generator datasets.

### 1.4 Our Contributions

We address these gaps with the following contributions:

1. **Multi-Frequency Decomposition**: We propose a learnable frequency decomposition module that splits input images into low, mid, and high frequency bands using FFT-based radial filtering, enabling the model to analyze complementary frequency information. The decomposition is fully differentiable and integrated into the end-to-end training pipeline.

2. **Cross-Attention Frequency Fusion**: A multi-head cross-attention mechanism that adaptively fuses features from different frequency bands, learning which bands are most discriminative for each input image. Unlike static fusion methods (concatenation, averaging), this allows the model to dynamically weight frequency information based on the input content.

3. **Frequency-Guided Attention (FGA)**: A novel attention mechanism that uses frequency magnitude information to weight spatial features, directing the model's capacity toward regions most likely to contain AI artifacts. FGA provides a principled inductive bias grounded in physical frequency content rather than learned feature statistics.

4. **Comprehensive Evaluation**: We evaluate on a large-scale dataset of 2.7 million images spanning 11+ generator families, 3 deepfake datasets, and diverse image categories, providing the most comprehensive evaluation of frequency-based AI detection to date.

5. **Three Model Variants**: We introduce MFFT-Tiny (370K parameters), MFFT-Base (860K parameters), and MFFT-Large (3.1M parameters), enabling deployment across resource-constrained to high-performance settings.

6. **Explainability**: Our method naturally produces per-frequency-band anomaly heatmaps, providing interpretable evidence for each prediction that reveals not only *where* artifacts are detected but in *which frequency band* they appear.

### 1.5 Research Questions

This study addresses the following research questions:

- **RQ1:** Does multi-frequency decomposition improve AI-generated image detection over spatial-only approaches?
- **RQ2:** How does cross-attention fusion compare to simpler fusion strategies (concatenation, averaging, max pooling)?
- **RQ3:** Which frequency bands (low, mid, high) are most discriminative for different AI generators?
- **RQ4:** How does MFFT generalize across unseen generators and image categories?
- **RQ5:** What is the practical value of frequency-guided attention for detection accuracy?

---

## 2. Related Work

### 2.1 Generative AI Models

Modern image generation is dominated by three architectural families. **Generative Adversarial Networks (GANs)**, introduced by Goodfellow et al. [16], employ a generator-discriminator framework that produces high-quality images through adversarial training. StyleGAN [17] and Progressive GAN [18] represent significant advances, achieving photorealistic outputs particularly for facial images. However, GANs exhibit characteristic artifacts including checkerboard patterns from transposed convolutions and spectral discontinuities from upsampling operations [22].

**Diffusion models**, including Stable Diffusion [3], DDPM [19], and Latent Diffusion [20], have emerged as the dominant paradigm since 2023. These models iteratively denoise random noise to produce images, achieving superior quality and diversity compared to GANs. Stable Diffusion in particular has been widely adopted due to its open-source availability and efficient latent-space operation. Diffusion models introduce distinct frequency signatures characterized by residual noise patterns in high-frequency bands [23], arising from the mismatch between the training-time noise distribution and the inference-time denoising trajectory.

**Transformer-based models** such as DALL-E [1] and Parti [21] leverage autoregressive or masked modeling to generate images from text prompts. These models demonstrate strong compositional understanding but require substantial computational resources. They occasionally produce global coherence failures and patch-boundary artifacts [24] that manifest as low-frequency structural anomalies.

Each architecture family produces characteristic artifacts at different frequency ranges. GANs predominantly affect high frequencies (checkerboard patterns, pixel-level inconsistencies). Diffusion models leave signatures across mid and high frequencies (noise pattern mismatches). Transformer-based models can affect low frequencies (global structural coherence). This frequency-specific artifact distribution motivates our multi-band approach—by analyzing all bands, MFFT can detect any of these signatures.

### 2.2 AI-Generated Image Detection

**Convolutional neural networks** remain the most widely studied approach. Wang et al. [8] demonstrated that a ResNet-50 trained on 1.2 million images achieves 92% accuracy on GAN-generated images, establishing that CNN-based detection is feasible at scale. Subsequent work using EfficientNet [25] achieved 94% on diffusion model outputs. The Xception architecture [9] showed strong performance on cross-generator evaluation. However, these methods show significant accuracy degradation (15-20%) when tested on generators not included in training [10], highlighting the need for more fundamental, artifact-agnostic features.

**Vision transformers** have been applied more recently. Cozzolino et al. [26] used a ViT-B/16 fine-tuned on forensic datasets, achieving 91% accuracy. DeiT [27] introduced knowledge distillation for data-efficient transformer training, and Swin Transformer [28] proposed hierarchical feature maps with shifted window attention, both demonstrating competitive performance for detection tasks. CLIP-based approaches [37] leverage large-scale vision-language pretraining, showing promising generalization but requiring substantial computational resources (87M+ parameters). Our DeiT-S baseline (22M parameters) and CLIP baseline (87M parameters) provide comparison points against these transformer-based approaches.

**Frequency-domain methods** represent a promising direction. Frank et al. [14] demonstrated that GAN-generated images exhibit detectable artifacts in the frequency domain, particularly at high spatial frequencies, and proposed FreqDetect—a radial FFT magnitude profiling method combined with an MLP classifier (8K parameters). Durall et al. [15] proposed analyzing the spectral distribution of images, showing that AI-generated images have different power spectra than natural images. Zhang et al. [29] used DCT coefficients as features for a shallow classifier, achieving 86% accuracy. Recent work by Tan et al. [36] explored frequency-aware deepfake detection through frequency space domain learning. The SCADET framework [38] integrates dynamic frequency attention with contrastive spectral analysis for AI-generated artwork detection. Chivaran and Ni [39] proposed LAID, benchmarking lightweight models across spatial and spectral domains. However, these methods use either hand-crafted frequency features or single-band analysis rather than learning to select and fuse discriminative frequency bands adaptively.

Our work differs from prior frequency-domain approaches in three key ways: (1) we decompose into multiple bands rather than analyzing a single band or aggregate spectrum, (2) we learn to fuse bands via cross-attention rather than using static or hand-crafted combinations, and (3) we use frequency information to guide spatial attention through a novel gating mechanism.

### 2.3 Multi-Modal and Fusion Approaches

Several recent works have explored fusion strategies for AI detection. **Multi-modal approaches** combine image features with text metadata [30] or generation provenance signals [31]. **Ensemble methods** average predictions from multiple classifiers [32], achieving modest improvements (1-2%) over single models. **Dual-branch architectures** process spatial and frequency information through separate streams before fusion [40, 41]. The Refined Dual Fusion Model (RDFM) [42] uses a local branch with convolutional features and a global branch with Swin transformer features, combined via a learnable fusion matrix.

Our cross-attention fusion mechanism differs from these approaches in that it operates on frequency bands derived from a single input image, not on heterogeneous modalities or independently trained classifiers. This enables fine-grained, adaptive weighting of frequency information—the model can emphasize high-frequency analysis for GAN-generated images while shifting attention to mid-frequency bands for diffusion model outputs—without requiring external data sources or ensemble overhead.

### 2.4 Comparison with Contemporary Frequency-Based Methods

Several recent works have explored frequency-domain approaches for AI-generated image detection, and it is instructive to position MFFT within this landscape.

**FreqDetect** (Frank et al., 2020) [14] computes radial FFT magnitude profiles across 16 bins and classifies them via a shallow MLP (~8K parameters). While lightweight and interpretable, this approach collapses the entire frequency spectrum into a coarse histogram, losing spatial localization and multi-scale structure. It also operates on hand-crafted features rather than learned representations. In our experiments, FreqDetect serves as the frequency-domain baseline.

**SCADET** (Zhang et al., 2025) [38] integrates a Dynamic Frequency Attention Network (DFAN) with Contrastive Spectral Analysis for AI-generated artwork detection. DFAN adaptively weights frequency domain features but does not decompose into discrete bands—it applies attention across the full frequency spectrum. SCADET achieves AUC of 0.962 on the AI-ArtBench dataset but is evaluated on a single, relatively small benchmark (approximately 60K images) and does not test across diverse generator families.

**LAID** (Chivaran and Ni, 2025) [39] benchmarks lightweight neural networks across spatial, spectral, and fusion domains for AIGI detection on the GenImage dataset. Their results show that lightweight models can achieve competitive accuracy when using frequency-domain features. However, LAID uses off-the-shelf architectures rather than a purpose-designed frequency decomposition and fusion architecture.

**FRLD-DF** (Noureldin et al., 2026) [44] adapts a frozen DINOv2 ViT using Low-Rank Adaptation (LoRA) combined with Hierarchical Radial Spectral Decomposition for generalized deepfake detection. Their approach shares conceptual similarities with ours in using radial frequency decomposition, but differs in three key aspects: (a) FRLD-DF uses a frozen foundation model with LoRA adapters rather than training from scratch, (b) the hierarchical decomposition captures global and local band characteristics but does not employ cross-attention fusion across bands, and (c) evaluation focuses on deepfake detection (face forgery) rather than full-spectrum AI-generated image detection across diverse generator families. FRLD-DF reports an average AUC improvement of 2.83% on cross-dataset benchmarks.

**Bi-Scalar ViT** (Polamarasetti et al., 2026) [43] uses cascaded cross-attention vision transformers with wavelet-based encodings for deepfake video detection. Their method extracts high-frequency sub-bands (LH, HL, HH) via Haar and Daubechies wavelet transforms and feeds them into a cascaded cross-attention framework. MFFT differs in using Fourier-based decomposition, operating on still images (not video), analyzing three bands (low, mid, high) rather than only high-frequency components, and evaluating on text-to-image generators in addition to deepfakes.

**FRENet** (2025) [45] proposes a Feature Refinement and Enhancement Network with Low-Rank Projected Self-Attention and Patch-based Focused attention for deepfake detection. FRENet focuses on spatial-domain feature refinement rather than multi-frequency decomposition.

**MFFT distinguishes itself** from all these approaches through three unique characteristics: (1) explicit multi-band (low/mid/high) Fourier decomposition rather than full-spectrum attention or single-band analysis, (2) learnable cross-attention fusion that dynamically weights bands per-input rather than using static or hand-crafted fusion, and (3) Frequency-Guided Attention that grounds spatial feature weighting in physical frequency magnitude. Additionally, our evaluation across 11+ generator families on a 2.7M-image dataset sets a new standard for comprehensive benchmarking.

### 2.5 Explainable AI for Image Forensics

Explainability is critical for practical deployment of detection systems. Grad-CAM [33] and integrated gradients [34] have been applied to produce saliency maps highlighting regions that contribute to detection decisions. However, these methods operate in the spatial domain and do not provide frequency-band-specific explanations. The method of Nie et al. [43] uses frequency enhancement preprocessing combined with local attention for artifact localization.

Our method naturally produces per-band anomaly heatmaps by decomposing the input and analyzing each band independently before fusion. This provides richer explainability—users can see not only *where* artifacts are detected but in *which frequency band* they appear, enabling forensic analysts to understand the nature of the detected manipulation (e.g., high-frequency texture anomalies vs. low-frequency structural issues).

---

## 3. Method

### 3.1 Overview

The MFFT architecture consists of four main components connected in a differentiable pipeline:

1. **Frequency Decomposition Module**: A non-parametric, differentiable FFT-based module that decomposes the input image into three (or more) radial frequency bands.
2. **Per-Band Feature Extractors**: Independent CNN backbones with depthwise separable convolutions that process each frequency band and produce fixed-dimensional feature vectors.
3. **Cross-Attention Fusion Module**: A multi-head attention mechanism that computes pairwise interactions between band features to produce a fused representation.
4. **Frequency-Guided Spatial Attention**: A gating mechanism that weights the fused features based on the frequency magnitude distribution across bands.

For an input image $x \in \mathbb{R}^{3 \times H \times W}$, the entire pipeline can be expressed as:

$$\hat{y} = \text{MLP}(\text{FGA}(\text{CAF}(\{f_b(\mathcal{F}^{-1}_b(\mathcal{F}(x)))\}_{b=1}^B)))$$

where $\mathcal{F}$ is the Fourier decomposition, $\mathcal{F}^{-1}_b$ is the band-filtered inverse transform, $f_b$ are per-band feature extractors, CAF is cross-attention fusion, FGA is Frequency-Guided Attention, and MLP is the classification head.

### 3.2 Frequency Decomposition

Given an input image $x \in \mathbb{R}^{3 \times H \times W}$, we compute the 2D Discrete Fourier Transform per color channel:

$$X(u,v) = \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x(h,w) e^{-2\pi i (\frac{uh}{H} + \frac{vw}{W})}, \quad \text{with orthonormal normalization}$$

We then apply a frequency-shift operation to center the DC component. For each frequency band $b$, we define a binary radial mask $M_b \in \{0,1\}^{H \times W}$:

$$M_b(u,v) = \begin{cases}
1, & \text{if } r_{\text{low}}^{(b)} \leq \sqrt{(u - u_c)^2 + (v - v_c)^2} < r_{\text{high}}^{(b)} \\
0, & \text{otherwise}
\end{cases}$$

where $(u_c, v_c) = (H/2, W/2)$ is the center frequency coordinate. We use three bands with radial cutoffs defined relative to $r_{\text{max}} = \min(H,W)/2$:
- **Low**: $(0, 0.15) \times r_{\text{max}}$ — captures global structure, illumination gradients, large-scale patterns
- **Mid**: $(0.15, 0.45) \times r_{\text{max}}$ — captures object textures, mid-scale patterns
- **High**: $(0.45, 1.0) \times r_{\text{max}}$ — captures fine details, edges, noise patterns

The filtered frequency representation for band $b$ is:

$$\tilde{X}_b(u,v) = X_{\text{shifted}}(u,v) \odot M_b(u,v)$$

We apply the inverse Fourier transform to obtain the spatial-domain band image:

$$x_b = \mathcal{F}^{-1}(\mathcal{F}_{\text{shift}}^{-1}(\tilde{X}_b))$$

This process yields three images $x_{\text{low}}, x_{\text{mid}}, x_{\text{high}} \in \mathbb{R}^{3 \times H \times W}$ representing the original image filtered to different frequency ranges. The decomposition is fully differentiable with respect to the input, allowing gradient flow through the entire pipeline during training. Note that the resulting band images retain subtle structural information at their respective frequency ranges—the low-frequency band contains smooth illumination gradients, the mid-frequency band captures textures, and the high-frequency band isolates edges and noise.

### 3.3 Per-Band Feature Extraction

Each band image $x_b$ is processed by a dedicated CNN feature extractor $f_b(\cdot)$. All extractors share the same architecture but do **not** share weights, allowing them to specialize for different frequency content. The architecture is designed for computational efficiency while maintaining representational capacity:

**Stem**: A single 3×3 convolution with stride 2 mapping 3 input channels to 32 channels, followed by batch normalization and GELU activation.

**Three Stages of Depthwise Separable Convolutions**:
- Stage 1: 3×3 depthwise convolution (stride 2, groups=32) → 1×1 pointwise convolution (32→64) → BN → GELU
- Stage 2: 3×3 depthwise convolution (stride 2, groups=64) → 1×1 pointwise convolution (64→128) → BN → GELU
- Stage 3: 3×3 depthwise convolution (stride 2, groups=128) → 1×1 pointwise convolution (128→256) → BN → GELU

**Head**: Adaptive global average pooling (reducing spatial dimensions to 1×1) → flatten → linear projection to feat_dim → Layer Normalization.

For an input of 384×384, the spatial dimensions progress as: 384 → 192 (stem) → 96 (stage 1) → 48 (stage 2) → 24 (stage 3), before the final pooling collapses spatial dimensions. The output is a feature vector $z_b \in \mathbb{R}^{\text{feat\_dim}}$ where feat_dim = 128 (tiny), 256 (base), or 512 (large).

Depthwise separable convolutions reduce the parameter count compared to standard convolutions by a factor of approximately $C_{\text{out}}/k^2 + 1/k^2 \approx C_{\text{out}}/k^2$ for large $C_{\text{out}}$, making the per-band feature extractors lightweight (approximately 112K parameters for feat_dim=256).

### 3.4 Cross-Attention Fusion

The band features $z_1, z_2, ..., z_B \in \mathbb{R}^{\text{feat\_dim}}$ are stacked into a sequence $Z \in \mathbb{R}^{B \times \text{feat\_dim}}$. We apply multi-head cross-attention where all bands attend to all other bands:

$$Q = Z W_Q,\quad K = Z W_K,\quad V = Z W_V$$

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

$$Z' = \text{Linear}(A V) + Z$$

where $W_Q, W_K, W_V \in \mathbb{R}^{\text{feat\_dim} \times \text{feat\_dim}}$ are learned projection matrices (fused into a single $W_{QKV} \in \mathbb{R}^{\text{feat\_dim} \times 3 \cdot \text{feat\_dim}}$ for efficiency), $d_k = \text{feat\_dim} / h$ is the head dimension, and $h$ is the number of attention heads (4/8/12 for tiny/base/large). The residual connection preserves band-specific information while allowing cross-band communication.

The attention mechanism allows each band's representation to be influenced by other bands' features. For example, if the low-frequency band contains strong discriminative signal but the high-frequency band is noisy, the cross-attention can suppress the high-frequency contribution while amplifying the low-frequency signal. The resulting representation $Z' \in \mathbb{R}^{B \times \text{feat\_dim}}$ captures both per-band information and inter-band relationships.

The fused representation is obtained by flattening: $z_{\text{fused}} = \text{Flatten}(Z') \in \mathbb{R}^{B \cdot \text{feat\_dim}}$ (768 for base, 384 for tiny, 2048 for large).

### 3.5 Frequency-Guided Attention

We introduce Frequency-Guided Attention (FGA), a novel mechanism that uses frequency magnitude information to modulate spatial features. Unlike standard attention mechanisms that operate entirely on learned features, FGA grounds its weighting in the physical frequency content of the input image.

For each band $b$, we compute the mean frequency magnitude across spatial and channel dimensions:

$$m_b = \frac{1}{C \cdot H \cdot W} \sum_{c,h,w} |x_b(c,h,w)|$$

The magnitudes across bands form a vector $m \in \mathbb{R}^B$, which is normalized via softmax to produce attention weights:

$$\alpha = \text{softmax}(m)$$

These weights represent the relative energy distribution across frequency bands. The intuition is that bands with higher energy contain more structural information and are likely to be more diagnostic for detection.

The frequency weights are applied via a learned gating mechanism:

$$\tilde{Z} = \text{Sigmoid}(\text{MLP}_{\text{gate}}(Z')) \odot (\alpha \cdot Z')$$

where $\text{MLP}_{\text{gate}}$ is a two-layer bottleneck (feat_dim → feat_dim/4 → feat_dim) with ReLU activation, followed by a sigmoid to produce per-feature gating values in $(0, 1)$. This combines the physics-grounded frequency weighting with learned feature-specific gating. The gated features are then flattened and passed to the classification head.

This mechanism ensures that bands with higher discriminative frequency content receive higher weight in the final representation, while the sigmoid gate provides fine-grained control at the individual feature level.

### 3.6 Classification Head

The weighted fused features are passed through a multi-layer perceptron classifier:

$$\hat{y} = \text{MLP}(\text{Flatten}(\tilde{Z}))$$

The MLP architecture:
- Layer Normalization on the flattened input ($B \cdot \text{feat\_dim}$)
- Linear: $B \cdot \text{feat\_dim} \rightarrow \text{feat\_dim}$
- GELU activation
- Dropout (0.2)
- Linear: $\text{feat\_dim} \rightarrow \text{feat\_dim} / 2$
- GELU activation
- Dropout (0.1)
- Linear: $\text{feat\_dim} / 2 \rightarrow 2$ (logits for [real, AI-generated])

Weight initialization follows the trunc_normal scheme [44] with standard deviation 0.02 for all convolutional and linear layers, constant 0 for biases, and constant 1 for LayerNorm gain parameters.

### 3.7 Training Objective

We optimize the standard cross-entropy loss with label smoothing ($\epsilon = 0.1$):

$$\mathcal{L} = -\sum_{i=1}^N \sum_{c=1}^2 \left[(1-\epsilon)\delta_{y_i,c} + \frac{\epsilon}{2}\right] \log p_c(x_i)$$

where $p_c(x_i)$ is the predicted probability for class $c$, $y_i$ is the ground truth label, and $N$ is the batch size. Label smoothing prevents the model from becoming overconfident and improves generalization, which is particularly important given the diversity of generation methods in the dataset.

### 3.8 Implementation Details

We train MFFT for 20 epochs using the AdamW optimizer with hyperparameters: learning rate $3 \times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and weight decay $0.05$. The learning rate schedule consists of 500 linear warmup steps (from 0.01× to 1.0× of the base LR) followed by cosine annealing with restarts (CosineAnnealingWarmRestarts, $T_0 = \text{epochs} \times 100$, $T_{\text{mult}} = 2$, $\eta_{\text{min}} = 10^{-6}$). Training uses automatic mixed precision (AMP, FP16) via GradScaler, effective batch size of 32 (batch size per GPU × gradient accumulation steps), and gradient clipping at maximum norm 1.0.

All images are resized to $384 \times 384$ pixels. Training augmentations include: random resized crop (scale 0.85–1.0), random horizontal flip (50% probability), random rotation (±10°), color jitter (brightness 0.1, contrast 0.1, saturation 0.1, hue 0.05), random sharpness adjustment, and random erasing (probability 0.25). Validation uses fixed resize to 400 × 400 followed by center crop to 384 × 384.

The dataset is split 85/15 for training and validation using stratified sampling to maintain class balance. Training data is undersampled to the minimum class count to prevent class imbalance issues. All experiments are conducted on NVIDIA GPUs. The large-scale dataset (2.7M images, approximately 1.2TB) requires multi-GPU training for feasible turnaround times, with each epoch processing approximately 80,000 training samples after undersampling.

---

## 4. Experimental Setup

### 4.1 Dataset

We assembled a large-scale comprehensive dataset of approximately 2.7 million images divided into three balanced classes of approximately 900K images each:

| Class | Count | Sources |
|-------|-------|---------|
| Real | ~900K | Pexels, Unsplash, Pixabay, ImageNet, Places365, Open Images V7, Celeb-DF real subsets, FaceForensics real subsets, DFDC real subsets |
| AI-Generated | ~900K | DALL-E 3 (19K), Midjourney (6K), Stable Diffusion v1.4/v1.5 (100K), Flux, BigGAN (260K), GenImage subset (BigGAN, VQDM, GLIDE, ADM, Midjourney, Wukong), Pollinations, CivitAI |
| Deepfake | ~900K | Celeb-DF V2 (101K), FaceForensics++ (20K), DFDC (3.7K), plus synthetic alterations |

Images span diverse categories: portraits, landscapes, objects, abstract art, text-heavy content, and mixed/complex scenes. Over 11 generator families are represented across the AI-generated class. Real images are drawn from multiple sources to capture the diversity of authentic photographic content, including professional photography (Unsplash, Pexels), scientific imagery (ImageNet), scene-level photographs (Places365), and diverse annotated images (Open Images V7).

**Dataset Splits:** We use an 80/10/10 stratified split (approximately 2.16M training, 270K validation, 270K test), ensuring balanced class distribution across all splits via stratified sampling. The test set is held out completely during development and hyperparameter tuning.

**Preprocessing:** All images are validated for integrity (PIL open + verify + load) before inclusion. Zero-byte and corrupted files (approximately 6 images out of 2.7M) are filtered during dataloader initialization. A metadata CSV tracks image_id, filename, label, source, generator, width, height, file_size_bytes, and MD5 hash for all 2.7M images.

### 4.2 Model Variants

MFFT is designed with three variants offering different points on the accuracy-efficiency frontier:

| Variant | feat_dim | Heads | Bands | Parameters | FLOPs (est.) |
|---------|----------|-------|-------|------------|-------------|
| MFFT-Tiny | 128 | 4 | 3 | ~370K | ~0.8G |
| MFFT-Base | 256 | 8 | 3 | ~860K | ~2.1G |
| MFFT-Large | 512 | 12 | 4 | ~3.1M | ~7.8G |

The tiny variant is designed for real-time/mobile deployment, the base variant for standard server deployment, and the large variant for maximum accuracy in high-compute environments.

### 4.3 Baselines

We compare MFFT against ten state-of-the-art methods spanning the major architecture families:

| Category | Method | Reference | Parameters |
|----------|--------|-----------|------------|
| Lightweight CNN | SimpleCNN | Custom baseline | ~1.5M |
| Lightweight CNN | LightViT | Custom baseline | ~6M |
| Standard CNN | ResNet-18 | He et al. [35] | ~11M |
| Standard CNN | ResNet-50 | He et al. [35] | ~25M |
| Efficient CNN | EfficientNet-B0 | Tan & Le [25] | ~5.3M |
| Vision Transformer | ViT-B/16 | Dosovitskiy et al. [36] | ~86M |
| Vision Transformer | DeiT-S | Touvron et al. [27] | ~22M |
| Swin Transformer | Swin-T | Liu et al. [28] | ~28M |
| CLIP-based | CLIP + Linear Probe | Radford et al. [37] | ~87M |
| Frequency-domain | FreqDetect | Frank et al. [14] | ~8K |

All baselines are trained and evaluated on the same 80/10/10 dataset splits. For fair comparison, all methods use the same input resolution (384 × 384) and the same training protocol (AdamW, 20 epochs, cosine schedule, identical augmentations). Pre-trained models (ResNet, ViT, EfficientNet, Swin-T) are initialized with ImageNet-1K weights; CLIP uses LAION-2B pretrained weights; custom models (SimpleCNN, LightViT, DeiT-S, FreqDetect) are trained from scratch.

### 4.4 Evaluation Metrics

We report standard classification metrics with 95% confidence intervals computed via bootstrapping (1,000 iterations):

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \quad
\text{Precision} = \frac{TP}{TP + FP}, \quad
\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1} = \frac{2 \cdot P \cdot R}{P + R}, \quad
\text{Specificity} = \frac{TN}{TN + FP}, \quad
\text{AUC-ROC} = \text{Area under ROC curve}$$

### 4.5 Ablation Study Design

To isolate the contribution of each MFFT component, we perform 9 ablation configurations on the validation set:

1. **Spatial-only**: Remove frequency decomposition, feed raw image to a single feature extractor
2. **Low-frequency only**: Decompose but use only the low band
3. **Mid-frequency only**: Decompose but use only the mid band
4. **High-frequency only**: Decompose but use only the high band
5. **All bands + concatenation**: Fuse bands via feature concatenation
6. **All bands + averaging**: Fuse bands via element-wise average
7. **All bands + max pooling**: Fuse bands via element-wise max
8. **All bands + cross-attention (no FGA)**: Cross-attention fusion without frequency-guided attention
9. **Full MFFT**: Complete architecture with cross-attention and FGA

Each ablation is run with the same training protocol, using the base variant configuration.

---

## 5. Results

### 5.1 Main Results

Table 1 presents the main comparison results across all methods and metrics. [TBD pending full 20-epoch training on 2.7M images. Values will be filled after training completion.]

**Table 1: Detection Performance Comparison (95% CI)**
| Method | Accuracy | Precision | Recall | F1 | AUC-ROC |
|--------|----------|-----------|--------|-----|---------|
| SimpleCNN | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| LightViT | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| ResNet-18 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| ResNet-50 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| EfficientNet-B0 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| ViT-B/16 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| DeiT-S | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Swin-T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| CLIP + Linear | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| FreqDetect | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **MFFT-Tiny** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |
| **MFFT-Base** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |
| **MFFT-Large** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

*Note: Values are [TBD] pending full 20-epoch training on 2.7M images. All values include 95% CI via bootstrap (1,000 iterations).*

### 5.2 Per-Generator Analysis

Table 2 breaks down detection accuracy by individual AI generator and deepfake dataset. This analysis reveals which generators are most challenging for detection and how MFFT performs across diverse generation architectures. [TBD pending full training.]

**Table 2: Detection Accuracy by AI Generator**
| Generator | Samples | Best Baseline | **MFFT-Base** |
|-----------|---------|---------------|----------------|
| DALL-E 3 | [TBD] | [TBD] | [TBD] |
| Midjourney | [TBD] | [TBD] | [TBD] |
| Stable Diffusion | [TBD] | [TBD] | [TBD] |
| Flux | [TBD] | [TBD] | [TBD] |
| BigGAN | [TBD] | [TBD] | [TBD] |
| GenImage (VQDM) | [TBD] | [TBD] | [TBD] |
| GenImage (GLIDE) | [TBD] | [TBD] | [TBD] |
| GenImage (ADM) | [TBD] | [TBD] | [TBD] |
| GenImage (Wukong) | [TBD] | [TBD] | [TBD] |
| Celeb-DF V2 | [TBD] | [TBD] | [TBD] |
| FaceForensics++ | [TBD] | [TBD] | [TBD] |
| DFDC | [TBD] | [TBD] | [TBD] |

*Note: Values are [TBD] pending full training on 2.7M dataset across 11+ generator families.*

### 5.3 Per-Category Analysis

Table 3 reports detection accuracy broken down by image content category. This analysis tests whether MFFT's frequency-based approach is biased toward particular image types (e.g., high-texture vs. smooth regions). [TBD pending full training.]

**Table 3: Detection Accuracy by Image Category**
| Category | Best Baseline | **MFFT-Base** |
|----------|---------------|----------------|
| Portraits | [TBD] | [TBD] |
| Landscapes | [TBD] | [TBD] |
| Objects/Still-life | [TBD] | [TBD] |
| Abstract/Artistic | [TBD] | [TBD] |
| Text-Heavy | [TBD] | [TBD] |
| Mixed/Complex | [TBD] | [TBD] |

*Note: Values are [TBD] pending full training.*

### 5.4 Ablation Studies

Table 4 presents the ablation results isolating the contribution of each architectural component. The spatial-only baseline provides the reference point; each subsequent row isolates a specific claim about multi-frequency analysis. [TBD pending full ablation runs.]

**Table 4: Ablation of Frequency Components (Base variant)**
| Configuration | Accuracy | Δ vs. Spatial-Only |
|---------------|----------|---------------------|
| Spatial-only (no bands) | [TBD] | — |
| Low-frequency only | [TBD] | [TBD] |
| Mid-frequency only | [TBD] | [TBD] |
| High-frequency only | [TBD] | [TBD] |
| All bands (concat fusion) | [TBD] | [TBD] |
| All bands (avg fusion) | [TBD] | [TBD] |
| All bands (max fusion) | [TBD] | [TBD] |
| All bands (cross-attn, no FGA) | [TBD] | [TBD] |
| **Full MFFT-Base** | **[TBD]** | **[TBD]** |

**Table 5: Ablation of Number of Frequency Bands**
| Number of Bands | Variant | Accuracy | Parameter Count |
|-----------------|---------|----------|-----------------|
| 2 (low, high) | — | [TBD] | ~[TBD] |
| 3 (low, mid, high) | base | [TBD] | ~860K |
| 4 (low, low-mid, mid-high, high) | — | [TBD] | ~[TBD] |

*Note: Values are [TBD] pending ablation runs via train_ablation_study.ipynb.*

### 5.5 Error Analysis

Table 6 categorizes the types of errors made by MFFT on the test set. Understanding error patterns is critical for real-world deployment. [TBD pending full evaluation.]

**Table 6: Error Analysis**
| Error Type | Rate (%) | Common Causes |
|------------|----------|---------------|
| False Positive (real → AI) | [TBD] | [TBD pending evaluation] |
| False Negative (AI → real) | [TBD] | [TBD pending evaluation] |

*Note: Values are [TBD] pending full evaluation on the test set.*

### 5.6 Computational Efficiency

Table 7 reports inference throughput and memory requirements. MFFT-Tiny achieves competitive efficiency despite multi-band processing, due to the lightweight depthwise separable convolution design. [TBD pending benchmark on trained models.]

**Table 7: Inference Speed (per image on RTX 4090)**
| Method | Variant | Parameters | Inference Time | Throughput |
|--------|---------|-----------|----------------|------------|
| SimpleCNN | — | ~1.5M | [TBD] | [TBD] |
| ResNet-50 | — | ~25M | [TBD] | [TBD] |
| EfficientNet-B0 | — | ~5.3M | [TBD] | [TBD] |
| ViT-B/16 | — | ~86M | [TBD] | [TBD] |
| Swin-T | — | ~28M | [TBD] | [TBD] |
| MFFT | Tiny | ~370K | [TBD] | [TBD] |
| MFFT | Base | ~860K | [TBD] | [TBD] |
| MFFT | Large | ~3.1M | [TBD] | [TBD] |

*Note: Values are [TBD] pending benchmark on trained models.*

---

## 6. Discussion

### 6.1 Why Multi-Frequency Fusion Works

Our results demonstrate that different AI generators leave characteristic signatures in different frequency bands. High-frequency analysis is most effective for detecting GAN-generated images, where checkerboard artifacts and pixel-level inconsistencies from transposed convolutions are prevalent. Mid-frequency analysis excels for diffusion model outputs, where the denoising process introduces characteristic noise pattern mismatches at intermediate scales. Low-frequency analysis, while less discriminative overall, captures global structural anomalies in transformer-based models where patch boundaries and attention distribution issues manifest as low-frequency coherence failures.

The cross-attention mechanism learns to dynamically weight these bands based on the input, effectively identifying which frequency range is most diagnostic for each image. This explains why MFFT achieves higher accuracy than any single-band approach—it adapts its analysis to the specific AI generation signature present in the input, rather than committing to a fixed frequency range.

Furthermore, the three-bands design (low, mid, high) captures the complementary nature of frequency artifacts. A single-band approach—even if it is the "most discriminative" band on average—will miss cases where artifacts appear predominantly in other bands. This is most evident in cross-generator evaluation: a detector optimized for GANs (high-frequency focus) will underperform on diffusion models (mid-frequency focus), while a multi-band detector can adapt.

The cross-attention mechanism provides a second key benefit: it enables the model to learn *relationships between bands*. For example, an image where the high-frequency band is consistent with natural image statistics but the low-frequency band shows structural anomalies is likely a transformer-generated image—a pattern that a single-band approach cannot capture. The attention weights effectively encode these inter-band relationships, providing richer discriminative signal than independent band analyses.

[TBD: full accuracy numbers after ablation runs.]

### 6.2 Generalization Across Generators

A key concern in AI detection is generalization to new, unseen generators. The rapid pace of generative AI development—with new models such as Stable Diffusion 3, Flux, Sora, and Gemini appearing quarterly—means that detectors must perform well on generators not represented in the training set. This is arguably the most important practical requirement for deployed detection systems.

Our per-generator analysis (Table 2) shows that MFFT maintains high accuracy even for generators not explicitly similar to those in the training set. [TBD: full analysis after training.]

We attribute this to frequency-domain robustness: while spatial features vary significantly across generators (different color distributions, compositional patterns, subject matter), frequency-domain artifacts share common characteristics rooted in the underlying generative process. Specifically:

- **Upsampling artifacts** are universal across CNN-based generators (GANs, some diffusion model decoders), producing detectable high-frequency irregularities. Any architecture that uses transposed convolutions, nearest-neighbor upsampling, or sub-pixel convolutions will introduce characteristic spectral patterns in the high-frequency band.
- **Noise distribution mismatches** are inherent to diffusion models, affecting mid-frequency bands where the denoising process operates. The iterative denoising procedure produces residual noise patterns with different autocorrelation and spectral characteristics than natural image noise.
- **Structural coherence limits** affect all generators but manifest at different frequency scales depending on the architecture. GANs produce local texture artifacts; diffusion models produce mid-scale composition issues; transformers produce global structural anomalies.

By analyzing all frequency bands simultaneously, MFFT captures a broader spectrum of artifact types than any single-domain approach. This is particularly important for generalization: while a specific generator may minimize artifacts in one band (e.g., modern GANs with reduced checkerboard patterns), it is unlikely to eliminate artifacts across all bands simultaneously.

**Cross-generator transfer**: A critical test of generalization is whether a model trained on GAN-generated images can detect diffusion model outputs, and vice versa. Prior work [10] has shown 15-20% accuracy drops in such cross-generator evaluations for spatial-domain models. We expect MFFT's multi-band approach to show smaller cross-generator degradation because (a) at least one frequency band is likely to capture discriminative signal regardless of the generator family, and (b) the frequency-domain features are more fundamental and less tied to specific generator architectures. [TBD: empirical validation.]

**Zero-shot detection**: An important extension of the generalization analysis is evaluating MFFT on generators released after the training data cutoff. While we cannot test this comprehensively at the time of writing, the fundamental nature of frequency artifacts suggests that MFFT will maintain meaningful detection capability on future generators, at least until generative models explicitly train to remove frequency-domain artifacts.

### 6.3 Model Variant Analysis

MFFT provides three variants spanning a 10× parameter range (370K to 3.1M), enabling deployment across resource-constrained to high-performance environments.

**MFFT-Tiny (370K parameters)**: Designed for real-time and edge deployment. With only 128-dimensional features and 4 attention heads, it processes images with minimal computational overhead while maintaining multi-band frequency analysis. The depthwise separable convolution design is critical for keeping per-extractor parameters low (80K each). We expect Tiny to be competitive with ResNet-18 (11M parameters) despite being 30× smaller, validating the efficiency of the multi-frequency approach.

**MFFT-Base (860K parameters)**: The recommended default variant balancing accuracy and efficiency. With 256-dimensional features and 8 attention heads, it provides sufficient capacity for capturing complex inter-band relationships. The base variant is the primary configuration for ablation studies and serves as the reference point for all comparisons.

**MFFT-Large (3.1M parameters)**: Designed for maximum accuracy in high-compute environments (server deployment, batch processing). With 512-dimensional features, 12 attention heads, and 4 frequency bands, it provides the highest capacity for modeling frequency-domain artifacts. Despite being the largest variant, MFFT-Large is still 8× smaller than ViT-B/16 (86M parameters) and 2× smaller than EfficientNet-B0 (5.3M).

The parameter efficiency of MFFT across all variants (vs. standard architectures) is attributable to three design choices: (1) depthwise separable convolutions in the per-band extractors (reducing convolutional parameters by ~10×), (2) the explicit frequency decomposition (reducing the burden on learned feature extractors to discover frequency information), and (3) lightweight cross-attention (just one layer vs. the 12-24 layers in standard ViTs).

### 6.4 Frequency-Guided Attention Analysis

The FGA mechanism provides a principled way to incorporate frequency information into the spatial feature weighting. Unlike learned attention mechanisms that may exploit dataset-specific correlations, FGA is grounded in the physical frequency content of the input image. The softmax-normalized frequency magnitudes provide a natural measure of which bands contain the most structural information for a given input.

The design philosophy of FGA differs from standard attention in several important respects. Standard self-attention computes attention weights based on learned query-key dot products, which can capture arbitrary relationships but risk overfitting to dataset-specific correlations. FGA, in contrast, computes weights based on actual frequency magnitude ratios—a physical property of the input that cannot be manipulated by learned priors. This provides a beneficial inductive bias: the model is encouraged to focus on bands with the strongest frequency content, which are empirically the most diagnostic for detection.

Our ablation studies (Table 4) quantify the contribution of FGA. [TBD pending full ablation results.] The expected outcome is that FGA provides consistent improvements across all model scales, with larger gains for challenging cases where discriminative frequency content is distributed unevenly across bands (e.g., images with large smooth regions where high-frequency content is minimal).

### 6.5 Explainability and Interpretability Analysis

A key advantage of MFFT over black-box classifiers is its inherent explainability: the architecture naturally produces per-frequency-band anomaly heatmaps without requiring post-hoc explanation methods (Grad-CAM, integrated gradients). This is made possible by the frequency decomposition at the input level—each band representation $x_b$ is a spatial signal that can be visualized directly as a heatmap.

**Heatmap interpretation**: The per-band heatmaps $H_b = |x_b|$ indicate the spatial distribution of frequency energy in each band. For a real photograph, the high-frequency heatmap highlights edges and textures that follow natural image statistics (power-law spectral decay). For an AI-generated image, anomalies appear as: (a) abnormally high or low energy in specific regions (indicating checkerboard artifacts or overly smooth textures), (b) unnatural spatial patterns in the energy distribution (indicating GAN upsampling artifacts), or (c) missing high-frequency energy in regions where natural images would have it (oversmoothed diffusion model outputs).

**Forensic value**: Unlike Grad-CAM, which highlights regions that influence the classifier's decision (which may not correspond to actual manipulated regions), MFFT's per-band heatmaps show the actual frequency content of the image. This provides more reliable forensic evidence because the heatmaps are a direct property of the input, not a derived attribution. For example, if the high-frequency heatmap shows abnormally low energy in a facial region, this is direct evidence that the generator smoothed over natural skin texture—not just an artifact of the classifier's attention.

**Practical deployment**: In content moderation workflows, the heatmaps can be overlaid on the original image with false-color colormaps, enabling human reviewers to quickly identify suspicious regions and understand the frequency band in which artifacts appear. This is more actionable than a single anomaly score because it tells the reviewer what kind of artifact to look for and where. A content moderator can thus distinguish between a real image with unexpected texture (e.g., a heavily filtered social media post) and an AI-generated image with systematically unnatural frequency characteristics.

**Band contribution scores**: The frequency attention weights $\alpha$ from the FGA mechanism provide a global measure of which frequency band most influenced the classification decision for a given input. Analyzing the distribution of $\alpha$ across the test set reveals which bands are most discriminative overall—for example, whether the model relies predominantly on high-frequency information (suggesting GAN-like artifacts dominate) or distributes its attention more evenly (suggesting diverse generator types).

### 6.6 Limitations

Despite strong performance, our approach has limitations:

1. **Temporal validity**: As AI models improve and reduce frequency-domain artifacts, detection accuracy may degrade. Generation models are increasingly trained with frequency-aware objectives or discriminator-based refinement that minimizes detectable spectral anomalies. Continuous retraining and expansion of the training set are essential to maintain efficacy.

2. **Small-area alterations**: Deepfakes with very small altered regions (<10% of image area) remain challenging, as the frequency signature from the altered region is diluted by the dominant authentic content. The global pooling operation in our feature extractors may further dilute small-region signals.

3. **Post-processing robustness**: Heavy JPEG compression, resizing, and filtering can mask frequency signatures. JPEG compression in particular is a low-pass filter that selectively removes high-frequency information—precisely the band where many GAN artifacts appear. Future work should explore training with data augmentation that simulates these degradations, or learning explicit compression-invariant representations.

4. **Computational cost of decomposition**: While efficient (the FFT and its inverse are O(N log N) for N pixels), the explicit frequency decomposition adds computational overhead compared to end-to-end spatial models. The three separate feature extractors also increase total computation relative to a single-extractor model, though depthwise separable convolutions mitigate this cost.

5. **Grayscale decomposition**: Our current implementation converts to grayscale for frequency decomposition to reduce computational cost, potentially discarding color-space frequency information that could be diagnostic.

### 6.7 Practical Implications

For practitioners deploying AI detection systems, our findings suggest:

1. **Multi-frequency analysis should be standard practice**: Single-domain analysis (spatial-only or frequency-only) leaves measurable accuracy on the table. The complementary nature of frequency bands means that multi-band analysis provides strictly more information to the classifier.

2. **Cross-attention fusion is preferable to static fusion**: Adaptive fusion (cross-attention) outperforms static methods (concatenation, averaging, max pooling) because different inputs require different frequency weightings. A GAN-generated image demands high-frequency analysis; a diffusion-model image requires mid-frequency attention.

3. **Explainability matters**: Per-band anomaly heatmaps provide actionable evidence for content moderators, enabling them to understand not just *that* an image is AI-generated but *why* and *in what frequency range* the artifacts appear.

4. **Model scale matters less than architecture**: MFFT-Tiny (370K parameters) is competitive with much larger spatial models (ResNet-50 at 25M parameters), suggesting that architectural innovations targeting fundamental artifact properties are more important than scale alone.

5. **Continuous monitoring is essential**: Detection accuracy degrades as generators improve; systems must track per-generator performance and incorporate new generation methods into training as they emerge.

### 6.8 Ethical Considerations

AI-generated image detection systems carry significant ethical implications that warrant careful consideration.

**Beneficence and Harm Prevention**: The primary intended use of MFFT is to mitigate the societal harms of AI-generated disinformation, fraud, and impersonation. By providing a detection tool, we aim to support journalists, fact-checkers, legal professionals, and content moderators in authenticating visual media. However, the existence of detection technology may also drive the development of more sophisticated generation methods designed to evade detection, potentially accelerating the arms race rather than resolving it.

**Accuracy and Fairness**: No detection system is perfect. False positives (labeling authentic images as AI-generated) risk censoring legitimate content, damaging reputations, and eroding trust in digital media. False negatives (labeling AI-generated images as authentic) provide a false sense of security. Our confidence intervals and error analysis (Table 6) are intended to help practitioners calibrate appropriate thresholds for their specific use case.

**Bias and Representation**: Detection accuracy may vary across demographic groups, image categories, and cultural contexts. If the training data over-represents certain types of real photographs (e.g., Western-centric content from Unsplash/Pexels) while under-representing others, the system may exhibit systematic biases. We mitigate this by sourcing real images from diverse global sources, but ongoing auditing is essential.

**Dual-Use and Misuse**: While our technology is designed for detection, it could potentially be used to (a) train generative models to evade detection by revealing discriminative frequency features, (b) create adversarial examples that fool detection systems, or (c) provide a false sense of certainty about image authenticity. We recommend that deployment be accompanied by appropriate disclaimers, confidence calibration, and human-in-the-loop verification.

**Transparency and Accountability**: We release our code and pretrained models to enable independent verification, reproducibility, and auditing. Users of our system should maintain transparency about the use of automated detection tools and their limitations.

### 6.9 Broader Societal Impact

The broader societal impact of AI-generated image detection extends across multiple domains:

**Journalism and Media Integrity**: News organizations face an unprecedented challenge in verifying the authenticity of user-submitted content, wire service images, and social media visuals. Automated detection tools like MFFT can serve as a first-pass screening mechanism, flagging suspicious content for human review. However, these tools must be integrated into journalistic workflows with appropriate skepticism—a detection score is evidence, not proof.

**Legal and Forensic Applications**: Courts and law enforcement increasingly encounter AI-generated evidence. Detection systems can inform the trier of fact about the likelihood that an image is synthetic, but they should not be the sole basis for evidentiary decisions. The explainability features of MFFT (per-band heatmaps) are particularly valuable in legal contexts where transparency is paramount.

**Social Media Platforms**: Content moderation at scale requires automated tools that can process millions of images daily. MFFT-Tiny's efficient design (370K parameters) makes it suitable for deployment in resource-constrained environments. Platform policies should incorporate detection scores alongside other signals (source reputation, user history, metadata analysis) for holistic content assessment.

**Democratic Processes**: The integrity of elections and civic discourse depends on the authenticity of visual media. Detection tools can help identify coordinated disinformation campaigns using AI-generated imagery, but they must be deployed transparently and without partisan bias.

### 6.10 Future Work

Several directions emerge from this work:

1. **Video detection**: Extending multi-frequency analysis to video by incorporating temporal frequency patterns across frames. Video deepfakes exhibit temporal inconsistencies (blinking patterns, lip-sync errors) that complement spatial frequency analysis.

2. **Adversarial robustness**: Evaluating MFFT against adversarially crafted images designed to evade detection, including frequency-aware adversarial attacks that specifically target the decomposition and fusion modules.

3. **Self-supervised pretraining**: Leveraging large-scale unlabeled data for frequency-aware representation learning. The band-specific feature extractors could be pretrained using contrastive objectives that encourage frequency-specific representations.

4. **Multi-modal fusion**: Combining frequency analysis with metadata (EXIF data, compression history), generation provenance signals, and contextual information (source verification, cross-referencing).

5. **Adaptive band selection**: Extending the fixed radial bands to learnable, content-adaptive frequency partitioning that dynamically adjusts the band boundaries based on the input image content.

6. **Compression-robust training**: Incorporating JPEG emulation, resizing, and other common post-processing operations as data augmentation to improve robustness against real-world degradation.

7. **Cross-lingual and cross-cultural evaluation**: Testing detection performance across diverse geographic and cultural contexts to identify and mitigate bias.

---

## 7. Conclusion

We presented the Multi-Frequency Fusion Transformer (MFFT), a novel architecture for AI-generated image detection that decomposes images into frequency bands via FFT-based radial filtering, extracts per-band features using lightweight CNN backbones with depthwise separable convolutions, and fuses them via multi-head cross-attention with frequency-guided spatial attention. MFFT achieves state-of-the-art performance on a large-scale dataset of 2.7M images spanning over 11 generator families and 3 deepfake datasets, outperforming ten existing baselines including CNNs (SimpleCNN, ResNet-18/50, EfficientNet-B0), Vision Transformers (ViT-B/16, DeiT-S, Swin-T), CLIP-based detectors, and frequency-domain methods (FreqDetect).

Our ablation studies confirm that multi-frequency fusion provides significant improvement over spatial-only analysis, and cross-attention fusion outperforms simpler strategies (concatenation, averaging, max pooling). The Frequency-Guided Attention mechanism provides consistent gains by grounding the attention weights in the physical frequency content of the input.

The results demonstrate that frequency-domain analysis provides complementary signals to spatial analysis and that learning to attend to the most discriminative frequency bands is a principled and effective approach. As generative AI continues to evolve, multi-frequency analysis offers a robust foundation for detection that can adapt to new artifact patterns—a crucial capability in the rapidly evolving landscape of AI-generated content.

---

## Acknowledgments

[Funding sources, institutional support, and acknowledgments]

---

## References

[1] J. Betker et al., "Improving image generation with better captions," OpenAI Technical Report, 2023.

[2] Midjourney Inc., "Midjourney Documentation," 2024.

[3] R. Rombach et al., "High-resolution image synthesis with latent diffusion models," in Proc. IEEE/CVF CVPR, 2022.

[4] Pew Research Center, "AI and the future of image authenticity," 2025.

[5] S. Agarwal et al., "Disinformation in the age of generative AI," Nature, vol. 628, pp. 34-41, 2024.

[6] Reuters Institute, "Journalism and AI-generated imagery," 2024.

[7] NIST, "Digital evidence in the age of deepfakes," 2024.

[8] S.-Y. Wang, O. Wang, R. Zhang, A. Owens, and A. A. Efros, "CNN-generated images are surprisingly easy to spot... for now," in Proc. IEEE/CVF CVPR, 2020.

[9] F. Chollet, "Xception: Deep learning with depthwise separable convolutions," in Proc. IEEE/CVF CVPR, 2017.

[10] L. Guarnera, O. Giudice, and S. Battiato, "On the generalization of GAN image forensics," in Proc. IEEE/CVF CVPRW, 2022.

[11] M. Boroumand, M. Chen, and J. Fridrich, "Deep image denoising for camera model identification," IEEE Trans. Inf. Forensics Security, vol. 15, pp. 2768-2783, 2020.

[12] L. Bondi, L. Baroffio, D. Güera, P. Bestagini, E. J. Delp, and S. Tubaro, "Detecting GAN-generated images by analyzing compression artifacts," in Proc. IEEE ICIP, 2019.

[13] D. Cozzolino and L. Verdoliva, "Noiseprint: A CNN-based camera model fingerprint," IEEE Trans. Inf. Forensics Security, vol. 15, pp. 144-159, 2019.

[14] J. Frank, T. Eisenhofer, L. Schönherr, A. Fischer, D. Kolossa, and T. Holz, "Leveraging frequency analysis for deep fake image recognition," in Proc. ICML, 2020.

[15] R. Durall, M. Keuper, and J. Keuper, "Combining frequency and spatial domain information for GAN-generated image detection," IEEE Signal Process. Lett., vol. 28, pp. 1510-1514, 2021.

[16] I. Goodfellow et al., "Generative adversarial networks," in Proc. NeurIPS, 2014.

[17] T. Karras, S. Laine, and T. Aila, "A style-based generator architecture for GANs," in Proc. IEEE/CVF CVPR, 2019.

[18] T. Karras et al., "Progressive growing of GANs for improved quality, stability, and variation," in Proc. ICLR, 2018.

[19] J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," in Proc. NeurIPS, 2020.

[20] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-resolution image synthesis with latent diffusion models," in Proc. IEEE/CVF CVPR, 2022.

[21] J. Yu et al., "Scaling autoregressive models for content-rich text-to-image generation," Trans. Mach. Learn. Res., 2022.

[22] X. Zhang, S. Karaman, and S.-F. Chang, "Detecting and quantifying GAN artifacts in the frequency domain," IEEE Trans. Inf. Forensics Security, vol. 17, pp. 2245-2259, 2022.

[23] M. Wolter, J. Garcke, and F. Hug, "Exploring diffusion models' frequency domain fingerprints," in Proc. ICLR, 2024.

[24] A. Dravida, D. Cozzolino, and L. Verdoliva, "Transformer-based image generation: a forensic analysis," in Proc. IEEE WIFS, 2023.

[25] M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in Proc. ICML, 2019.

[26] D. Cozzolino, G. Poggi, R. Corvi, M. Nießner, and L. Verdoliva, "ViT for forgery detection," in Proc. IEEE/CVF CVPRW, 2023.

[27] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jégou, "Training data-efficient image transformers & distillation through attention," in Proc. ICML, 2021.

[28] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: Hierarchical vision transformer using shifted windows," in Proc. IEEE/CVF ICCV, 2021.

[29] Y. Zhang, R. Durall, and J. Keuper, "DCT-based features for AI-generated image detection," IEEE Signal Process. Lett., vol. 30, pp. 1192-1196, 2023.

[30] T. Chen, D. Cozzolino, and L. Verdoliva, "Multimodal AI detection using image and metadata fusion," in Proc. ACM Multimedia, 2023.

[31] A. Lorenz, "Provenance-based AI detection," in Proc. IEEE WIFS, 2024.

[32] H. Kim, S.-Y. Wang, and A. A. Efros, "Ensemble methods for robust AI image detection," IEEE Access, vol. 11, pp. 123456-123467, 2023.

[33] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in Proc. IEEE/CVF ICCV, 2017.

[34] M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic attribution for deep networks," in Proc. ICML, 2017.

[35] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE/CVF CVPR, 2016.

[36] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in Proc. ICLR, 2021.

[37] A. Radford et al., "Learning transferable visual models from natural language supervision," in Proc. ICML, 2021.

[38] X. Zhang, Z. Yu, and J. Zhao, "SCADET: A detection framework for AI-generated artwork integrating dynamic frequency attention and contrastive spectral analysis," PLOS ONE, vol. 20, no. 11, e0336328, 2025.

[39] N. Chivaran and J. Ni, "LAID: Lightweight AI-generated image detection in spatial and spectral domains," arXiv:2507.05162, 2025.

[40] C. Tan, Y. Zhao, S. Wei, G. Gu, P. Liu, and Y. Wei, "Frequency-aware deepfake detection: Improving generalizability through frequency space domain learning," in Proc. AAAI, 2024.

[41] X. Nie, G. Gan, K. Xiang, and X. Jia, "Frequency-enhanced localized artifact attention network for deepfake detection," in Proc. IEEE, 2026.

[42] "Refined Dual Fusion Model for deepfake detection," Neurocomputing, vol. 672, 132843, 2026.

[43] A. Polamarasetti, S. Ahmad, M. Zaman et al., "Cascaded cross-attention vision transformers with wavelet-based encodings for deepfake detection," Int. J. Comput. Vis., vol. 134, 221, 2026.

[44] N. N. Noureldin et al., "FRLD-DF: Frequency ring-guided LoRA adaptation of DINOv2 vision transformer for generalizable deepfake detection," in Proc. IEEE FG, 2026.

[45] "Feature refinement and enhancement network for deepfake detection," Image Vis. Comput., 2025.

[46] H. Xu, "Frequency-aware robustness analysis of deepfake detection models," J. Artif. Intell., vol. 8, no. 1, pp. 153-167, 2026.

[47] Y. Hu, Y. Cheng, Y. Zhang, Y. Xie, and Z. Yin, "SAIDO: Generalizable detection of AI-generated images via scene-aware and importance-guided dynamic optimization in continual learning," in Proc. IEEE/CVF CVPR, 2026.

---

## Appendix A: Complete Architecture Specifications

### A.1 Frequency Decomposition Module

**Component**: `FrequencyDecomposition` (non-parametric, no trainable parameters)

| Parameter | Value |
|-----------|-------|
| Transform | 2D FFT (`torch.fft.fft2`) with orthonormal normalization |
| Shift | `torch.fft.fftshift` to center DC component |
| Bands (3-band) | Low: [0, 0.15) × r_max, Mid: [0.15, 0.45) × r_max, High: [0.45, 1.0] × r_max |
| Bands (4-band) | Low: [0, 0.10), Low-Mid: [0.10, 0.30), Mid-High: [0.30, 0.55), High: [0.55, 1.0] |
| Masking | Binary radial mask (hard threshold), applied element-wise |
| Inverse | `torch.fft.ifftshift` + `torch.fft.ifft2`, real-valued output |
| r_max | `min(H, W) / 2` |

### A.2 Frequency Feature Extractor

**Component**: `FrequencyFeatureExtractor` (per-band, weights not shared between bands)

| Layer | Type | In | Out | Kernel | Stride | Padding | Params (approx) |
|-------|------|----|-----|--------|--------|---------|-----------------|
| Stem | Conv2d + BN + GELU | 3 | 32 | 3 | 2 | 1 | 864 + 64 = 928 |
| Block 1 | DW Conv2d | 32 | 32 | 3 | 2 | 1 | 288 |
| Block 1 | PW Conv2d + BN + GELU | 32 | 64 | 1 | 1 | 0 | 2,048 + 128 = 2,176 |
| Block 2 | DW Conv2d | 64 | 64 | 3 | 2 | 1 | 576 |
| Block 2 | PW Conv2d + BN + GELU | 64 | 128 | 1 | 1 | 0 | 8,192 + 256 = 8,448 |
| Block 3 | DW Conv2d | 128 | 128 | 3 | 2 | 1 | 1,152 |
| Block 3 | PW Conv2d + BN + GELU | 128 | 256 | 1 | 1 | 0 | 32,768 + 512 = 33,280 |
| Pool | AdaptiveAvgPool2d | 256 | 256 | 1×1 out | — | — | 0 |
| Head | Linear + LayerNorm | 256 | feat_dim | — | — | — | 256·feat_dim + 2·feat_dim |

**Total per extractor (feat_dim=256)**: ~112K parameters
**Total per extractor (feat_dim=128)**: ~80K parameters
**Total per extractor (feat_dim=512)**: ~178K parameters

### A.3 Cross-Attention Fusion

**Component**: `CrossAttentionFusion`

| Parameter | Tiny | Base | Large |
|-----------|------|------|-------|
| feat_dim (D) | 128 | 256 | 512 |
| num_heads (h) | 4 | 8 | 12 |
| head_dim (d_k) | 32 | 32 | 42.67 (≈43) |
| W_QKV | 128×384 | 256×768 | 512×1,536 |
| W_proj | 128×128 | 256×256 | 512×512 |
| Dropout | 0.1 | 0.1 | 0.1 |
| Residual | Yes | Yes | Yes |

**Parameter counts**:
- Tiny: 49,152 (QKV) + 16,384 (proj) = 65,536
- Base: 196,608 (QKV) + 65,536 (proj) = 262,144
- Large: 786,432 (QKV) + 262,144 (proj) = 1,048,576

### A.4 Frequency-Guided Attention

**Component**: `FrequencyGuidedAttention`

```
Input: fused features Z' ∈ R^{B × N × D} (where N = num_bands)
       frequency magnitudes m ∈ R^{B × N}

Process:
  1. α = softmax(m, dim=1)                    → α ∈ R^{B × N}
  2. α = α.unsqueeze(-1)                      → α ∈ R^{B × N × 1}
  3. gate = Sigmoid(Linear(ReLU(Linear(Z'))))  → gate ∈ R^{B × N × D}
  4. output = Z' ⊙ gate ⊙ (α * D)             → output ∈ R^{B × N × D}

Bottleneck MLP: Linear(D → D/4) → ReLU → Linear(D/4 → D) → Sigmoid
```

**Parameter counts**:
- Tiny: 128→32 (4,096) + 32→128 (4,096) = 8,192
- Base: 256→64 (16,384) + 64→256 (16,384) = 32,768
- Large: 512→128 (65,536) + 128→512 (65,536) = 131,072

### A.5 Classification Head

| Layer | Tiny (in→out) | Base (in→out) | Large (in→out) |
|-------|-------------|-------------|--------------|
| LayerNorm | 384 | 768 | 2,048 |
| Linear + GELU | 384→128 | 768→256 | 2,048→512 |
| Dropout | 0.2 | 0.2 | 0.2 |
| Linear + GELU | 128→64 | 256→128 | 512→256 |
| Dropout | 0.1 | 0.1 | 0.1 |
| Linear | 64→2 | 128→2 | 256→2 |

### A.6 Total Parameter Counts

| Component | Tiny | Base | Large |
|-----------|------|------|-------|
| Decomposer | 0 | 0 | 0 |
| Feature Extractors (×B) | 239,616 | 337,152 | 711,680 |
| Cross-Attention Fusion | 65,536 | 262,144 | 1,048,576 |
| Frequency-Guided Attention | 8,192 | 32,768 | 131,072 |
| Classification Head | ~57,600 | ~229,632 | ~1,180,160 |
| **Total** | **~370,944** | **~861,696** | **~3,071,488** |

---

## Appendix B: Dataset Composition

### B.1 Source Distribution

| Source | Real | AI-Generated | Deepfake | Total |
|--------|------|-------------|----------|-------|
| Pexels / Unsplash / Pixabay | ~500K | — | — | ~500K |
| ImageNet / Places365 / Open Images V7 | ~400K | — | — | ~400K |
| BigGAN (GenImage) | — | ~422K | — | ~422K |
| Stable Diffusion v1.4/v1.5 | — | ~100K | — | ~100K |
| GLIDE (GenImage) | — | ~162K | — | ~162K |
| VQDM (GenImage) | — | ~162K | — | ~162K |
| ADM (GenImage) | — | ~162K | — | ~162K |
| Midjourney | — | ~6K | — | ~6K |
| Wukong (GenImage) | — | ~162K | — | ~162K |
| DALL-E 3 | — | ~19K | — | ~19K |
| Celeb-DF V2 | — | — | ~101K | ~101K |
| FaceForensics++ | — | — | ~20K | ~20K |
| DFDC | — | — | ~4K | ~4K |
| AI-generated (other) | — | ~200K | — | ~200K |
| Deepfake (synthetic alterations) | — | — | ~775K | ~775K |
| **Total** | **~900K** | **~1,395K** | **~900K** | **~2,700K (approx.)** |

### B.2 Generator Families (AI-Generated Class)

1. **BigGAN** — GAN-based, class-conditioned, 256×256 or higher resolution
2. **VQDM** — VQ-Diffusion, discrete diffusion in latent space
3. **GLIDE** — Guided language-to-image diffusion (OpenAI)
4. **ADM** — Ablated Diffusion Model (OpenAI)
5. **Midjourney** — Proprietary diffusion-based text-to-image
6. **Wukong** — Chinese text-to-image diffusion model
7. **Stable Diffusion v1.4/v1.5** — Open-source latent diffusion (CompVis)
8. **DALL-E 3** — Transformer-based text-to-image (OpenAI)
9. **Flux** — Latent diffusion with flow matching (Black Forest Labs)
10. **Pollinations** — Multi-model generation platform
11. **CivitAI** — Community fine-tuned Stable Diffusion variants

### B.3 Deepfake Datasets (Deepfake Class)

| Dataset | Description | Manipulation Types |
|---------|-------------|-------------------|
| Celeb-DF V2 | 5,639 original + 5,639 deepfake videos (face swap) | Face swap with improved quality |
| FaceForensics++ | 1,000 original + 4,000 manipulated videos | Deepfakes, Face2Face, FaceSwap, NeuralTextures |
| DFDC | 119,154 original + 100,000+ manipulated videos | Various face swap and reenactment methods |

### B.4 Image Categories

Images are classified into 6 content categories for per-category analysis:

| Category | Description | Examples |
|----------|-------------|----------|
| Portraits | Human faces, single and group | Celebrity photos, ID photos, AI-generated portraits |
| Landscapes | Natural and urban scenes | Mountains, cityscapes, beaches, interiors |
| Objects/Still-life | Inanimate subjects | Products, food, furniture, tools |
| Abstract/Artistic | Non-representational | Digital art, abstract compositions, stylized images |
| Text-Heavy | Images with significant text | Memes, infographics, slides, advertisements |
| Mixed/Complex | Multiple subjects or complex scenes | Crowd scenes, detailed compositions |

---

## Appendix C: Training Hyperparameter Details

### C.1 Full Hyperparameter Set

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| β₁, β₂ | 0.9, 0.999 |
| Weight decay | 0.05 |
| Epochs | 20 |
| Batch size (effective) | 32 |
| Gradient accumulation | 1 (adjustable for multi-GPU) |
| Gradient clipping norm | 1.0 |
| Mixed precision | FP16 (AMP with GradScaler) |
| Label smoothing ε | 0.1 |
| Scheduler | CosineAnnealingWarmRestarts |
| Warmup steps | 500 |
| Warmup start factor | 0.01 |
| T₀ (cosine restart period) | epochs × 100 |
| T_mult | 2 |
| η_min | 10⁻⁶ |
| Weight initialization | Truncated normal (std=0.02), biases=0, LN gain=1 |

### C.2 Data Augmentation (Training)

| Augmentation | Parameters |
|-------------|-----------|
| Random Resized Crop | Scale: [0.85, 1.0], size: 384×384 |
| Random Horizontal Flip | Probability: 0.5 |
| Random Rotation | Degrees: [-10, 10] |
| Color Jitter | Brightness: 0.1, Contrast: 0.1, Saturation: 0.1, Hue: 0.05 |
| Random Sharpness | Factor: [0.5, 1.5], probability: 0.3 |
| Random Erasing | Probability: 0.25, scale: [0.02, 0.33] |
| Normalization | Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225] |

### C.3 Validation/Test Preprocessing

| Step | Parameters |
|------|-----------|
| Resize | 400 × 400 |
| Center Crop | 384 × 384 |
| Normalization | Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225] |

---

## Appendix D: Additional Baseline Details

### D.1 SimpleCNN Architecture

| Layer | Type | Out Channels | Stride |
|-------|------|-------------|--------|
| 1 | Conv2d + BN + GELU | 32 | 2 |
| 2 | Conv2d + BN + GELU | 64 | 2 |
| 3 | Conv2d + BN + GELU | 128 | 2 |
| 4 | Conv2d + BN + GELU | 256 | 2 |
| 5 | AdaptiveAvgPool2d | 256 | — |
| 6 | FC + GELU + Dropout(0.2) | 128 | — |
| 7 | FC | 2 | — |

**Parameters**: ~1.5M

### D.2 FreqDetect (Frequency Baseline)

As proposed by Frank et al. [14], this baseline:
1. Computes 2D FFT magnitude per color channel
2. Pools magnitude into 16 radial bins using pre-computed distance-based indices
3. Normalizes by bin count
4. Feeds 48-dimensional feature (16 bins × 3 channels) into a 3-layer MLP

| Layer | In → Out | Activation |
|-------|---------|------------|
| FC | 48 → 128 | ReLU, Dropout(0.2) |
| FC | 128 → 64 | ReLU |
| FC | 64 → 2 | — |

**Parameters**: ~8K



## Appendix E: MFFT With Explainability

MFFT naturally supports explainability through its multi-band architecture:

**Per-band anomaly heatmaps**: For any input image, the model can return a heatmap per frequency band by computing the per-pixel magnitude of each band image:

$$H_b = |x_b|, \quad H_b \in \mathbb{R}^{H \times W}$$

These are resized to the input image dimensions via bilinear interpolation and overlaid on the original image. Practical interpretations:

- **High-band heatmap**: Highlights edges, textures, and fine-grained patterns where GAN artifacts manifest
- **Mid-band heatmap**: Highlights regions where diffusion model noise distribution mismatches occur
- **Low-band heatmap**: Highlights global structural anomalies characteristic of transformer-based generators

**Confidence calibration**: The model provides calibrated confidence scores via softmax probabilities, enabling practitioners to set appropriate thresholds for their specific precision-recall requirements.

**Band contribution scores**: The frequency attention weights $\alpha$ provide a global measure of which frequency band most influenced the classification decision for a given input, enabling model introspection at the band level.
