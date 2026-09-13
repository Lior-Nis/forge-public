"""
Backbone models for FoG detection from accelerometer data.

This module contains all backbone architectures for processing accelerometer signals
and their spectral representations. Includes temporal models, transformers, 
vision models, and ensemble architectures.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from typing import Dict, Optional, Union

from model.normalizers import NormalizerFactory


class FogFormer(nn.Module):
    """
    TemporalFormer backbone implementing ViT-based patch-wise feature extraction.

    This backbone focuses on the first stage of the 3rd place competition solution:
    - Patch-wise feature extraction using ViT components

    The temporal sequence modeling (RNN) is now handled by separate heads,
    allowing for more flexible mixing and matching of different temporal
    modeling approaches.

    Supports any spectral representation from the transform pipeline.
    """

    def __init__(
        self,
        patch_size: int = 12,
        embed_dim: int = 256,
        num_heads: int = 8,
        vit_depth: int = 3,
        dropout: float = 0.2,
        use_alibi: bool = True,
        pre_norm: bool = False,
        activation: str = "GELU",
        input_channels: int = 3,
        **kwargs,
    ):
        super(FogFormer, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.vit_depth = vit_depth
        self.dropout = dropout
        self.use_alibi = use_alibi
        self.pre_norm = pre_norm
        self.input_channels = input_channels

        # Activation function selection
        if activation == "GELU":
            self.activation_fn = F.gelu
        elif activation == "CELU":
            self.activation_fn = F.celu
        elif activation == "ReLU":
            self.activation_fn = F.relu
        else:
            self.activation_fn = F.gelu

        # Initialize layers immediately
        self._initialize_layers()

        # Output dimension is the embedding dimension from ViT
        self.output_dim = embed_dim

    def _initialize_layers(self):
        """Initialize layers based on expected input dimensions."""
        # Use provided input dimensions instead of lazy initialization
        C = self.input_channels

        # Input projection to handle different spectral representations
        # Frequency collapse: adaptive pooling handles any H dimension
        # Temporal patching: fixed kernel size for temporal patches
        self.freq_collapse = nn.AdaptiveAvgPool2d((1, None))  # Collapse H → 1, keep W
        self.input_projection = nn.Conv2d(
            C,
            self.embed_dim,
            kernel_size=(1, self.patch_size),  # 1 × patch_size (H-agnostic!)
            stride=(1, self.patch_size),       # Non-overlapping temporal patches
        )

        # Patch embedding with positional encoding
        self.patch_embedding = nn.Sequential(
            nn.Flatten(2), nn.Dropout(self.dropout)  # [B, embed_dim, num_patches]
        )

        # ALiBi positional bias if enabled (will be set dynamically based on sequence length)
        if self.use_alibi:
            # Create a placeholder - will be resized in forward pass if needed
            self.alibi_bias = None

        # ViT layers for patch-wise feature extraction
        vit_layers = []
        for _ in range(self.vit_depth):
            vit_layers.extend(
                [
                    MultiHeadAttention(
                        embed_dim=self.embed_dim,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                        use_alibi=self.use_alibi,
                        pre_norm=self.pre_norm,
                        activation_fn=self.activation_fn,
                    ),
                    FeedForward(
                        embed_dim=self.embed_dim,
                        hidden_dim=self.embed_dim * 4,
                        dropout=self.dropout,
                        activation_fn=self.activation_fn,
                        pre_norm=self.pre_norm,
                    ),
                ]
            )
        self.vit_layers = nn.ModuleList(vit_layers)

    def _get_alibi_bias(self, seq_len, num_heads):
        """Generate ALiBi positional bias for better temporal position encoding."""
        # Create relative position matrix
        positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(
            seq_len
        ).unsqueeze(1)

        # ALiBi slopes - different for each head
        slopes = torch.pow(2, -torch.arange(1, num_heads + 1) * 8.0 / num_heads)

        # Apply slopes to positions
        alibi_bias = positions.unsqueeze(0) * slopes.unsqueeze(1).unsqueeze(2)

        return alibi_bias

    def forward(self, x):
        """
        Forward pass implementing ViT-based patch-wise feature extraction with frequency preservation.

        Args:
            x: Input tensor of shape [B, C, H, W] from any transform
               (mel spectrograms, wavelets, GAF, etc.)

        Returns:
            Tensor of shape [B, num_patches, embed_dim] - sequence of patch embeddings
        """
        B, C, H, W = x.shape

        # Calculate number of patches from time dimension
        num_patches = W // self.patch_size

        # Update ALiBi bias if needed
        if self.use_alibi and (self.alibi_bias is None or self.alibi_bias.shape[-1] != num_patches):
            self.alibi_bias = self._get_alibi_bias(num_patches, self.num_heads).to(x.device)

        # Frequency-preserving temporal patch extraction (H-agnostic)
        # Step 1: Collapse frequency dimension adaptively (works with any H)
        x = self.freq_collapse(x)  # [B, C, H, W] → [B, C, 1, W]
        # Step 2: Extract temporal patches with full frequency content
        x = self.input_projection(x)  # [B, C, 1, W] → [B, embed_dim, 1, num_patches]
        x = x.squeeze(2)  # [B, embed_dim, num_patches]
        x = x.permute(0, 2, 1)  # [B, num_patches, embed_dim]

        # Apply patch embedding and dropout
        patch_embeddings = x

        # Apply ViT layers (attention + feedforward)
        for i in range(0, len(self.vit_layers), 2):
            # Multi-head attention
            patch_embeddings = self.vit_layers[i](
                patch_embeddings, alibi_bias=self.alibi_bias if self.use_alibi else None
            )
            # Feedforward
            patch_embeddings = self.vit_layers[i + 1](patch_embeddings)

        # Return sequence of frequency-rich temporal patch embeddings for head processing
        return patch_embeddings  # [B, num_patches, embed_dim]


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional ALiBi bias."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        use_alibi=False,
        pre_norm=False,
        activation_fn=F.gelu,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_alibi = use_alibi
        self.pre_norm = pre_norm

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, alibi_bias=None):
        B, seq_len, embed_dim = x.shape

        # Pre-normalization if enabled
        if self.pre_norm:
            x_norm = self.layer_norm(x)
        else:
            x_norm = x

        # Generate Q, K, V
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(B, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation in float32 to prevent fp16 overflow → NaN in softmax
        # (matmul(Q,K^T) can produce inf in fp16 → softmax(inf) = NaN)
        q_f32, k_f32, v_f32 = q.float(), k.float(), v.float()
        attn_weights = torch.matmul(q_f32, k_f32.transpose(-2, -1)) / (self.head_dim**0.5)

        # Add ALiBi bias if enabled
        if self.use_alibi and alibi_bias is not None:
            attn_weights = attn_weights + alibi_bias.float().to(attn_weights.device)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v_f32).to(q.dtype)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, seq_len, embed_dim)

        # Output projection and residual connection
        output = self.out_proj(attn_output)
        output = x + self.dropout(output)

        # Post-normalization if not pre-norm
        if not self.pre_norm:
            output = self.layer_norm(output)

        return output


class FeedForward(nn.Module):
    """Feedforward network with configurable activation."""

    def __init__(
        self, embed_dim, hidden_dim, dropout=0.1, activation_fn=F.gelu, pre_norm=False
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.activation_fn = activation_fn

        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Pre-normalization if enabled
        if self.pre_norm:
            x_norm = self.layer_norm(x)
        else:
            x_norm = x

        # Feedforward computation
        output = self.linear1(x_norm)
        output = self.activation_fn(output)
        output = self.dropout(output)
        output = self.linear2(output)

        # Residual connection
        output = x + self.dropout(output)

        # Post-normalization if not pre-norm
        if not self.pre_norm:
            output = self.layer_norm(output)

        return output


# =============================================================================
# TRANSFORMER MODELS
# =============================================================================

class FOGTransformerBackbone(nn.Module):
    """
    FOG Transformer backbone based on the winning solution.
    Uses transformer encoder layers without the final BiLSTM (which goes in temporal).
    """

    def __init__(
        self,
        input_dim: int,
        sequence_len: int,
        model_dim: int = 320,
        num_heads: int = 6,
        num_layers: int = 5,
        first_dropout: float = 0.1,
        encoder_dropout: float = 0.1,
        mha_dropout: float = 0.0,
        normalize_factor: float = 50.0,
        **kwargs,
    ):
        super().__init__()

        # Model parameters from winning solution
        self.input_dim = input_dim  # patch_size * channels
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.first_dropout = first_dropout
        self.encoder_dropout = encoder_dropout
        self.mha_dropout = mha_dropout
        self.sequence_len = sequence_len  # num_patches
        self.normalize_factor = normalize_factor

        # Input projection
        self.first_linear = nn.Linear(self.input_dim, self.model_dim)
        self.first_dropout_layer = nn.Dropout(self.first_dropout)

        # Positional encoding (learnable, like in the winning solution)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.sequence_len, self.model_dim) * 0.02
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.model_dim * 4,  # Standard 4x expansion
            dropout=self.encoder_dropout,
            activation="relu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Output dimension
        self.output_dim = self.model_dim

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, num_patches, patch_size * channels]
        Returns:
            Features of shape [batch_size, num_patches, model_dim]
        """
        batch_size = x.shape[0]

        # Normalize input (like in winning solution)
        x = x / self.normalize_factor

        # Project to model dimension
        x = self.first_linear(x)

        # Add positional encoding with random roll augmentation during training
        if self.training:
            # Random roll for each sample in batch
            shifts = torch.randint(
                -self.sequence_len, 0, (batch_size,), device=x.device
            )
            pos_encoding_batch = self.pos_encoding.repeat(batch_size, 1, 1)

            # Roll each sample individually
            for i in range(batch_size):
                pos_encoding_batch[i] = torch.roll(
                    pos_encoding_batch[i], shifts=int(shifts[i]), dims=0
                )

            x = x + pos_encoding_batch
        else:
            # No augmentation during inference
            x = x + self.pos_encoding

        x = self.first_dropout_layer(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        return x


# =============================================================================
# VISION MODELS
# =============================================================================

class ViT(nn.Module):
    """
    Vision Transformer model for spectral representations of time series.

    Treats input as a 2D image (e.g., mel-spectrogram or wavelet transform)
    where temporal dynamics are already captured in the representation.
    """

    def __init__(
        self,
        input_channels: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        depth: int = 6,
        dropout: float = 0.3,
        norm_layer: str = "nn.LayerNorm",
        global_pool: str = "avg",
        pretrained: bool = False,
        pretrained_model: str = "vit_large_patch16_384",
        **kwargs,
    ):
        super().__init__()
        # Model parameters
        self.input_channels = input_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.depth = depth
        self.dropout = dropout
        self.norm_layer = _get_norm_layer(norm_layer)
        self.global_pool = global_pool
        self.use_pretrained = pretrained
        self.pretrained_model = pretrained_model

        # Vision Transformer for feature extraction
        if self.use_pretrained and self.pretrained_model:
            # Load a pretrained ViT model
            self.vit = timm.create_model(
                self.pretrained_model,
                pretrained=True,
                img_size=self.img_size,
                in_chans=self.input_channels,
                num_classes=0,  # Remove classifier head
                global_pool=self.global_pool,  # Use average pooling for features
            )
            # Get embed_dim from the model
            self.embed_dim = self.vit.embed_dim
        else:
            # Create a fresh ViT model
            self.vit = VisionTransformer(
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_chans=self.input_channels,
                num_classes=0,  # No classification head
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.dropout,
                attn_drop_rate=self.dropout,
                drop_path_rate=self.dropout,
                norm_layer=self.norm_layer,
                global_pool=self.global_pool,  # Use average pooling
            )

        # Create config dictionary for NormalizerFactory
        normalizer_config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "use_group_norm": kwargs.get("use_group_norm", False),
            "pre_norm": kwargs.get("pre_norm", False),
            "layer_scale_init": kwargs.get("layer_scale_init", 0.0),
            "layer_scale_depth": kwargs.get("layer_scale_depth", 1.0),
            "use_alibi": kwargs.get("use_alibi", False),
            "custom_pos_embed": kwargs.get("custom_pos_embed", False),
        }
        self = NormalizerFactory.apply_normalizations(self, normalizer_config)

        # Output dimension for the head
        self.output_dim = self.embed_dim

    def forward(self, x):
        """
        Forward pass through the Vision Transformer.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
               This represents spectral images (e.g., mel-spectrograms)

        Returns:
            Features of shape [batch_size, embed_dim]
        """
        # ViT expects input shape [B, C, H, W]
        features = self.vit(x)
        return features


class ResNet(nn.Module):
    """
    ResNet model for spectral representations of time series.

    Uses a standard 2D ResNet to process spectral images
    (e.g., mel-spectrograms or wavelet transforms).
    """

    def __init__(
        self,
        input_channels: int = 3,
        resnet_model: str = "resnet50",
        pretrained: bool = False,
        global_pool: str = "avg",
        **kwargs,
    ):
        super().__init__()
        # Model parameters
        self.input_channels = input_channels
        self.resnet_model = resnet_model
        self.pretrained = pretrained
        self.global_pool = global_pool

        # Load ResNet from timm
        self.backbone = timm.create_model(
            self.resnet_model,
            pretrained=self.pretrained,
            in_chans=self.input_channels,
            num_classes=0,  # Remove classifier
            global_pool=self.global_pool,  # Use average pooling
        )

        # Get the feature dimension
        # This will be model-specific (e.g., 2048 for ResNet50, 512 for ResNet18)
        # LEGITIMATE HASATTR: Introspecting external library (timm) with varying APIs
        # Different timm backbones expose dimensions differently:
        # - ResNet family: .num_features
        # - Vision Transformers: .feature_info
        # This adapts to external library variations, not hiding config errors.
        if hasattr(self.backbone, "num_features"):
            self.feature_dim = self.backbone.num_features
        elif hasattr(self.backbone, "feature_info"):
            self.feature_dim = self.backbone.feature_info[-1]["num_chs"]
        else:
            raise ValueError(
                f"Cannot determine feature dimension for {type(self.backbone).__name__}. "
                "Expected 'num_features' or 'feature_info' attribute."
            )

        # Output dimension for the head
        self.output_dim = self.feature_dim

    def forward(self, x):
        """
        Forward pass through the ResNet.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
               This represents spectral images (e.g., mel-spectrograms)

        Returns:
            Features of shape [batch_size, feature_dim]
        """
        # ResNet expects input shape [B, C, H, W]
        features = self.backbone(x)
        return features


class SpectralPatchEncoder(nn.Module):
    """
    2D patch encoder for spectrograms without interpolation.

    Replaces the FogFormer's `AdaptiveAvgPool2d + 1D Conv` stem with a single
    asymmetric Conv2d that creates 2D patch tokens directly from the raw
    spectrogram. Each token encodes a (freq_patch × time_patch) region,
    preserving frequency structure instead of collapsing it.

    Design for [B, C, 100, 1000] (wavelet, no interpolation):
        freq_patch=20, time_patch=50  →  5 × 20 = 100 tokens  (same as FogFormer)
        freq_patch=10, time_patch=10  →  10 × 100 = 1000 tokens (high-res, slower)

    The token sequence [B, N, D] is fed to a standard Transformer encoder (same
    ALiBi-based ViT blocks as FogFormer). Positional encoding is 2D-aware:
    tokens are flattened row-major (freq-major), so ALiBi relative bias still
    captures local temporal proximity.

    Works with the existing MAE pipeline and 2D patch masking (mask_mode="2d_patch").
    The backbone exposes `patch_size` = time_patch for compatibility with the MAE
    pipeline's temporal masking helpers; freq_patch is stored separately.
    """

    def __init__(
        self,
        freq_patch: int = 20,
        time_patch: int = 50,
        embed_dim: int = 512,
        num_heads: int = 8,
        vit_depth: int = 4,
        dropout: float = 0.1,
        use_alibi: bool = True,
        input_channels: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.freq_patch = freq_patch
        self.time_patch = time_patch
        self.patch_size = time_patch   # MAE pipeline compatibility
        self.embed_dim = embed_dim
        self.output_dim = embed_dim

        # Single asymmetric Conv2d: maps [B, C, H, W] → [B, D, H/fp, W/tp]
        # No frequency collapse — each token covers a full (freq_patch × time_patch) block
        self.patch_embed = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=(freq_patch, time_patch),
            stride=(freq_patch, time_patch),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Standard transformer encoder (same blocks as FogFormer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=vit_depth)

        self.use_alibi = use_alibi
        self._alibi_cache: dict = {}

    def _get_alibi_bias(self, n_tokens: int, device: torch.device) -> torch.Tensor | None:
        """ALiBi bias [num_heads, N, N] — cached per sequence length."""
        if not self.use_alibi:
            return None
        if n_tokens not in self._alibi_cache:
            num_heads = self.transformer.layers[0].self_attn.num_heads
            slopes = torch.pow(2, -torch.arange(1, num_heads + 1, dtype=torch.float32) * 8 / num_heads)
            pos = torch.arange(n_tokens, dtype=torch.float32)
            rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # [N, N]
            bias = -slopes.unsqueeze(-1).unsqueeze(-1) * rel.unsqueeze(0)  # [H, N, N]
            self._alibi_cache[n_tokens] = bias
        return self._alibi_cache[n_tokens].to(device)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] spectrogram — H and W must be divisible by freq_patch/time_patch

        Returns:
            [B, N, embed_dim] where N = (H/freq_patch) * (W/time_patch)
        """
        # Truncate to patch boundaries
        H, W = x.shape[2], x.shape[3]
        H_use = (H // self.freq_patch) * self.freq_patch
        W_use = (W // self.time_patch) * self.time_patch
        if H_use != H or W_use != W:
            x = x[:, :, :H_use, :W_use]

        # Patch embedding: [B, C, H, W] → [B, D, nH, nW]
        tokens = self.patch_embed(x)                          # [B, D, nH, nW]
        B, D, nH, nW = tokens.shape

        # Store grid dimensions for downstream consumers (e.g. classification pipeline)
        self._last_nH = nH
        self._last_nW = nW

        # Flatten 2D grid to sequence: [B, D, nH, nW] → [B, nH*nW, D]
        tokens = tokens.flatten(2).permute(0, 2, 1)           # [B, N, D]
        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)

        # Transformer with ALiBi
        N = tokens.shape[1]
        alibi = self._get_alibi_bias(N, tokens.device)
        if alibi is not None or causal:
            num_heads = self.transformer.layers[0].self_attn.num_heads
            B_heads = B * num_heads
            mask = None
            if alibi is not None:
                # Expand for batch: TransformerEncoderLayer expects [B*H, N, N]
                mask = alibi.unsqueeze(0).expand(B, -1, -1, -1).reshape(B_heads, N, N)
            if causal:
                # Tokens are freq-major; causal attention blocks future time positions across frequencies.
                time_idx = torch.arange(N, device=tokens.device) % nW
                causal_mask = torch.where(
                    time_idx.unsqueeze(0) <= time_idx.unsqueeze(1),
                    torch.tensor(0.0, device=tokens.device, dtype=torch.float32),
                    torch.tensor(float('-inf'), device=tokens.device, dtype=torch.float32),
                )
                causal_mask = causal_mask.unsqueeze(0).expand(B_heads, N, N)
                mask = causal_mask if mask is None else mask + causal_mask
            tokens = self.transformer(tokens, mask=mask)
        else:
            tokens = self.transformer(tokens)

        return tokens  # [B, N, embed_dim]


class ResNetEncoder(nn.Module):
    """
    ResNet feature extractor for MAE pretraining on 2D spectrograms.

    Unlike ResNet (which uses global avg pooling → [B, D]), this backbone
    removes global pooling and extracts spatial feature maps at an intermediate
    layer (layer3, stride-16). The frequency dimension is collapsed with
    AdaptiveAvgPool2d, yielding a 1D token sequence [B, N, D] that is
    compatible with the existing MAEDecoderHead and temporal masking logic.

    Design rationale:
    - Wavelet input: [B, 3, 100, 224]
    - Layer3 output (stride-16): [B, 1024, ~6, 14]
    - After freq collapse: [B, 14, 1024]  (14 temporal tokens)
    - patch_size=16 aligns with stride-16: each token ↔ 16 input timesteps
    - Temporal masking (patch_size=16) on the input zeroes exactly the
      columns that map to each token, giving a proper masked reconstruction task.
    """

    def __init__(
        self,
        input_channels: int = 3,
        resnet_model: str = "resnext50_32x4d",
        pretrained: bool = False,
        patch_size: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.resnet_model = resnet_model
        self.pretrained = pretrained
        self.patch_size = patch_size  # required by MAE pipeline

        # Feature extractor: layer3 output only (stride-16, 1024 channels)
        self.backbone = timm.create_model(
            self.resnet_model,
            pretrained=self.pretrained,
            in_chans=self.input_channels,
            num_classes=0,
            global_pool="",         # no pooling — keep spatial dims
            features_only=True,     # return intermediate feature maps
            out_indices=(3,),       # layer3 only (stride-16, 1024ch for resnext50)
        )

        # Determine feature channels at layer3
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 64, 64)
            feats = self.backbone(dummy)
            self._feature_channels = feats[0].shape[1]

        # Collapse frequency (H) dimension → [B, C, 1, W'] → [B, W', C]
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        self.embed_dim = self._feature_channels
        self.output_dim = self._feature_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] masked spectrogram

        Returns:
            [B, N, embed_dim] token sequence where N = W / patch_size
        """
        # Truncate W to be divisible by patch_size so output token count = W // patch_size.
        # The MAE pipeline's temporal mask already does this truncation before calling the
        # backbone, but we guard here to handle direct usage and edge cases.
        W = x.shape[-1]
        W_usable = (W // self.patch_size) * self.patch_size
        if W_usable != W:
            x = x[..., :W_usable]

        # Extract layer3 feature map: [B, 1024, H/16, W/16]
        features = self.backbone(x)[0]

        # Collapse frequency dim: [B, 1024, 1, W/16]
        features = self.freq_pool(features)

        # Reshape to token sequence: [B, W/16, 1024]
        B, C, _, N = features.shape
        tokens = features.squeeze(2).permute(0, 2, 1)  # [B, N, C]
        return tokens


class EfficientNet(nn.Module):
    """
    EfficientNet model for spectral representations of time series.

    More efficient CNN architecture than ResNet, with better
    accuracy-to-parameters ratio for image-like spectral data.
    """

    def __init__(
        self,
        input_channels: int = 3,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        feature_dim: int = 1280,
        **kwargs,
    ):
        super().__init__()
        # Model parameters
        self.input_channels = input_channels
        self.model_name = model_name
        self.pretrained = pretrained
        self.feature_dim = feature_dim  # Will be overridden

        # Load EfficientNet from timm
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            in_chans=self.input_channels,
            num_classes=0,  # Remove classifier
            global_pool="avg",  # Use average pooling
        )

        # Get the feature dimension
        # LEGITIMATE HASATTR: Introspecting external library (timm)
        if hasattr(self.backbone, "num_features"):
            self.feature_dim = self.backbone.num_features
        else:
            raise ValueError(
                f"Cannot determine feature dimension for {self.model_name}. "
                "Expected 'num_features' attribute."
            )

        # Output dimension for the head
        self.output_dim = self.feature_dim

        self.losses = {}

    def forward(self, x):
        """
        Forward pass through the EfficientNet.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
               This represents spectral images (e.g., mel-spectrograms)

        Returns:
            Features of shape [batch_size, feature_dim]
        """
        # EfficientNet expects input shape [B, C, H, W]
        features = self.backbone(x)
        return features


class ConvNeXt(nn.Module):
    """
    ConvNeXt model for spectral representations of time series.

    Modern convolutional architecture that performs competitively with
    transformers on image tasks, suitable for spectral data.
    """

    def __init__(
        self,
        input_channels: int = 3,
        model_name: str = "convnext_tiny",
        pretrained: bool = False,
        feature_dim: int = 768,
        global_pool: str = "avg",
        **kwargs,
    ):
        super().__init__()
        # Model parameters
        self.input_channels = input_channels
        self.model_name = model_name
        self.pretrained = pretrained
        self.feature_dim = feature_dim
        self.global_pool = global_pool

        # Load ConvNeXt from timm
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            in_chans=self.input_channels,
            num_classes=0,  # Remove classifier head
            global_pool=self.global_pool,
        )

        # Get the feature dimension
        # LEGITIMATE HASATTR: Introspecting external library (timm)
        if hasattr(self.backbone, "num_features"):
            self.feature_dim = self.backbone.num_features
        else:
            raise ValueError(
                f"Cannot determine feature dimension for {self.model_name}. "
                "Expected 'num_features' attribute."
            )

        # Output dimension for the head
        self.output_dim = self.feature_dim

    def forward(self, x):
        """
        Forward pass through the ConvNeXt.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
               This represents spectral images (e.g., mel-spectrograms)

        Returns:
            Features of shape [batch_size, feature_dim] if global_pool is used,
            or [batch_size, seq_len, feature_dim] if global_pool is disabled.
        """
        # ConvNeXt expects input shape [B, C, H, W]
        features = self.backbone(x)

        # If global pooling is disabled, reshape for sequence processing
        if self.global_pool == "":
            # Input: (batch_size, feature_dim, height, width)
            # Output: (batch_size, seq_len, feature_dim)
            features = features.flatten(2).transpose(1, 2)

        return features


class DenseNet(nn.Module):
    """
    DenseNet model for spectral representations of time series.

    Uses dense connectivity pattern to enhance feature propagation and
    reduce parameter count, suitable for spectral images.
    """

    def __init__(
        self,
        input_channels: int = 3,
        model_name: str = "densenet121",
        pretrained: bool = True,
        feature_dim: int = 1024,
        **kwargs,
    ):
        super().__init__()
        # Model parameters
        self.input_channels = input_channels
        self.model_name = model_name
        self.pretrained = pretrained
        self.feature_dim = feature_dim  # Will be overridden

        # Load DenseNet from timm
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            in_chans=self.input_channels,
            num_classes=0,  # Remove classifier
            global_pool="avg",  # Use average pooling
        )

        # Get the feature dimension
        # LEGITIMATE HASATTR: Introspecting external library (timm)
        if hasattr(self.backbone, "num_features"):
            self.feature_dim = self.backbone.num_features
        else:
            raise ValueError(
                f"Cannot determine feature dimension for {self.model_name}. "
                "Expected 'num_features' attribute."
            )

        # Output dimension for the head
        self.output_dim = self.feature_dim

        self.losses = {}

    def forward(self, x):
        """
        Forward pass through the DenseNet.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
               This represents spectral images (e.g., mel-spectrograms)

        Returns:
            Features of shape [batch_size, feature_dim]
        """
        # DenseNet expects input shape [B, C, H, W]
        features = self.backbone(x)
        return features


def _get_norm_layer(norm_layer_str: str) -> type:
    """
    Safely get normalization layer class from string.

    Args:
        norm_layer_str: String representation of the norm layer

    Returns:
        Normalization layer class
    """
    norm_layer_map = {
        "nn.LayerNorm": nn.LayerNorm,
        "nn.BatchNorm1d": nn.BatchNorm1d,
        "nn.BatchNorm2d": nn.BatchNorm2d,
        "nn.GroupNorm": nn.GroupNorm,
        "nn.InstanceNorm1d": nn.InstanceNorm1d,
        "nn.InstanceNorm2d": nn.InstanceNorm2d,
    }

    if norm_layer_str in norm_layer_map:
        return norm_layer_map[norm_layer_str]
    else:
        raise ValueError(f"Unsupported norm layer: {norm_layer_str}")


# =============================================================================
# MAMBA (SELECTIVE STATE SPACE) BACKBONE
# =============================================================================


class MambaBlock(nn.Module):
    """
    Single Mamba block with selective state space model (SSM).

    Implements the core Mamba operation:
      1. Input projection → x (signal branch) + z (gate branch)
      2. Causal depthwise conv1d for local context
      3. Selective SSM: input-dependent A, B, C parameters
      4. Gated output projection back to d_model

    Pure-PyTorch implementation — no external mamba-ssm dependency.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        # Causal conv: padding=d_conv-1 then trim ensures no future leakage
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )
        # Projects inner dim → B (d_state), C (d_state), log_Δ (1)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Log-parameterized A (negative eigenvalues ensure stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))  # skip connection

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model] → [B, L, d_model]"""
        B, L, _ = x.shape
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)                           # [B, L, 2*d_inner]
        x_in, z = xz.chunk(2, dim=-1)                  # [B, L, d_inner] each

        # Causal depthwise conv1d
        x_in = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)  # [B, L, d_inner]
        x_in = F.silu(x_in)

        # Compute input-dependent SSM parameters (selectivity)
        ssm = self.x_proj(x_in)                        # [B, L, 2*d_state+1]
        B_t, C_t, log_dt = ssm.split([self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(log_dt))           # [B, L, d_inner]

        A = -torch.exp(self.A_log.float())              # [d_inner, d_state]

        y = self._selective_scan(x_in.float(), dt.float(), A, B_t.float(), C_t.float())
        y = y.to(x_in.dtype)
        y = y + self.D * x_in                          # skip connection
        y = y * F.silu(z)                              # gating
        y = self.out_proj(y)
        y = self.drop(y)

        return y + residual

    def _selective_scan(
        self,
        u: torch.Tensor,      # [B, L, d_inner]
        delta: torch.Tensor,  # [B, L, d_inner]
        A: torch.Tensor,      # [d_inner, d_state]
        B: torch.Tensor,      # [B, L, d_state]
        C: torch.Tensor,      # [B, L, d_state]
    ) -> torch.Tensor:
        """
        Sequential SSM scan — O(L) steps, O(B*d_inner*d_state) memory per step.
        Computes dA/dB step-by-step to avoid materializing [B, L, d_inner, d_state]
        (which is ~1.7 GB at batch_size=512).
        """
        B_size, L, d_inner = u.shape

        h = torch.zeros(B_size, d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            delta_i = delta[:, i]                                          # [B, d_inner]
            dA_i = torch.exp(delta_i.unsqueeze(-1) * A.unsqueeze(0))      # [B, d_inner, d_state]
            dB_i = delta_i.unsqueeze(-1) * B[:, i].unsqueeze(1)           # [B, d_inner, d_state]
            h = dA_i * h + dB_i * u[:, i].unsqueeze(-1)                   # [B, d_inner, d_state]
            ys.append((h * C[:, i].unsqueeze(1)).sum(-1))                  # [B, d_inner]
        return torch.stack(ys, dim=1)                                      # [B, L, d_inner]


class MambaBackbone(nn.Module):
    """
    Bidirectional Mamba backbone for FoG detection.

    Architecture mirrors FogFormer:
      1. Frequency collapse (AdaptiveAvgPool2d) — H-agnostic, works with any transform
      2. Temporal patch embedding (Conv2d with patch_size stride)
      3. N bidirectional Mamba layers (forward + backward, merged via linear projection)
      4. LayerNorm output

    Returns [B, num_patches, embed_dim] — compatible with all existing heads.

    Why Mamba for FoG:
      - FoG is a gait *state transition* — Mamba's selective gates can ignore stable
        walking cycles and activate when dynamics change.
      - Linear O(L) complexity vs O(L²) attention — efficient on long sequences.
      - Bidirectional scan captures both "what preceded" and "what follows" each patch,
        useful for transition detection.
    """

    def __init__(
        self,
        patch_size: int = 10,
        embed_dim: int = 256,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        input_channels: int = 3,
        bidirectional: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.input_channels = input_channels

        # Frequency collapse + patch embedding (identical to FogFormer, H-agnostic)
        self.freq_collapse = nn.AdaptiveAvgPool2d((1, None))
        self.input_projection = nn.Conv2d(
            input_channels, embed_dim,
            kernel_size=(1, patch_size), stride=(1, patch_size),
        )

        # Forward Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(embed_dim, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])

        if bidirectional:
            # Separate set of Mamba layers for the reversed sequence
            self.backward_layers = nn.ModuleList([
                MambaBlock(embed_dim, d_state, d_conv, expand, dropout)
                for _ in range(num_layers)
            ])
            # Merge forward + backward representations
            self.merge_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)

        self.norm = nn.LayerNorm(embed_dim)
        self.output_dim = embed_dim  # required by BackboneConfig schema

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] — any transform output (raw [H=1], wavelet [H=100], etc.)
        Returns:
            [B, num_patches, embed_dim]
        """
        # Frequency collapse + patch embedding (same as FogFormer)
        x = self.freq_collapse(x)                    # [B, C, H, W] → [B, C, 1, W]
        x = self.input_projection(x)                 # [B, embed_dim, 1, N]
        x = x.squeeze(2).permute(0, 2, 1)            # [B, N, embed_dim]

        if self.bidirectional:
            fwd = x
            bwd = x.flip(1)                          # reverse sequence for backward pass
            for f_layer, b_layer in zip(self.layers, self.backward_layers):
                # Gradient checkpointing: recompute activations during backward
                # instead of storing all L SSM hidden states → ~3× memory reduction
                fwd = torch.utils.checkpoint.checkpoint(f_layer, fwd, use_reentrant=False)
                bwd = torch.utils.checkpoint.checkpoint(b_layer, bwd, use_reentrant=False)
            bwd = bwd.flip(1)                        # restore original order
            x = self.merge_proj(torch.cat([fwd, bwd], dim=-1))  # [B, N, embed_dim]
        else:
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)

        return self.norm(x)                          # [B, N, embed_dim]


# =============================================================================
# ENSEMBLE MODELS
# =============================================================================

class EnsembleBackbone(nn.Module):
    """
    Combines two backbone networks with configurable fusion strategies.
    Designed to work with any existing backbone without modification.
    The ensemble handles different output dimensions and shapes by dynamically
    identifying sequential and global feature streams.
    """

    def __init__(
        self,
        backbone1: nn.Module,
        backbone2: nn.Module,
        fusion_method: str = "mlp_fusion",
        fusion_dropout: float = 0.1,
        reduce_dim: Optional[int] = None,
        fusion_weights: Optional[Union[float, tuple]] = None,
        **kwargs,
    ):
        super().__init__()

        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.fusion_method = fusion_method
        self.fusion_dropout = fusion_dropout
        self.reduce_dim = reduce_dim

        # Get output dimensions
        # LEGITIMATE HASATTR: Validating component interface contract
        # All backbones MUST expose output_dim - this enforces the requirement
        if not hasattr(self.backbone1, "output_dim") or not hasattr(self.backbone2, "output_dim"):
            raise ValueError(
                "Both backbones must have 'output_dim' attribute. "
                f"backbone1 ({type(self.backbone1).__name__}): {hasattr(self.backbone1, 'output_dim')}, "
                f"backbone2 ({type(self.backbone2).__name__}): {hasattr(self.backbone2, 'output_dim')}"
            )

        self.backbone1_dim = self.backbone1.output_dim
        self.backbone2_dim = self.backbone2.output_dim

        # Setup fusion layer based on method
        self._setup_fusion_layer(fusion_weights)

        # Set final output dimension
        if self.fusion_method == "concat":
            self.output_dim = self.backbone1_dim + self.backbone2_dim
        elif self.fusion_method in ["add", "weighted_add"]:
            if self.backbone1_dim != self.backbone2_dim:
                raise ValueError(
                    f"For {self.fusion_method}, both backbones must have same output_dim"
                )
            self.output_dim = self.backbone1_dim
        elif self.fusion_method == "mlp_fusion":
            self.output_dim = self.reduce_dim or max(
                self.backbone1_dim, self.backbone2_dim
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def _setup_fusion_layer(self, fusion_weights: Optional[Union[float, tuple]]):
        """Setup fusion layers based on fusion method."""
        if self.fusion_method == "weighted_add":
            if fusion_weights is None:
                self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
            elif isinstance(fusion_weights, (list, tuple)):
                self.fusion_weights = nn.Parameter(torch.tensor(fusion_weights))
            else:
                self.fusion_weights = nn.Parameter(
                    torch.tensor([fusion_weights, 1.0 - fusion_weights])
                )

        elif self.fusion_method == "mlp_fusion":
            fusion_input_dim = self.backbone1_dim + self.backbone2_dim
            fusion_output_dim = (
                self.reduce_dim
                if self.reduce_dim
                else max(self.backbone1_dim, self.backbone2_dim)
            )

            self.fusion_mlp = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_input_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.fusion_dropout),
                nn.Linear(fusion_input_dim // 2, fusion_output_dim),
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through both backbones and fusion layer.
        Args:
            inputs: A dictionary of tensors, typically from a CompositeTransform.
                    Expected to contain two tensors for the two backbones.
        Returns:
            Fused features maintaining the temporal structure of the sequential input.
        """
        # If input is a single tensor, pass it to the first backbone.
        # This allows single-branch models to reuse the ensemble config structure.
        if isinstance(inputs, torch.Tensor):
            return self.backbone1(inputs)

        if len(inputs) != 2:
            raise ValueError(
                f"EnsembleBackbone expects exactly 2 inputs, but got {len(inputs)}"
            )

        # Process through both backbones
        f1 = self.backbone1(list(inputs.values())[0])
        f2 = self.backbone2(list(inputs.values())[1])

        # Dynamically identify sequential and global features
        if f1.ndim == 3 and f2.ndim == 2:
            seq_features, global_features = f1, f2
        elif f2.ndim == 3 and f1.ndim == 2:
            seq_features, global_features = f2, f1
        else:
            raise ValueError(
                "Ensemble requires one sequential [B, S, F] and one global [B, F] backbone."
            )

        # Expand global features to match temporal dimension of sequential features
        seq_len = seq_features.shape[1]
        global_features_expanded = global_features.unsqueeze(1).expand(-1, seq_len, -1)

        # Apply fusion
        if self.fusion_method == "concat":
            fused_features = torch.cat([seq_features, global_features_expanded], dim=-1)

        elif self.fusion_method == "add":
            fused_features = seq_features + global_features_expanded

        elif self.fusion_method == "weighted_add":
            weights = torch.softmax(self.fusion_weights, dim=0)
            fused_features = (
                weights[0] * seq_features + weights[1] * global_features_expanded
            )

        elif self.fusion_method == "mlp_fusion":
            concat_features = torch.cat(
                [seq_features, global_features_expanded], dim=-1
            )
            fused_features = self.fusion_mlp(concat_features)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return fused_features


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_backbone(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function for easy backbone instantiation.
    
    Args:
        model_name: Name of the backbone model
        **kwargs: Model-specific parameters
        
    Returns:
        Instantiated backbone model
        
    Example:
        >>> backbone = create_backbone('FogFormer', embed_dim=256, num_heads=8)
        >>> backbone = create_backbone('FOGTransformerBackbone', base_channels=64)
    """
    model_registry = {
        'FogFormer': FogFormer,
        
        # Transformer models
        'FOGTransformerBackbone': FOGTransformerBackbone,
        
        # Vision models
        'ViT': ViT,
        'ResNet': ResNet,
        'EfficientNet': EfficientNet,
        'ConvNeXt': ConvNeXt,
        'DenseNet': DenseNet,
        
        # Ensemble models
        'EnsembleBackbone': EnsembleBackbone,

        # SSM models
        'MambaBackbone': MambaBackbone,
        'MOMENTBackbone': MOMENTBackbone,

        # Spectral models
        'SpectralPatchEncoder': SpectralPatchEncoder,
    }

    if model_name not in model_registry:
        available_models = list(model_registry.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")

    return model_registry[model_name](**kwargs)


class MOMENTBackbone(nn.Module):
    """
    MOMENT-1-large foundation model backbone for time series classification.

    Wraps the AutonLab/MOMENT-1-large pretrained model (341M params, T5-based encoder)
    pretrained on a diverse corpus of time series data. Used as a frozen feature extractor
    for downstream FOG classification.

    Input:  [B, C, T] — raw accelerometer signal (C=3, T=seq_len)
    Output: [B, 1024] — mean-pooled patch embeddings

    Notes:
    - MOMENT normalizes inputs internally (RevIN) — do not pre-normalize
    - patch_len=8; seq_len should be a multiple of 8 (1000 → padded to 1000 internally)
    - Frozen by default; set freeze=False only for full fine-tuning
    """

    def __init__(
        self,
        pretrained_model_name: str = "AutonLab/MOMENT-1-large",
        freeze: bool = True,
        **kwargs,
    ):
        super().__init__()
        try:
            from momentfm import MOMENTPipeline
        except ImportError:
            raise ImportError(
                "momentfm is required for MOMENTBackbone. "
                "Install with: uv pip install momentfm --no-deps && uv pip install transformers"
            )

        self.moment = MOMENTPipeline.from_pretrained(
            pretrained_model_name,
            model_kwargs={"task_name": "embedding"},
        )

        if freeze:
            for p in self.moment.parameters():
                p.requires_grad = False
            self.moment.eval()

        self.output_dim = 1024  # MOMENT-1-large d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] — accepts any seq_len; MOMENT handles padding internally

        Returns:
            [B, 1024] embedding
        """
        # MOMENT expects channel-first [B, C, T] — already our format
        # Strip extra dims if transform added them (e.g. RawSignalTransform gives [B, C, 1, T])
        if x.dim() == 4:
            x = x.squeeze(2)

        out = self.moment.embed(x_enc=x)

        return out.embeddings  # [B, 1024]


# Export all models
__all__ = [
    # Temporal models
    "FogFormer",
    "MultiHeadAttention",
    "FeedForward",
    
    # Transformer models
    "FOGTransformerBackbone",
    
    # Vision models
    "ViT",
    "ResNet", 
    "EfficientNet",
    "ConvNeXt",
    "DenseNet",
    
    # Ensemble models
    "EnsembleBackbone",

    # SSM models
    "MambaBlock",
    "MambaBackbone",

    # Foundation models
    "MOMENTBackbone",

    # Spectral models
    "SpectralPatchEncoder",

    # Factory function
    "create_backbone",
]
