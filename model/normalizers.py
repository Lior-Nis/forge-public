import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm1d(nn.Module):
    """
    GroupNorm implementation that properly handles 1D time series data.

    This module handles the dimension re-ordering required to apply GroupNorm
    correctly to 1D time series data (batch, channels, time) format.

    Args:
        num_groups: Number of groups to separate the channels into
        num_channels: Total number of channels
        eps: Small value added to variance for numerical stability
        affine: If True, apply learnable affine parameters
    """

    def __init__(
        self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True
    ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different tensor dimensions
        orig_shape = x.shape

        if len(x.shape) == 3:  # [batch, channels, time]
            # Rearrange to apply norm across channels
            x = x.permute(0, 2, 1)  # -> [batch, time, channels]
            x = x.reshape(-1, x.shape[-1])  # -> [batch*time, channels]
            x = self.norm(x.unsqueeze(2)).squeeze(2)  # Apply norm
            x = x.view(orig_shape[0], orig_shape[2], orig_shape[1])  # Restore shape
            x = x.permute(0, 2, 1)  # -> [batch, channels, time]
        else:  # [batch, channels]
            x = self.norm(x.unsqueeze(2)).squeeze(2)

        return x


class PreNormAttention(nn.Module):
    """
    Pre-normalization attention block with optional layer scaling.

    Applies layer normalization before self-attention, and optionally
    scales the output using learned scaling parameters.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        layer_scale_init_value: Initial value for layer scaling (0 to disable)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        layer_scale_init_value: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.use_layer_scale = True
        else:
            self.use_layer_scale = False

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        normed_x = self.norm(x)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x, attn_mask=attn_mask)

        if self.use_layer_scale:
            attn_output = self.gamma.unsqueeze(0).unsqueeze(1) * attn_output

        return x + self.dropout(attn_output)


class PreNormMLP(nn.Module):
    """
    Pre-normalization MLP block with optional layer scaling.

    Applies layer normalization before MLP, and optionally
    scales the output using learned scaling parameters.

    Args:
        dim: Feature dimension
        mlp_ratio: Ratio of hidden dimension to input dimension
        dropout: Dropout rate
        layer_scale_init_value: Initial value for layer scaling (0 to disable)
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init_value: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.use_layer_scale = True
        else:
            self.use_layer_scale = False

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed_x = self.norm(x)
        mlp_output = self.mlp(normed_x)

        if self.use_layer_scale:
            mlp_output = self.gamma.unsqueeze(0).unsqueeze(1) * mlp_output

        return x + self.dropout(mlp_output)


class PreNormBlock(nn.Module):
    """
    Complete pre-normalization Transformer block with layer scaling.

    Combines PreNormAttention and PreNormMLP into a single transformer block
    with pre-normalization and optional layer scaling.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of hidden dimension to input dimension
        dropout: Dropout rate
        layer_scale_init_value: Initial value for layer scaling
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init_value: float = 1e-5,
    ):
        super().__init__()

        self.attn = PreNormAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.mlp = PreNormMLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            layer_scale_init_value=layer_scale_init_value,
        )

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.attn(x, attn_mask=attn_mask)
        x = self.mlp(x)
        return x


def get_alibi_bias(
    num_heads: int, seq_len: int, device: torch.device = None
) -> torch.Tensor:
    """
    Generate ALiBi (Attention with Linear Biases) bias pattern.

    Creates a bias pattern that linearly decreases attention scores based on
    distance between tokens, with different slopes for each attention head.

    Args:
        num_heads: Number of attention heads
        seq_len: Sequence length
        device: Torch device to create tensor on

    Returns:
        Tensor of shape [num_heads, seq_len, seq_len] containing ALiBi bias pattern
    """
    # Initialize bias tensor
    bias = torch.zeros(num_heads, seq_len, seq_len, device=device)

    # Create distance matrix
    positions = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(2)
    positions_t = positions.transpose(1, 2)
    distance = torch.abs(positions - positions_t)

    # Apply different slope for each head
    for h in range(num_heads):
        # Heads get progressively stronger distance penalty
        # Using 2^-(h+a) as in the original ALiBi paper
        slope = -1.0 / (2 ** (h + 1))
        bias[h] = slope * distance.squeeze(0)

    return bias


def apply_alibi_to_model(model: nn.Module, num_heads: int, seq_len: int) -> nn.Module:
    """
    Apply ALiBi bias to an existing model with attention blocks.

    Searches for MultiheadAttention modules or attention blocks with
    relative position bias tables and updates them with ALiBi bias pattern.

    Args:
        model: Model to update
        num_heads: Number of attention heads
        seq_len: Sequence length

    Returns:
        Updated model with ALiBi bias applied
    """
    device = next(model.parameters()).device
    alibi_bias = get_alibi_bias(num_heads, seq_len, device)

    # Apply to all modules
    for name, module in model.named_modules():
        # Check for MultiheadAttention or custom attention modules
        if isinstance(module, nn.MultiheadAttention):
            # Add hooks to apply bias to attention scores
            def apply_alibi_hook(module, inputs, output):
                attn_output, attn_weights = output
                # Apply alibi bias to attention weights if available
                if attn_weights is not None:
                    attn_weights = attn_weights + alibi_bias.unsqueeze(0)
                return attn_output, attn_weights

            module.register_forward_hook(apply_alibi_hook)

        # For Vision Transformer from timm
        elif hasattr(module, "attn") and hasattr(module.attn, "rel_pos"):
            # Update relative position bias table
            if hasattr(module.attn.rel_pos, "relative_position_bias_table"):
                module.attn.rel_pos.relative_position_bias_table.data += (
                    alibi_bias.reshape(
                        -1, module.attn.rel_pos.relative_position_bias_table.shape[-1]
                    )
                )

    return model


def get_sinusoidal_pos_embed(
    seq_len: int, dim: int, cls_token: bool = False, base: int = 10000
) -> torch.Tensor:
    """
    Generate sinusoidal positional embeddings.

    Creates positional embeddings using sine and cosine functions
    with different frequencies, as in the original Transformer paper.

    Args:
        seq_len: Sequence length
        dim: Embedding dimension
        cls_token: Whether to include a position for a class token
        base: Base for exponentially increasing wavelengths

    Returns:
        Tensor of shape [1, seq_len(+1), dim] containing positional embeddings
    """
    # Create position indices
    positions = torch.arange(0, seq_len).float().unsqueeze(1)

    # Create dimension indices
    dim_indices = torch.arange(0, dim, 2).float()

    # Calculate exponentially decreasing wavelengths
    div_term = torch.exp(-math.log(base) * dim_indices / dim)

    # Calculate embeddings
    pos_embed = torch.zeros(1, seq_len, dim)
    pos_embed[0, :, 0::2] = torch.sin(positions * div_term)
    pos_embed[0, :, 1::2] = torch.cos(positions * div_term)

    # Add class token position if requested
    if cls_token:
        pos_embed_cls = torch.zeros(1, 1, dim)
        pos_embed = torch.cat([pos_embed_cls, pos_embed], dim=1)

    return pos_embed


def apply_sinusoidal_pos_embed(
    model: nn.Module, seq_len: int, embed_dim: int, mix_ratio: float = 0.2
) -> nn.Module:
    """
    Apply sinusoidal positional embeddings to an existing model.

    Searches for positional embedding parameter and updates it by
    mixing with sinusoidal embeddings.

    Args:
        model: Model to update
        seq_len: Sequence length
        embed_dim: Embedding dimension
        mix_ratio: How much of the sinusoidal embedding to mix in

    Returns:
        Updated model with mixed positional embeddings
    """
    # Generate sinusoidal embeddings
    has_cls = hasattr(model, "cls_token") and model.cls_token is not None
    sin_pos = get_sinusoidal_pos_embed(seq_len, embed_dim, has_cls)

    # Find and update positional embedding
    if hasattr(model, "pos_embed"):
        device = model.pos_embed.device
        # Mix learned and sinusoidal embeddings
        model.pos_embed.data = (
            1 - mix_ratio
        ) * model.pos_embed.data + mix_ratio * sin_pos.to(device)

    return model


def apply_layer_scaling(
    model: nn.Module, layer_scale_init: float, layer_scale_depth: float = 1.0
) -> nn.Module:
    """
    Apply progressive layer scaling to transformer blocks.

    Adds scaling parameters to attention and MLP outputs, with
    scaling that increases with network depth.

    Args:
        model: Model to update
        layer_scale_init: Initial scaling value
        layer_scale_depth: Factor to increase scaling by layer depth

    Returns:
        Updated model with layer scaling applied
    """
    # Find transformer blocks to update
    blocks = []
    for name, module in model.named_children():
        if name == "blocks" and isinstance(module, nn.ModuleList):
            blocks = module

    if not blocks:
        # Handle custom block naming as in ViT implementations
        for name, module in model.named_children():
            if name in ["encoder", "transformer"] and hasattr(module, "layers"):
                blocks = module.layers

    # Apply layer scaling to each block
    for i, block in enumerate(blocks):
        # Calculate scaling factor based on layer depth
        scale = layer_scale_init * (layer_scale_depth**i)

        # Apply to attention output
        if hasattr(block, "attn"):
            # Vision transformer style
            attn_path = block.attn if hasattr(block.attn, "proj") else block.attn.attn
            if not hasattr(attn_path, "gamma"):
                attn_path.gamma = nn.Parameter(
                    scale * torch.ones(attn_path.proj.out_features)
                )
            else:
                attn_path.gamma.data.fill_(scale)

        # Apply to MLP output
        if hasattr(block, "mlp"):
            # Vision transformer style
            if not hasattr(block.mlp, "gamma"):
                out_features = (
                    block.mlp[-1].out_features
                    if isinstance(block.mlp, nn.Sequential)
                    else block.mlp.fc2.out_features
                )
                block.mlp.gamma = nn.Parameter(scale * torch.ones(out_features))
            else:
                block.mlp.gamma.data.fill_(scale)

    return model


class NormalizerFactory:
    """
    Factory class for creating and applying normalizers to models.

    Provides a centralized way to apply various normalization techniques
    to backbone models with consistent configuration.
    """

    @staticmethod
    def apply_normalizations(model: nn.Module, config: Dict) -> nn.Module:
        """
        Apply multiple normalization techniques to a model based on config.

        Args:
            model: Model to update
            config: Dictionary of normalization configurations

        Returns:
            Updated model with normalizations applied
        """
        # Extract normalization parameters
        use_group_norm = config["use_group_norm"]
        pre_norm = config["pre_norm"]
        layer_scale_init = config["layer_scale_init"]
        layer_scale_depth = config["layer_scale_depth"]
        use_alibi = config["use_alibi"]
        custom_pos_embed = config["custom_pos_embed"]

        # Get model dimensions
        embed_dim = config["embed_dim"]
        num_heads = config["num_heads"]
        img_size = config["img_size"]
        patch_size = config["patch_size"]
        seq_len = (img_size // patch_size) ** 2  # For 2D data

        # Apply normalizations in appropriate order

        # 1. Layer scaling should be applied first
        if layer_scale_init > 0:
            model = apply_layer_scaling(model, layer_scale_init, layer_scale_depth)

        # 2. Custom positional embeddings
        if custom_pos_embed:
            model = apply_sinusoidal_pos_embed(model, seq_len, embed_dim)

        # 3. ALiBi attention bias
        if use_alibi:
            model = apply_alibi_to_model(model, num_heads, seq_len)

        # 4. Group normalization (requires custom patch embedding)
        if use_group_norm and hasattr(model, "patch_embed"):
            # Get the original patch embedding
            old_patch_embed = model.patch_embed

            # Create a new sequential module with GroupNorm1d added
            model.patch_embed = nn.Sequential(
                old_patch_embed, GroupNorm1d(min(8, embed_dim // 16), embed_dim)
            )

        return model
