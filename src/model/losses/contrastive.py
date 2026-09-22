"""
Contrastive loss functions for self-supervised learning.

This module provides loss functions for contrastive learning:
- SimCLRLoss: NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.losses.base import BaseLoss

logger = logging.getLogger(__name__)


class SimCLRLoss(BaseLoss):
    """
    SimCLR contrastive loss using NT-Xent (Normalized Temperature-scaled Cross Entropy).

    Computes contrastive loss between two augmented views of the same input.
    The loss encourages representations of positive pairs (different views of the same sample)
    to be similar while pushing apart representations of negative pairs (different samples).

    Supports both standard 2D [B, D] and patch-level 3D [B, N, D] inputs.
    For 3D inputs, N independent NT-Xent problems are solved simultaneously via
    batched matrix multiplication — no Python loop over positions.

    Args:
        temperature: Temperature parameter for scaling similarities (default: 0.1)
        reduction: Reduction method for loss ('mean', 'sum', or 'none')

    Reference:
        "A Simple Framework for Contrastive Learning of Visual Representations"
        Chen et al., ICML 2020
    """

    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = 'mean',
        **kwargs,
    ):
        super().__init__(reduction=reduction)
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature

    def _nt_xent(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> tuple:
        """
        Vectorized NT-Xent for z1, z2 of shape [B, D] or [B, N, D].

        For 3D input, runs N independent NT-Xent problems simultaneously via bmm.
        Diagonal (self-similarity) is masked with -inf before softmax.

        Returns:
            logits: [2B, 2B] for 2D input | [N*2B, 2B] for 3D input
            labels: [2B]     for 2D input | [N*2B]     for 3D input
        """
        is_3d = z1.dim() == 3
        if not is_3d:
            z1 = z1.unsqueeze(1)
            z2 = z2.unsqueeze(1)

        B, N, D = z1.shape

        z = torch.cat([z1, z2], dim=0)          # [2B, N, D]
        z = F.normalize(z, dim=-1)
        z_n = z.permute(1, 0, 2)                # [N, 2B, D]
        sim = torch.bmm(z_n, z_n.transpose(1, 2)) / self.temperature  # [N, 2B, 2B]

        # Mask self-similarities
        eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye.unsqueeze(0), float('-inf'))

        # Positive for view-1 sample i is view-2 at index i+B, and vice versa
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(B, device=z.device),
        ])  # [2B]

        logits = sim.reshape(N * 2 * B, 2 * B)
        labels = labels.unsqueeze(0).expand(N, -1).reshape(-1)  # [N*2B]

        return logits, labels

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent contrastive loss.

        Args:
            z1: Representations from first view.
                2D [batch_size, feature_dim]              — standard SimCLR
                3D [batch_size, num_patches, feature_dim] — patch-level SimCLR

        Returns:
            Contrastive loss scalar.
        """
        if z1.shape != z2.shape:
            raise ValueError(f"Shape mismatch: z1 {z1.shape} vs z2 {z2.shape}")
        if z1.dim() not in (2, 3):
            raise ValueError(f"Expected 2D or 3D tensors, got shape {z1.shape}")

        logits, labels = self._nt_xent(z1, z2)
        return F.cross_entropy(logits, labels, reduction=self.reduction)

    def compute_per_sample(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample contrastive losses.

        Args:
            z1: First view. [batch_size, feature_dim] or [batch_size, num_patches, feature_dim]
            z2: Second view. Same shape as z1.

        Returns:
            Per-sample losses [batch_size]
        """
        if z1.shape != z2.shape:
            raise ValueError(f"Shape mismatch: z1 {z1.shape} vs z2 {z2.shape}")
        if z1.dim() not in (2, 3):
            raise ValueError(f"Expected 2D or 3D tensors, got shape {z1.shape}")

        B = z1.shape[0]
        N = z1.shape[1] if z1.dim() == 3 else 1

        logits, labels = self._nt_xent(z1, z2)
        per_element = F.cross_entropy(logits, labels, reduction='none')  # [N*2B]

        # Average view-1 and view-2 losses, then average across positions
        per_element = per_element.view(N, 2 * B)
        per_sample_per_pos = (per_element[:, :B] + per_element[:, B:]) / 2.0  # [N, B]
        return per_sample_per_pos.mean(dim=0)  # [B]

    def compute_per_position(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute mean NT-Xent loss per patch position.

        Args:
            z1: [batch_size, N, feature_dim]
            z2: [batch_size, N, feature_dim]

        Returns:
            Per-position losses [N]
        """
        if z1.dim() != 3:
            raise ValueError(f"Expected 3D tensors [batch, patches, features], got {z1.shape}")

        B, N, D = z1.shape
        logits, labels = self._nt_xent(z1, z2)
        per_element = F.cross_entropy(logits, labels, reduction='none')  # [N*2B]
        return per_element.view(N, 2 * B).mean(dim=1)  # [N]

    def __repr__(self):
        return f"SimCLRLoss(temperature={self.temperature}, reduction={self.reduction})"
