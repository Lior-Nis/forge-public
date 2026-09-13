"""
Segmentation loss functions for sequence labeling tasks.

This module provides loss functions for segmentation tasks:
- SegmentationLoss: Cross-entropy for sequence segmentation
- SoftDiceLoss: Dice loss for detecting small FoG episodes
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.losses.base import BaseLoss

logger = logging.getLogger(__name__)


class SegmentationLoss(BaseLoss):
    """
    Cross-entropy loss for sequence segmentation with optional focal weighting.

    Handles class imbalance through configurable weights and label smoothing.

    Args:
        class_weights: Per-class weights for imbalance handling
        focal_gamma: Focal loss gamma (0 disables focal weighting)
        label_smoothing: Label smoothing factor
    """

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        focal_gamma: float = 0.0,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(reduction='mean')
        self.focal_gamma = focal_gamma

        weight = torch.tensor(class_weights) if class_weights else None
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
            reduction="none"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute segmentation loss.

        Args:
            logits: Predicted logits [batch, seq_len, num_classes]
            targets: Target class indices [batch, seq_len]

        Returns:
            Segmentation loss
        """
        num_classes = logits.shape[-1]

        # Reshape for cross-entropy: [N, C] and [N]
        loss = self.ce_loss(
            logits.reshape(-1, num_classes),
            targets.reshape(-1).long()
        )

        # Apply focal weighting if enabled
        if self.focal_gamma > 0:
            p = torch.exp(-loss)
            focal_loss = (1 - p) ** self.focal_gamma * loss
            return focal_loss.mean()

        return loss.mean()


class SoftDiceLoss(BaseLoss):
    """
    Soft Dice loss for detecting small FoG episodes.

    More sensitive to small positive regions than cross-entropy.
    Particularly useful for segmentation tasks with class imbalance.

    Args:
        smooth: Smoothing factor to avoid division by zero
        reduction: Reduction method
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__(reduction=reduction)
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Soft Dice loss.

        Args:
            logits: Model predictions [batch_size, seq_len, num_classes]
            targets: Target class indices [batch_size, seq_len]
            mask: Optional validity mask [batch_size, seq_len]

        Returns:
            Dice loss (1 - dice_coefficient)
        """
        # Convert logits to probabilities
        if logits.dim() == 3:
            pred_probs = F.softmax(logits, dim=-1)
            if pred_probs.shape[-1] == 2:
                # Binary: use positive class
                pred_probs = pred_probs[..., 1]
            else:
                # Multiclass: 1 - background probability
                pred_probs = 1.0 - pred_probs[..., 0]
        else:
            pred_probs = torch.sigmoid(logits)

        # Convert targets to binary (FoG vs No-FoG)
        if targets.dim() == 2:
            binary_targets = (targets > 0).float()
        else:
            binary_targets = targets.float()

        # Apply validity mask if provided
        if mask is not None:
            pred_probs = pred_probs * mask.float()
            binary_targets = binary_targets * mask.float()

        # Flatten and compute Dice coefficient
        pred_probs_flat = pred_probs.view(-1)
        binary_targets_flat = binary_targets.view(-1)

        intersection = (pred_probs_flat * binary_targets_flat).sum()
        union = pred_probs_flat.sum() + binary_targets_flat.sum()

        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coefficient

        return self._apply_reduction(dice_loss * pred_probs_flat.numel()
                                     if self.reduction == 'sum' else dice_loss)
