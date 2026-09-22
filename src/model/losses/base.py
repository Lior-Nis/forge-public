"""
Base classes and utility functions for loss computation.

This module provides the foundation for all loss functions including:
- Utility functions for validation, weight computation, and reduction
- BaseLoss abstract class with common functionality
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_tensor_inputs(*tensors: torch.Tensor) -> bool:
    """
    Validate that tensors don't contain NaN or inf values.

    Args:
        *tensors: Variable number of tensors to validate

    Returns:
        True if all tensors are valid, False otherwise
    """
    for tensor in tensors:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False
    return True


def compute_inverse_frequency_weights(
    class_counts: List[float],
    normalize: bool = True,
    background_weight_factor: float = 1.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute class weights using inverse frequency weighting.

    Args:
        class_counts: Count of samples per class
        normalize: Whether to normalize weights to sum to num_classes
        background_weight_factor: Multiplicative factor for background class weight
        device: Target device for weights

    Returns:
        Tensor of class weights
    """
    class_counts = torch.tensor(class_counts, dtype=torch.float32)
    total_samples = class_counts.sum()
    weights = total_samples / (len(class_counts) * class_counts)

    if normalize:
        weights = weights * len(class_counts) / weights.sum()

    # Apply background weight adjustment if needed
    if background_weight_factor != 1.0 and len(weights) > 0:
        weights[0] = weights[0] * background_weight_factor
        if normalize:
            weights = weights * len(class_counts) / weights.sum()

    if device is not None:
        weights = weights.to(device)

    return weights


def compute_median_frequency_weights(
    labels: torch.Tensor,
    num_classes: Optional[int] = None
) -> torch.Tensor:
    """
    Compute class weights using median-frequency weighting.
    Formula: w_c = median(freq) / freq(c)

    Args:
        labels: Tensor of class labels
        num_classes: Number of classes (inferred if None)

    Returns:
        Tensor of class weights
    """
    class_freqs = torch.bincount(labels.flatten())

    if num_classes is None:
        num_classes = len(class_freqs)

    # Ensure we have frequencies for all classes
    if len(class_freqs) < num_classes:
        padded_freqs = torch.ones(num_classes, device=labels.device)
        padded_freqs[:len(class_freqs)] = class_freqs.float()
        class_freqs = padded_freqs
    else:
        class_freqs = class_freqs.float()

    # Avoid division by zero
    class_freqs = torch.clamp(class_freqs, min=1.0)

    # Compute median and weights
    median_freq = torch.median(class_freqs)
    weights = median_freq / class_freqs

    return weights


def apply_reduction(
    loss: torch.Tensor,
    reduction: str = 'mean',
    valid_count: Optional[int] = None
) -> torch.Tensor:
    """
    Apply reduction operation to loss tensor.

    Args:
        loss: Loss tensor to reduce
        reduction: Reduction method ('mean', 'sum', 'none')
        valid_count: Number of valid elements (for masked mean)

    Returns:
        Reduced loss
    """
    if reduction == 'mean':
        if valid_count is not None and valid_count > 0:
            return loss.sum() / valid_count
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


# ============================================================================
# BASE CLASSES
# ============================================================================

class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for loss functions providing common functionality.

    Implements:
    - Input validation
    - Reduction operations
    - Device handling
    - Optional masking support
    """

    def __init__(self, reduction: str = 'mean', **kwargs):
        super().__init__()
        self.reduction = reduction

    def _validate_inputs(self, *tensors: torch.Tensor) -> bool:
        """Validate input tensors for NaN/inf values."""
        return validate_tensor_inputs(*tensors)

    def _apply_reduction(
        self,
        loss: torch.Tensor,
        reduction: Optional[str] = None,
        valid_count: Optional[int] = None
    ) -> torch.Tensor:
        """Apply reduction to loss tensor."""
        reduction = reduction if reduction is not None else self.reduction
        return apply_reduction(loss, reduction, valid_count)

    def _apply_mask(
        self,
        loss: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply validity mask to loss tensor.

        Args:
            loss: Per-sample loss tensor
            valid_mask: Boolean mask where True indicates valid samples

        Returns:
            Masked loss tensor (invalid samples set to 0)
        """
        if valid_mask is not None:
            loss = loss * valid_mask.float()
        return loss

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute loss - must be implemented by subclasses."""
        pass

    def compute_per_sample(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute per-sample losses - should be implemented by subclasses that support it.

        Returns:
            Per-sample losses [batch_size]
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_per_sample()"
        )
