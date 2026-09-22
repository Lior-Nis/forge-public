"""
Composite loss functions for combining multiple losses.

This module provides wrappers and combiners:
- MaskedLossWrapper: Applies validity masking to any loss function
- JointLoss: Flexible weighted combination of multiple losses
"""

import inspect
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from model.losses.base import BaseLoss


logger = logging.getLogger(__name__)


class MaskedLossWrapper(BaseLoss):
    """
    Wrapper that applies validity masking to any loss function.

    Only computes loss on valid timesteps, maintaining the original loss
    characteristics while handling missing or invalid data.

    Args:
        base_loss: The underlying loss function to wrap
    """

    def __init__(self, base_loss: nn.Module):
        super().__init__(reduction='none')
        self.base_loss = base_loss

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Forward pass with validity masking.

        Handles both sequence-level and episode-level predictions:
        - Sequence: logits [batch, seq, classes], targets [batch, seq], mask [batch, seq]
        - Episode: logits [batch, classes], targets [batch], mask [batch]

        Args:
            logits: Model predictions
            targets: Ground truth labels
            valid_mask: Validity mask
            reduction: 'mean' for scalar loss, 'none' for per-sample losses

        Returns:
            Masked loss
        """
        assert (
            logits.shape[0] == valid_mask.shape[0] == targets.shape[0]
        ), "Batch dimensions must match"

        # Compute base loss
        loss = self.base_loss(logits, targets)

        # Apply mask
        loss[~valid_mask.bool()] = 0.0

        # Apply reduction
        if reduction == 'mean':
            total_valid = valid_mask.sum().item()
            if total_valid > 0:
                return loss.sum() / total_valid
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

    def __repr__(self):
        return f"MaskedLossWrapper({self.base_loss.__class__.__name__})"


class JointLoss(BaseLoss):
    """
    Flexible composite loss combining multiple loss functions.

    Supports dynamic weight normalization and component loss logging
    for monitoring individual contributions during training.

    Args:
        losses: List of loss configurations, each containing:
            - _target_ or target: Class path for the loss function
            - weight: Weight for this loss component
            - Additional loss-specific parameters

    Example config:
        losses:
          - _target_: torch.nn.MSELoss
            weight: 0.7
          - _target_: model.losses.SSIMLoss
            weight: 0.3
            window_size: 11
    """

    def __init__(self, losses: List[Dict], **kwargs):
        super().__init__(reduction='mean')

        if not losses:
            raise ValueError("At least one loss must be specified")

        self.loss_components = nn.ModuleList()
        self.loss_weights = []
        self.loss_names = []

        # Initialize each loss component
        for i, loss_config in enumerate(losses):
            loss_config = loss_config.copy()

            # Extract weight and target
            weight = loss_config.pop('weight', 1.0)
            target = loss_config.pop('_target_', loss_config.pop('target', None))

            if target is None:
                raise ValueError(
                    f"Loss config {i} must specify '_target_' or 'target'"
                )

            # Import and instantiate the loss class
            loss_instance = self._instantiate_loss(target, loss_config)

            self.loss_components.append(loss_instance)
            self.loss_weights.append(weight)
            self.loss_names.append(f"{loss_instance.__class__.__name__.lower()}")

        # Normalize weights to sum to 1.0
        total_weight = sum(self.loss_weights)
        self.loss_weights = [w / total_weight for w in self.loss_weights]

        # Cache signature introspection results (avoid inspect.signature() in hot path)
        self._loss_signatures = []
        for loss_fn in self.loss_components:
            sig_forward = inspect.signature(loss_fn.forward)
            has_compute_per_sample = hasattr(loss_fn, 'compute_per_sample')
            sig_per_sample = (
                inspect.signature(loss_fn.compute_per_sample)
                if has_compute_per_sample else None
            )
            self._loss_signatures.append({
                'forward_has_patch_size': 'patch_size' in sig_forward.parameters,
                'forward_has_mask': 'mask' in sig_forward.parameters,
                'has_compute_per_sample': has_compute_per_sample,
                'per_sample_has_patch_size': (
                    sig_per_sample is not None
                    and 'patch_size' in sig_per_sample.parameters
                ),
            })

    def _instantiate_loss(
        self,
        target: str,
        config: Dict[str, Any]
    ) -> nn.Module:
        """Dynamically instantiate a loss class using Hydra."""
        import hydra

        # Build config dict with _target_
        full_config = {"_target_": target, **config}

        # Use Hydra's instantiate for type-safe dynamic loading
        return hydra.utils.instantiate(full_config)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute weighted combination of all loss components.

        Args:
            pred: Predicted data
            target: Target data
            mask: Optional mask (passed to losses that support it)
            patch_size: Optional patch size (for MAE losses with masking)

        Returns:
            Combined weighted loss
        """
        total_loss = 0.0

        for loss_fn, weight, sig_info in zip(
            self.loss_components, self.loss_weights, self._loss_signatures
        ):
            try:
                if sig_info['forward_has_patch_size'] and sig_info['forward_has_mask']:
                    component_loss = loss_fn(pred, target, mask, patch_size)
                elif sig_info['forward_has_mask']:
                    component_loss = loss_fn(pred, target, mask)
                else:
                    component_loss = loss_fn(pred, target)

                total_loss += weight * component_loss

            except Exception as e:
                logger.warning(
                    f"Failed to compute {loss_fn.__class__.__name__}: {e}"
                )
                continue

        return total_loss

    def compute_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute per-sample joint losses.

        Matches classification pipeline API for clean logging integration.

        Args:
            pred: Predicted data
            target: Target data
            mask: Optional mask
            patch_size: Optional patch size (for MAE losses with masking)

        Returns:
            Per-sample combined losses [batch_size]
        """
        per_sample_total = None

        for loss_fn, weight, sig_info in zip(
            self.loss_components, self.loss_weights, self._loss_signatures
        ):
            try:
                if sig_info['has_compute_per_sample']:
                    if sig_info['per_sample_has_patch_size']:
                        component_per_sample = loss_fn.compute_per_sample(pred, target, mask, patch_size)
                    else:
                        component_per_sample = loss_fn.compute_per_sample(pred, target, mask)
                else:
                    # Fallback: temporarily set reduction='none'
                    if hasattr(loss_fn, 'reduction'):
                        original_reduction = loss_fn.reduction
                        loss_fn.reduction = 'none'
                        try:
                            if sig_info['forward_has_patch_size'] and sig_info['forward_has_mask']:
                                component_loss = loss_fn(pred, target, mask, patch_size)
                            elif sig_info['forward_has_mask']:
                                component_loss = loss_fn(pred, target, mask)
                            else:
                                component_loss = loss_fn(pred, target)

                            # Ensure per-sample shape
                            if component_loss.dim() > 1:
                                component_per_sample = component_loss.view(component_loss.shape[0], -1).mean(dim=1)
                            else:
                                component_per_sample = component_loss
                        finally:
                            loss_fn.reduction = original_reduction
                    else:
                        # Last resort: compute with mean and broadcast
                        if sig_info['forward_has_patch_size'] and sig_info['forward_has_mask']:
                            scalar_loss = loss_fn(pred, target, mask, patch_size)
                        elif sig_info['forward_has_mask']:
                            scalar_loss = loss_fn(pred, target, mask)
                        else:
                            scalar_loss = loss_fn(pred, target)
                        component_per_sample = scalar_loss.expand(pred.shape[0])

                # Accumulate weighted per-sample losses
                if per_sample_total is None:
                    per_sample_total = weight * component_per_sample
                else:
                    per_sample_total += weight * component_per_sample

            except Exception as e:
                logger.warning(f"Failed to compute per-sample for {loss_fn.__class__.__name__}: {e}")
                continue

        if per_sample_total is None:
            # Fallback if all components failed
            return torch.zeros(pred.shape[0], device=pred.device)

        return per_sample_total

    def get_component_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get individual loss components for logging.

        Args:
            pred: Predicted data
            target: Target data
            mask: Optional mask
            patch_size: Optional patch size (for MAE losses with masking)

        Returns:
            Dictionary with component losses and total loss
        """
        component_losses = {}
        total_loss = 0.0

        for loss_fn, weight, name, sig_info in zip(
            self.loss_components, self.loss_weights, self.loss_names,
            self._loss_signatures
        ):
            try:
                if sig_info['forward_has_patch_size'] and sig_info['forward_has_mask']:
                    component_loss = loss_fn(pred, target, mask, patch_size)
                elif sig_info['forward_has_mask']:
                    component_loss = loss_fn(pred, target, mask)
                else:
                    component_loss = loss_fn(pred, target)

                component_losses[name] = component_loss
                total_loss += weight * component_loss

            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")
                continue

        component_losses['total_loss'] = total_loss
        return component_losses

    def __repr__(self):
        components = [
            f"{name}({weight:.2f})"
            for name, weight in zip(self.loss_names, self.loss_weights)
        ]
        return f"JointLoss({', '.join(components)})"
