"""Multi-Task SSL loss for pretext task classification."""

from typing import Dict

import torch
import torch.nn.functional as F

from model.losses.base import BaseLoss


class MultiTaskSSLLoss(BaseLoss):
    """
    Multi-task self-supervised loss.

    Averages cross-entropy losses across multiple binary pretext tasks
    (arrow-of-time, segment permutation, time warp).
    """

    def __init__(self, reduction: str = 'mean', **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def forward(
        self,
        task_logits: Dict[str, torch.Tensor],
        task_labels: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute mean cross-entropy across all tasks.

        Args:
            task_logits: {task_name: [B, 2]} logits per task
            task_labels: {task_name: [B]} long labels per task

        Returns:
            Scalar loss
        """
        total = sum(
            F.cross_entropy(task_logits[name], task_labels[name])
            for name in task_logits
        )
        return total / len(task_logits)

    def compute_per_sample(
        self,
        task_logits: Dict[str, torch.Tensor],
        task_labels: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute per-sample loss averaged across tasks.

        Returns:
            Per-sample losses [B]
        """
        per_sample = sum(
            F.cross_entropy(task_logits[name], task_labels[name], reduction='none')
            for name in task_logits
        )
        return per_sample / len(task_logits)
