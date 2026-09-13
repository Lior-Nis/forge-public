"""iBOT metrics management for self-supervised pretraining."""

import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .pretrain_base import PretrainingMetricsManager

logger = logging.getLogger(__name__)


class IBOTMetricsManager(PretrainingMetricsManager):
    """Metrics manager for iBOT pretraining — tracks cosine similarity between student and teacher."""

    def __init__(self, device: torch.device, track_gradients: bool = True):
        super().__init__(device, track_gradients)

    def to(self, device: Optional[torch.device] = None):
        if device is not None:
            self.device = device
        return self

    def reset_metrics(self):
        if hasattr(super(), "reset"):
            super().reset()

    def update_metrics(self, probas: torch.Tensor, targets: torch.Tensor):
        """iBOT does not use the probas/targets paradigm — no-op."""
        pass

    def compute_reconstruction_metrics(
        self, student: torch.Tensor, teacher: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute iBOT-specific token-level metrics.

        Args:
            student: Student embeddings [B, N, D]
            teacher: Teacher embeddings [B, N, D]

        Returns:
            Dictionary of metrics.
        """
        metrics = {}
        with torch.no_grad():
            s_flat = student.reshape(-1, student.shape[-1])
            t_flat = teacher.reshape(-1, teacher.shape[-1])
            cos_sim = F.cosine_similarity(s_flat.float(), t_flat.float(), dim=-1)
            metrics["cosine_similarity"] = cos_sim.mean().item()
            metrics["student_std"] = student.std().item()
            metrics["teacher_std"] = teacher.std().item()
        return metrics
