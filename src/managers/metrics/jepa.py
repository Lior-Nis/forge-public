"""JEPA metrics management for self-supervised pretraining."""

import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .pretrain_base import PretrainingMetricsManager

logger = logging.getLogger(__name__)


class JEPAMetricsManager(PretrainingMetricsManager):
    """Metrics manager for JEPA pretraining — tracks embedding prediction quality."""

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
        """JEPA does not use probas/targets paradigm — no-op."""
        pass

    def compute_reconstruction_metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute JEPA-specific embedding prediction metrics.

        Args:
            outputs: Predicted embeddings [B, num_masked, D]
            targets: Target embeddings [B, num_masked, D]

        Returns:
            Dictionary of metrics.
        """
        metrics = {}
        with torch.no_grad():
            metrics["mse"] = F.mse_loss(outputs, targets).item()
            metrics["smooth_l1"] = F.smooth_l1_loss(outputs, targets).item()

            # Cosine similarity between predicted and target
            pred_flat = outputs.reshape(-1, outputs.shape[-1])
            tgt_flat = targets.reshape(-1, targets.shape[-1])
            cos_sim = F.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean()
            metrics["cosine_similarity"] = cos_sim.item()

            # Embedding statistics (collapse detection)
            metrics["pred_std"] = outputs.std().item()
            metrics["target_std"] = targets.std().item()

        return metrics
