"""
Patient-contrastive loss using Supervised Contrastive Learning (SupCon).

Uses patient_id as supervision signal: same patient = positive pair,
different patient = negative pair. Single-view (no dual augmentation).

Reference:
    "Supervised Contrastive Learning" - Khosla et al., NeurIPS 2020
"""

import logging
from typing import List

import torch
import torch.nn.functional as F

from model.losses.base import BaseLoss

logger = logging.getLogger(__name__)


class PatientContrastiveLoss(BaseLoss):
    """
    Supervised contrastive loss using patient_id as labels.

    For each anchor, positives are other samples from the same patient,
    negatives are samples from different patients. Anchors with no
    positives (solo patient in batch) are excluded from the mean.

    Args:
        temperature: Temperature scaling for similarities (default: 0.1)
        reduction: Reduction method ('mean', 'sum', 'none')
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

    def _encode_patient_ids(self, patient_ids: List[str], device: torch.device) -> torch.Tensor:
        """Convert patient_id strings to integer labels within this batch."""
        unique = {pid: i for i, pid in enumerate(sorted(set(patient_ids)))}
        labels = torch.tensor([unique[pid] for pid in patient_ids], device=device)
        return labels

    def _compute_loss(self, embeddings: torch.Tensor, patient_ids: List[str]):
        """
        Core SupCon computation.

        Returns:
            per_sample_losses: [B] tensor (0 for orphan anchors)
            orphan_mask: [B] bool tensor (True for anchors with no positives)
        """
        B = embeddings.shape[0]
        device = embeddings.device

        labels = self._encode_patient_ids(patient_ids, device)

        # Normalize embeddings
        z = F.normalize(embeddings, dim=-1)

        # Similarity matrix [B, B]
        sim = torch.matmul(z, z.T) / self.temperature

        # Positive mask: same patient, exclude self
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))  # [B, B]
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = pos_mask & ~diag_mask  # exclude self

        # Orphan anchors: no positives in batch
        num_positives = pos_mask.sum(dim=1)  # [B]
        orphan_mask = (num_positives == 0)

        # For numerical stability, subtract max from each row (excluding self)
        sim_for_max = sim.masked_fill(diag_mask, float('-inf'))
        sim_max, _ = sim_for_max.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Mask out self-similarity for denominator
        sim_no_self = sim.masked_fill(diag_mask, float('-inf'))

        # Log-sum-exp over all non-self entries (denominator)
        log_sum_exp = torch.logsumexp(sim_no_self, dim=1)  # [B]

        # Mean log-prob of positives for each anchor
        # Use torch.where to avoid -inf * 0 = NaN
        per_sample_losses = torch.zeros(B, device=device)
        valid = ~orphan_mask
        if valid.any():
            # Extract positive similarities safely (use 0 where not positive)
            pos_sim = torch.where(pos_mask, sim, torch.zeros_like(sim))
            pos_sum = pos_sim.sum(dim=1)  # [B]
            per_sample_losses[valid] = -(pos_sum[valid] / num_positives[valid].float() - log_sum_exp[valid])

        return per_sample_losses, orphan_mask

    def forward(
        self,
        embeddings: torch.Tensor,
        patient_ids: List[str],
    ) -> torch.Tensor:
        """
        Compute patient-contrastive loss.

        Args:
            embeddings: Projected representations [B, D]
            patient_ids: List of patient ID strings, length B

        Returns:
            Scalar loss (mean over non-orphan anchors)
        """
        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensor [B, D], got shape {embeddings.shape}")
        if len(patient_ids) != embeddings.shape[0]:
            raise ValueError(f"Batch size mismatch: embeddings {embeddings.shape[0]} vs patient_ids {len(patient_ids)}")

        per_sample_losses, orphan_mask = self._compute_loss(embeddings, patient_ids)

        valid = ~orphan_mask
        orphan_rate = orphan_mask.float().mean().item()
        if orphan_rate > 0:
            logger.debug(f"Orphan rate: {orphan_rate:.2%} ({orphan_mask.sum().item()}/{len(patient_ids)} anchors)")

        if not valid.any():
            logger.warning("All anchors are orphans — returning zero loss")
            return per_sample_losses.sum() * 0.0  # Differentiable zero

        if self.reduction == 'mean':
            return per_sample_losses[valid].mean()
        elif self.reduction == 'sum':
            return per_sample_losses[valid].sum()
        else:
            return per_sample_losses

    def compute_per_sample(
        self,
        embeddings: torch.Tensor,
        patient_ids: List[str],
    ) -> torch.Tensor:
        """
        Compute per-sample losses.

        Args:
            embeddings: Projected representations [B, D]
            patient_ids: List of patient ID strings, length B

        Returns:
            Per-sample losses [B] (orphans get 0)
        """
        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensor [B, D], got shape {embeddings.shape}")

        per_sample_losses, _ = self._compute_loss(embeddings, patient_ids)
        return per_sample_losses

    def __repr__(self):
        return f"PatientContrastiveLoss(temperature={self.temperature}, reduction={self.reduction})"
