"""
VICReg loss for self-supervised learning.

Addresses the low-rank collapse failure of SimCLR on bandlimited accelerometer data.
The variance and covariance terms structurally prevent the model from collapsing all
representations into a low-dimensional subspace, regardless of data rank.

Reference:
    "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
    Bardes et al., ICLR 2022
    https://arxiv.org/abs/2105.04906
"""

import logging

import torch
import torch.nn.functional as F

from model.losses.base import BaseLoss

logger = logging.getLogger(__name__)


class VICRegLoss(BaseLoss):
    """
    VICReg loss combining three complementary regularization terms:

    1. Variance: Pushes the standard deviation of each embedding dimension toward 1
       across the batch. Prevents dimensional collapse (all samples mapping to same point).

    2. Invariance: MSE between z1 and z2. Encourages the two augmented views of the
       same sample to produce similar representations.

    3. Covariance: Penalizes off-diagonal elements of the covariance matrix of embeddings.
       Forces different embedding dimensions to encode different information. Directly
       counters the low-rank structure of bandlimited accelerometer data.

    Args:
        lambda_var: Weight for variance term (default: 25.0)
        lambda_inv: Weight for invariance term (default: 25.0)
        lambda_cov: Weight for covariance term (default: 1.0)
        epsilon: Small constant for variance stability (default: 1e-4)
        reduction: Reduction method ('mean', 'sum', or 'none') — only affects per_sample
    """

    def __init__(
        self,
        lambda_var: float = 25.0,
        lambda_inv: float = 25.0,
        lambda_cov: float = 1.0,
        epsilon: float = 1e-4,
        reduction: str = 'mean',
        **kwargs,
    ):
        super().__init__(reduction=reduction)
        self.lambda_var = lambda_var
        self.lambda_inv = lambda_inv
        self.lambda_cov = lambda_cov
        self.epsilon = epsilon

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Hinge loss pushing per-dimension std toward 1. [B, D] -> scalar."""
        std = torch.sqrt(z.var(dim=0) + self.epsilon)
        return F.relu(1.0 - std).mean()

    def _invariance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """MSE between the two views. [B, D] -> scalar."""
        return F.mse_loss(z1, z2)

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Penalizes off-diagonal covariance. [B, D] -> scalar."""
        B, D = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (B - 1)  # [D, D]
        # Sum squared off-diagonal elements, normalize by D
        off_diag = cov.pow(2)
        off_diag.fill_diagonal_(0.0)
        return off_diag.sum() / D

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute VICReg loss.

        Args:
            z1: Embeddings from first view [batch_size, feature_dim]
            z2: Embeddings from second view [batch_size, feature_dim]

        Returns:
            Scalar VICReg loss
        """
        if z1.shape != z2.shape:
            raise ValueError(f"Shape mismatch: z1 {z1.shape} vs z2 {z2.shape}")
        if z1.dim() != 2:
            raise ValueError(f"Expected 2D tensors [batch, features], got shape {z1.shape}")
        if z1.shape[0] < 2:
            raise ValueError(f"Batch size must be >= 2 for covariance computation, got {z1.shape[0]}")

        var_loss = self._variance_loss(z1) + self._variance_loss(z2)
        inv_loss = self._invariance_loss(z1, z2)
        cov_loss = self._covariance_loss(z1) + self._covariance_loss(z2)

        loss = self.lambda_var * var_loss + self.lambda_inv * inv_loss + self.lambda_cov * cov_loss

        logger.debug(
            f"VICReg components — var: {var_loss.item():.4f}, "
            f"inv: {inv_loss.item():.4f}, cov: {cov_loss.item():.4f}, "
            f"total: {loss.item():.4f}"
        )
        return loss

    def compute_per_sample(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample VICReg loss for logging.

        The variance and covariance terms are batch-level statistics (cannot be
        decomposed per sample). The per-sample contribution is approximated as
        the invariance term (MSE per sample) plus a uniform allocation of the
        batch-level terms.

        Args:
            z1: Embeddings from first view [batch_size, feature_dim]
            z2: Embeddings from second view [batch_size, feature_dim]

        Returns:
            Per-sample loss approximation [batch_size]
        """
        if z1.shape != z2.shape:
            raise ValueError(f"Shape mismatch: z1 {z1.shape} vs z2 {z2.shape}")
        if z1.dim() != 2:
            raise ValueError(f"Expected 2D tensors [batch, features], got shape {z1.shape}")

        # Per-sample invariance (MSE per sample across feature dims)
        per_sample_inv = F.mse_loss(z1, z2, reduction='none').mean(dim=1)  # [B]

        # Batch-level terms distributed uniformly
        var_loss = self._variance_loss(z1) + self._variance_loss(z2)
        cov_loss = self._covariance_loss(z1) + self._covariance_loss(z2)
        batch_term = (self.lambda_var * var_loss + self.lambda_cov * cov_loss).detach()

        return self.lambda_inv * per_sample_inv + batch_term

    def __repr__(self):
        return (
            f"VICRegLoss(lambda_var={self.lambda_var}, lambda_inv={self.lambda_inv}, "
            f"lambda_cov={self.lambda_cov}, epsilon={self.epsilon})"
        )
