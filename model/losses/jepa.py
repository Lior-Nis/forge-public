"""JEPA loss for latent prediction in joint embedding predictive architecture."""

import logging

import torch
import torch.nn.functional as F

from model.losses.base import BaseLoss

logger = logging.getLogger(__name__)


class JEPALoss(BaseLoss):
    """
    Loss for JEPA pretraining: Smooth L1 between predicted and target embeddings,
    plus VICReg-style variance regularization to prevent representation collapse.

    Both predicted and target embeddings are L2-normalized before computing the
    prediction loss. The variance term explicitly penalizes collapsed dimensions,
    preventing the trivial fixed point where constant embeddings minimize Smooth L1.

    All computation is done in float32 to avoid NaN from F.normalize in float16.
    """

    def __init__(
        self,
        beta: float = 2.0,
        normalize: bool = True,
        reduction: str = "mean",
        var_weight: float = 25.0,
        **kwargs,
    ):
        super().__init__(reduction=reduction, **kwargs)
        self.beta = beta
        self.normalize = normalize
        self.var_weight = var_weight

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Hinge loss pushing per-dim std toward 1. z: [B, N, D] -> scalar."""
        z = z.reshape(-1, z.shape[-1]).float()  # [B*N, D]
        std = torch.sqrt(z.var(dim=0) + 1e-4)   # [D]
        return F.relu(1.0 - std).mean()

    def _compute_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Shared loss computation in float32. Returns per-sample losses [B]."""
        # Upcast to float32 for numerical stability (AMP can pass float16)
        predicted = predicted.float()
        target = target.float()

        if self.normalize:
            predicted = F.normalize(predicted, dim=-1, eps=1e-6)
            target = F.normalize(target, dim=-1, eps=1e-6)

        loss = F.smooth_l1_loss(predicted, target, beta=self.beta, reduction="none")
        return loss.mean(dim=(-1, -2))  # [B]

    def forward(self, predicted: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        per_sample = self._compute_loss(predicted, target)
        base = self._apply_reduction(per_sample)
        if self.var_weight > 0:
            var_reg = self._variance_loss(predicted) + self._variance_loss(target)
            return base + self.var_weight * var_reg
        return base

    def compute_per_sample(self, predicted: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._compute_loss(predicted, target)
