"""LeJEPA loss: prediction + SIGReg (Sketched Isotropic Gaussian Regularization).

Reference: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the
Heuristics" (Balestriero & LeCun, arXiv:2511.08544, 2025).
"""

import math
import logging

import torch
import torch.nn.functional as F

from model.losses.base import BaseLoss

logger = logging.getLogger(__name__)


class LeJEPALoss(BaseLoss):
    """
    Loss for LeJEPA pretraining.

    Combines the JEPA prediction loss (Smooth L1 on L2-normalized embeddings)
    with SIGReg to prevent representation collapse — no EMA target encoder or
    stop-gradient required.

    SIGReg enforces that encoder outputs follow an isotropic Gaussian by testing
    Gaussianity along n_projs random 1D projections via the Epps-Pulley statistic.
    Collapse is impossible: a constant distribution maximally violates the test.
    """

    def __init__(
        self,
        beta: float = 2.0,
        normalize: bool = True,
        sigreg_weight: float = 1.0,
        n_projs: int = 128,
        reduction: str = "mean",
        **kwargs,
    ):
        super().__init__(reduction=reduction, **kwargs)
        self.beta = beta
        self.normalize = normalize
        self.sigreg_weight = sigreg_weight
        self.n_projs = n_projs

    def _sigreg(self, z: torch.Tensor) -> torch.Tensor:
        """
        SIGReg via Epps-Pulley test averaged over random 1D projections.

        Minimum is achieved at an isotropic Gaussian. Collapse to a constant
        vector is the global maximum, so gradient always pushes away from collapse.

        Args:
            z: [B, D] mean-pooled encoder embeddings (cast to float32 internally)

        Returns:
            Scalar loss.
        """
        z = z.float()
        B, D = z.shape

        # Random unit projections [D, K]
        w = F.normalize(
            torch.randn(D, self.n_projs, device=z.device, dtype=z.dtype), dim=0
        )
        s = z @ w  # [B, K]

        # Epps-Pulley statistic per projection:
        #   T_k = mean_{i,j} exp(-0.5*(s_ik-s_jk)^2) - sqrt(2)*mean_i exp(-0.25*s_ik^2) + 1/sqrt(3)
        # The constant 1/sqrt(3) = E[exp(-0.5*(X-Y)^2)] for X,Y~N(0,1) independently,
        # so T_k = 0 exactly at N(0,1) making the optimum interpretable.
        si = s.unsqueeze(1)  # [B, 1, K]
        sj = s.unsqueeze(0)  # [1, B, K]
        cross = torch.exp(-0.5 * (si - sj).pow(2)).mean(dim=(0, 1))  # [K]
        gauss = math.sqrt(2) * torch.exp(-0.25 * s.pow(2)).mean(dim=0)  # [K]
        return (cross - gauss + 1.0 / math.sqrt(3)).mean()

    def _compute_prediction_loss(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Smooth L1 on (optionally L2-normalised) embeddings. Returns [B]."""
        predicted = predicted.float()
        target = target.float()
        if self.normalize:
            predicted = F.normalize(predicted, dim=-1, eps=1e-6)
            target = F.normalize(target, dim=-1, eps=1e-6)
        loss = F.smooth_l1_loss(predicted, target, beta=self.beta, reduction="none")
        return loss.mean(dim=(-1, -2))  # [B]

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        full_emb: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        per_sample = self._compute_prediction_loss(predicted, target)
        pred_loss = self._apply_reduction(per_sample)

        if self.sigreg_weight > 0.0 and full_emb is not None:
            # Mean-pool [B, N, D] → [B, D] before SIGReg for tractable B×B pairwise cost
            z = full_emb.mean(dim=1) if full_emb.dim() == 3 else full_emb
            sigreg_val = self._sigreg(z)
            return pred_loss + self.sigreg_weight * sigreg_val

        return pred_loss

    def compute_per_sample(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Returns prediction loss only [B] — SIGReg is a batch-level regulariser with
        # no per-sample decomposition, so it is excluded here and applied in forward().
        return self._compute_prediction_loss(predicted, target)
