"""Adversarial components for domain-adversarial training (DANN).

Gradient reversal layer and patient classifier head for learning
patient-invariant features via adversarial training.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient reversal: identity forward, negate-and-scale backward."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction with a configurable lambda."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class PatientClassifierHead(nn.Module):
    """MLP that classifies patient identity from backbone features.

    Input: [B, N, D] (patch embeddings) → mean-pool → MLP → [B, num_patients]
    """

    def __init__(
        self,
        input_dim: int,
        num_patients: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool over patches
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_patients),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] patch embeddings or [B, D] pooled features

        Returns:
            [B, num_patients] logits
        """
        if x.dim() == 3:
            # [B, N, D] → [B, D, N] → pool → [B, D]
            x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.mlp(x)
