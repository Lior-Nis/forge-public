"""iBOT loss: cosine self-distillation between student and EMA teacher tokens."""

import logging

import torch
import torch.nn.functional as F

from model.losses.base import BaseLoss

logger = logging.getLogger(__name__)


class IBOTLoss(BaseLoss):
    """
    Loss for iBOT pretraining: cosine similarity between student and EMA teacher
    token embeddings at masked positions.

    Student sees a masked input; teacher (EMA) sees the full input. Loss is
    (1 - cos_sim) averaged over masked tokens, encouraging the student to predict
    the teacher's contextualized representations from partial context.

    Both embeddings are L2-normalized in float32 before similarity computation.
    """

    def __init__(self, reduction: str = "mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def compute_per_sample(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student:    [B, N, D] — student backbone output (from masked input)
            teacher:    [B, N, D] — EMA teacher backbone output (from full input)
            patch_mask: [B, N] bool — True at masked positions

        Returns:
            per_sample_loss: [B] mean cosine loss over masked tokens
        """
        s = F.normalize(student.float(), dim=-1)
        t = F.normalize(teacher.detach().float(), dim=-1)

        cos_sim = (s * t).sum(dim=-1)                   # [B, N]
        loss = (1.0 - cos_sim) * patch_mask.float()     # zero at visible positions

        num_masked = patch_mask.sum(dim=-1).clamp(min=1).float()
        return loss.sum(dim=-1) / num_masked             # [B]

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        patch_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.compute_per_sample(student, teacher, patch_mask).mean()
