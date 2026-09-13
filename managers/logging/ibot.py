"""iBOT logging management for self-supervised pretraining."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import numpy as np

from .pretrain_base import PretrainingLoggingManager

if TYPE_CHECKING:
    from pipeline.config import Config
    from pipeline.schemas import IBOTBatchLogData

logger = logging.getLogger(__name__)


class IBOTLoggingManager(PretrainingLoggingManager):
    """Logging manager for iBOT pretraining — logs cosine similarity and embedding statistics."""

    def __init__(
        self,
        config: "Config",
        device: str,
        trainer: Any,
        datamodule: Optional[Any] = None,
    ):
        super().__init__(config, device, trainer, datamodule)
        self._accumulated_student: List[torch.Tensor] = []
        self._accumulated_teacher: List[torch.Tensor] = []
        self._accumulated_masks: List[torch.Tensor] = []
        self._accumulated_losses: List[torch.Tensor] = []

    def _log_task_specific_visualizations(self, stage: str, epoch: int):
        """Log iBOT-specific visualizations at epoch end."""
        self.log_embedding_statistics(stage=stage)

    def accumulate_embeddings(self, data: "IBOTBatchLogData", stage: str) -> None:
        """Accumulate embeddings for epoch-end analysis (limit memory)."""
        if len(self._accumulated_student) >= 10:
            return
        self._accumulated_student.append(data.student_tokens)
        self._accumulated_teacher.append(data.teacher_tokens)
        self._accumulated_masks.append(data.patch_mask)
        self._accumulated_losses.append(data.losses)

    # ------------------------------------------------------------------
    # Diagnostic helpers (shared with JEPA logging pattern)
    # ------------------------------------------------------------------

    def _effective_rank(self, emb_flat: torch.Tensor) -> Optional[float]:
        """Effective rank via singular value entropy: exp(-sum(p*log(p))).

        Healthy iBOT → high rank (many dimensions active).
        Collapse → approaches 1.0 (all variance on one dimension).
        """
        try:
            max_samples = min(512, emb_flat.shape[0])
            s = torch.linalg.svdvals(emb_flat[:max_samples].float())
            s = s[s > 1e-10]
            p = s / s.sum()
            return torch.exp(-(p * torch.log(p)).sum()).item()
        except Exception:
            return None

    def _log_cosine_sim_histogram(
        self, student_flat: torch.Tensor, teacher_flat: torch.Tensor, stage: str
    ) -> None:
        """Cosine similarity histogram between student and teacher embeddings at masked positions.

        Healthy: unimodal peak gradually shifting right toward 1.
        Collapse: spike at exactly 1.0 (trivial solution).
        """
        try:
            import wandb
            cos_sims = torch.nn.functional.cosine_similarity(
                student_flat.float(), teacher_flat.float(), dim=-1
            ).cpu().numpy()
            self.logger.experiment.log(
                {f"ibot/{stage}_cos_sim_hist": wandb.Histogram(cos_sims)},
                commit=False,
            )
        except Exception as e:
            logger.warning(f"Error logging cosine sim histogram: {e}")

    def _log_per_dim_std_chart(self, tgt_flat: torch.Tensor, stage: str) -> None:
        """Per-dimension std of teacher embeddings sorted ascending (log-scale).

        Healthy: relatively flat curve well above 0.
        Collapse: cliff on the left — many dimensions near 0.
        """
        try:
            import matplotlib.pyplot as plt
            import wandb

            per_dim_std = tgt_flat.float().std(dim=0).cpu().numpy()
            sorted_std = np.sort(per_dim_std)
            D = len(sorted_std)
            dead = int((sorted_std < 0.1).sum())

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(D), sorted_std, width=1.0, color="steelblue", alpha=0.8)
            ax.axhline(0.1, color="red", linestyle="--", linewidth=1.5,
                       label=f"collapse threshold (0.1) — {dead}/{D} dims dead")
            ax.set_yscale("log")
            ax.set_xlabel("Dimension index (sorted by std)")
            ax.set_ylabel("Std (log scale)")
            ax.set_title(f"Teacher Embedding Per-Dim Std — {stage.upper()}")
            ax.legend(fontsize=9)
            plt.tight_layout()

            self.logger.experiment.log(
                {f"ibot/{stage}_dim_std_chart": wandb.Image(fig)},
                commit=False,
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Error logging per-dim std chart: {e}")

    def _log_pca_scatter(self, tgt_flat: torch.Tensor, stage: str) -> None:
        """2-component PCA scatter of teacher embeddings."""
        try:
            import matplotlib.pyplot as plt
            import wandb

            max_pts = min(1024, tgt_flat.shape[0])
            emb = tgt_flat[:max_pts].float().cpu().numpy()

            emb_c = emb - emb.mean(axis=0)
            _, _, Vt = np.linalg.svd(emb_c, full_matrices=False)
            proj = emb_c @ Vt[:2].T  # [N, 2]

            fig, ax = plt.subplots(figsize=(6, 6))
            sc = ax.scatter(proj[:, 0], proj[:, 1],
                            c=np.arange(max_pts), cmap="viridis",
                            s=8, alpha=0.6)
            plt.colorbar(sc, ax=ax, label="accumulation index")
            ax.set_title(f"Teacher Embeddings PCA — {stage.upper()}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.tight_layout()

            self.logger.experiment.log(
                {f"ibot/{stage}_pca_scatter": wandb.Image(fig)},
                commit=False,
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Error logging PCA scatter: {e}")

    # ------------------------------------------------------------------
    # Main statistics logger
    # ------------------------------------------------------------------

    def log_embedding_statistics(self, stage: str) -> None:
        """Log aggregate embedding statistics and diagnostic plots to WandB."""
        if not self._accumulated_student or not self._can_log_to_wandb():
            return

        try:
            student = torch.cat(self._accumulated_student, dim=0)  # [B_total, N, D]
            teacher = torch.cat(self._accumulated_teacher, dim=0)
            masks = torch.cat(self._accumulated_masks, dim=0)       # [B_total, N] bool
            losses = torch.cat(self._accumulated_losses, dim=0)

            # Extract only masked positions for diagnostics
            student_masked = student[masks]   # [num_masked_total, D]
            teacher_masked = teacher[masks]

            student_flat = student.reshape(-1, student.shape[-1])
            teacher_flat = teacher.reshape(-1, teacher.shape[-1])

            cos_sims = torch.nn.functional.cosine_similarity(
                student_masked.float(), teacher_masked.float(), dim=-1
            )

            student_min_dim_std = student_flat.float().std(dim=0).min().item()
            teacher_min_dim_std = teacher_flat.float().std(dim=0).min().item()

            eff_rank_student = self._effective_rank(student_flat)
            eff_rank_teacher = self._effective_rank(teacher_flat)

            stats: Dict[str, float] = {
                f"ibot/{stage}_cos_sim_mean": cos_sims.mean().item(),
                f"ibot/{stage}_cos_sim_std": cos_sims.std().item(),
                f"ibot/{stage}_student_std": student.std().item(),
                f"ibot/{stage}_teacher_std": teacher.std().item(),
                f"ibot/{stage}_student_min_dim_std": student_min_dim_std,
                f"ibot/{stage}_teacher_min_dim_std": teacher_min_dim_std,
                f"ibot/{stage}_loss_mean": losses.mean().item(),
                f"ibot/{stage}_loss_std": losses.std().item(),
            }
            if eff_rank_student is not None:
                stats[f"ibot/{stage}_effective_rank_student"] = eff_rank_student
            if eff_rank_teacher is not None:
                stats[f"ibot/{stage}_effective_rank_teacher"] = eff_rank_teacher

            self.logger.experiment.log(stats, commit=False)

            self._log_cosine_sim_histogram(student_masked, teacher_masked, stage)
            self._log_per_dim_std_chart(teacher_flat, stage)
            self._log_pca_scatter(teacher_flat, stage)

        except Exception as e:
            logger.warning(f"Error logging iBOT embedding stats: {e}")

    def clear_validation_data(self):
        """Clear accumulated validation data."""
        super().clear_validation_data()
        self._accumulated_student.clear()
        self._accumulated_teacher.clear()
        self._accumulated_masks.clear()
        self._accumulated_losses.clear()

    def clear_test_data(self):
        """Clear accumulated test data."""
        super().clear_test_data()
        self._accumulated_student.clear()
        self._accumulated_teacher.clear()
        self._accumulated_masks.clear()
        self._accumulated_losses.clear()
