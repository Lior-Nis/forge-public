"""JEPA logging management for self-supervised pretraining."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import numpy as np

from .pretrain_base import PretrainingLoggingManager

if TYPE_CHECKING:
    from pipeline.config import Config
    from pipeline.schemas import JEPABatchLogData

logger = logging.getLogger(__name__)


class JEPALoggingManager(PretrainingLoggingManager):
    """Logging manager for JEPA pretraining — logs loss and embedding statistics."""

    def __init__(
        self,
        config: "Config",
        device: str,
        trainer: Any,
        datamodule: Optional[Any] = None,
    ):
        super().__init__(config, device, trainer, datamodule)
        self._accumulated_predicted: List[torch.Tensor] = []
        self._accumulated_target: List[torch.Tensor] = []
        self._accumulated_losses: List[torch.Tensor] = []

    def _log_task_specific_visualizations(self, stage: str, epoch: int):
        """Log JEPA-specific visualizations at epoch end."""
        self.log_embedding_statistics(stage=stage)

    def accumulate_embeddings(self, data: "JEPABatchLogData", stage: str) -> None:
        """Accumulate embeddings for epoch-end analysis (limit memory)."""
        if len(self._accumulated_predicted) >= 10:
            return
        self._accumulated_predicted.append(data.predicted)
        self._accumulated_target.append(data.target)
        self._accumulated_losses.append(data.losses)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def _effective_rank(self, emb_flat: torch.Tensor) -> Optional[float]:
        """Effective rank via singular value entropy: exp(-sum(p*log(p))).

        Healthy I-JEPA → high rank (many dimensions active).
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
        self, pred_flat: torch.Tensor, tgt_flat: torch.Tensor, stage: str
    ) -> None:
        """Cosine similarity histogram between predicted and target embeddings.

        Healthy: unimodal peak gradually shifting right toward 1.
        Collapse: spike at exactly 1.0 (trivial solution).
        """
        try:
            import wandb
            cos_sims = torch.nn.functional.cosine_similarity(
                pred_flat.float(), tgt_flat.float(), dim=-1
            ).cpu().numpy()
            self.logger.experiment.log(
                {f"jepa/{stage}_cos_sim_hist": wandb.Histogram(cos_sims)},
                commit=False,
            )
        except Exception as e:
            logger.warning(f"Error logging cosine sim histogram: {e}")

    def _log_per_dim_std_chart(self, tgt_flat: torch.Tensor, stage: str) -> None:
        """Per-dimension std of target embeddings sorted ascending (log-scale).

        Healthy: relatively flat curve well above 0.
        Collapse: cliff on the left — many dimensions near 0.
        Red dashed line marks the 0.1 collapse threshold.
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
            ax.set_title(f"Target Embedding Per-Dim Std — {stage.upper()}")
            ax.legend(fontsize=9)
            plt.tight_layout()

            self.logger.experiment.log(
                {f"jepa/{stage}_dim_std_chart": wandb.Image(fig)},
                commit=False,
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Error logging per-dim std chart: {e}")

    def _log_pca_scatter(self, tgt_flat: torch.Tensor, stage: str) -> None:
        """2-component PCA scatter of target embeddings.

        Points coloured by their index within the accumulated batch — a proxy
        for temporal position. If the EMA encoder learns temporal dynamics,
        points at nearby positions should cluster together.
        """
        try:
            import matplotlib.pyplot as plt
            import wandb

            max_pts = min(1024, tgt_flat.shape[0])
            emb = tgt_flat[:max_pts].float().cpu().numpy()

            # Manual PCA via SVD (no sklearn dependency)
            emb_c = emb - emb.mean(axis=0)
            _, _, Vt = np.linalg.svd(emb_c, full_matrices=False)
            proj = emb_c @ Vt[:2].T  # [N, 2]

            fig, ax = plt.subplots(figsize=(6, 6))
            sc = ax.scatter(proj[:, 0], proj[:, 1],
                            c=np.arange(max_pts), cmap="viridis",
                            s=8, alpha=0.6)
            plt.colorbar(sc, ax=ax, label="accumulation index")
            ax.set_title(f"Target Embeddings PCA — {stage.upper()}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.tight_layout()

            self.logger.experiment.log(
                {f"jepa/{stage}_pca_scatter": wandb.Image(fig)},
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
        if not self._accumulated_predicted or not self._can_log_to_wandb():
            return

        try:
            predicted = torch.cat(self._accumulated_predicted, dim=0)
            target = torch.cat(self._accumulated_target, dim=0)
            losses = torch.cat(self._accumulated_losses, dim=0)

            pred_flat = predicted.reshape(-1, predicted.shape[-1])
            tgt_flat = target.reshape(-1, target.shape[-1])

            cos_sims = torch.nn.functional.cosine_similarity(
                pred_flat.float(), tgt_flat.float(), dim=-1
            )

            pred_min_dim_std = pred_flat.float().std(dim=0).min().item()
            tgt_min_dim_std = tgt_flat.float().std(dim=0).min().item()

            eff_rank_pred = self._effective_rank(pred_flat)
            eff_rank_tgt = self._effective_rank(tgt_flat)

            stats: Dict[str, float] = {
                f"jepa/{stage}_cos_sim_mean": cos_sims.mean().item(),
                f"jepa/{stage}_cos_sim_std": cos_sims.std().item(),
                f"jepa/{stage}_pred_std": predicted.std().item(),
                f"jepa/{stage}_target_std": target.std().item(),
                f"jepa/{stage}_pred_min_dim_std": pred_min_dim_std,
                f"jepa/{stage}_target_min_dim_std": tgt_min_dim_std,
                f"jepa/{stage}_loss_mean": losses.mean().item(),
                f"jepa/{stage}_loss_std": losses.std().item(),
            }
            if eff_rank_pred is not None:
                stats[f"jepa/{stage}_effective_rank_pred"] = eff_rank_pred
            if eff_rank_tgt is not None:
                stats[f"jepa/{stage}_effective_rank_target"] = eff_rank_tgt

            self.logger.experiment.log(stats, commit=False)

            # Diagnostic plots — every validation epoch
            self._log_cosine_sim_histogram(pred_flat, tgt_flat, stage)
            self._log_per_dim_std_chart(tgt_flat, stage)
            self._log_pca_scatter(tgt_flat, stage)

        except Exception as e:
            logger.warning(f"Error logging JEPA embedding stats: {e}")

    def clear_validation_data(self):
        """Clear accumulated validation data."""
        super().clear_validation_data()
        self._accumulated_predicted.clear()
        self._accumulated_target.clear()
        self._accumulated_losses.clear()

    def clear_test_data(self):
        """Clear accumulated test data."""
        super().clear_test_data()
        self._accumulated_predicted.clear()
        self._accumulated_target.clear()
        self._accumulated_losses.clear()
