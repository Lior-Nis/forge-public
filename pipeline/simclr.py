"""SimCLR Pipeline for contrastive learning pretraining."""

import logging
from typing import Dict, List, Literal, Tuple

import torch
import torch.nn.functional as F

from managers.logging.simclr import SimCLRLoggingManager
from managers.metrics.simclr import SimCLRMetricsManager
from managers.weight import WeightManager
from pipeline.base import BasePipeline
from pipeline.config import Config
from pipeline.schemas import SimCLRBatchLogData

logger = logging.getLogger(__name__)


class SimCLRPipeline(BasePipeline):
    """
    Pipeline for SimCLR contrastive learning pretraining.

    Handles:
    - Contrastive loss (NT-Xent) via loss module
    - Temperature scaling
    - Positive/negative pair sampling
    - Representation quality metrics
    """

    def __init__(self, config: Config, temperature: float = 0.07):
        """Initialize SimCLR pipeline."""
        super().__init__(config)
        self.temperature = getattr(self.loss, 'temperature', temperature)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for SimCLR representation learning."""
        # Apply preprocessors first (before augmentations)
        if self.preprocessors is not None:
            x = self.preprocessors(x)

        # Apply signal augmentation during training (if not already done in dataset)
        if self.training and self.signal_augmentor is not None:
            x = self.signal_augmentor(x)

        # Transform signal
        x = self.transform(x)

        # Apply spectral augmentation during training
        if self.training and self.spectral_augmentor is not None:
            x = self.spectral_augmentor(x)

        # Backbone feature extraction
        x = self.backbone(x)

        # Projection head for contrastive learning
        x = self.head(x)

        # Note: Normalization happens in loss module for training/validation
        # For inference, normalize here if needed
        if not self.training:
            x = F.normalize(x, dim=-1)

        return x

    def _create_dual_views(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two augmented views for contrastive learning on GPU.

        This method applies the full transformation pipeline twice with independent
        augmentation to create two views of the same input for contrastive learning.

        Args:
            x: Raw input signal [B, C, T]

        Returns:
            Tuple of (x1_transformed, x2_transformed) where each is [B, C', H, W]
            after going through preprocessors, signal augmentation, transform, and
            spectral augmentation.
        """
        # Apply preprocessors to original signal (shared preprocessing)
        if self.preprocessors is not None:
            x = self.preprocessors(x)

        # Create view 1
        x1 = x.clone()
        if self.signal_augmentor is not None:
            x1 = self.signal_augmentor(x1)
        x1 = self.transform(x1)
        if self.spectral_augmentor is not None:
            x1 = self.spectral_augmentor(x1)

        # Create view 2 (independent augmentation)
        x2 = x.clone()
        if self.signal_augmentor is not None:
            x2 = self.signal_augmentor(x2)
        x2 = self.transform(x2)
        if self.spectral_augmentor is not None:
            x2 = self.spectral_augmentor(x2)

        return x1, x2

    def on_validation_start(self):
        """Called when validation starts - setup accumulation flags."""
        super().on_validation_start()
        lm = self.logging_manager
        epoch = self.current_epoch
        # Configure accumulation intervals
        self.accumulate_representations = (epoch % lm.representation_interval == 0)

    def on_test_start(self):
        """Called when test starts - enable all accumulation."""
        super().on_test_start()
        self.accumulate_representations = True

    def _common_step(
        self,
        batch,
        stage: Literal["train", "val", "test"],
        include_x: bool = False,
        log_metrics: bool = False
    ) -> torch.Tensor:
        """
        Common step logic shared across training, validation, and test steps.

        Args:
            batch: Input batch from dataloader
            stage: Current stage ("train", "val", or "test")
            include_x: Whether to include raw input in BatchLogData
            log_metrics: Whether to log detailed contrastive metrics

        Returns:
            Mean loss tensor for this batch
        """
        # Parse batch
        x_raw = batch['input']
        metadata = batch['metadata']
        batch_size = x_raw.shape[0]

        # Create dual views with GPU augmentation
        x1_transformed, x2_transformed = self._create_dual_views(x_raw)

        # Forward through backbone and head for both views
        z1 = self.head(self.backbone(x1_transformed))
        z2 = self.head(self.backbone(x2_transformed))

        # Validate representation shapes
        if z1.shape != z2.shape:
            raise RuntimeError(
                f"Representation shape mismatch: "
                f"z1 {z1.shape} vs z2 {z2.shape}. "
                f"Check backbone output dimensions and projection head configuration."
            )

        if z1.dim() not in (2, 3):
            raise ValueError(
                f"Expected 2D [batch, features] or 3D [batch, patches, features], "
                f"got z1.shape={z1.shape}. Check projection head output."
            )

        # Compute contrastive loss using clean delegation to loss module
        loss = self.loss(z1, z2)
        per_sample_losses = self.loss.compute_per_sample(z1, z2)

        # Pool patch tokens to 2D for logging/metrics (logging managers expect [B, D])
        z1_log = z1.mean(dim=1) if z1.dim() == 3 else z1
        z2_log = z2.mean(dim=1) if z2.dim() == 3 else z2

        # Log loss
        self._log_loss(loss, stage, batch_size)

        # Log detailed metrics if requested
        if log_metrics:
            metrics_dict = self.metrics_manager.compute_batch_contrastive_metrics(
                z1=z1_log,
                z2=z2_log,
                temperature=self.temperature,
                stage=stage,
                batch_size=batch_size
            )
            for metric_name, metric_value in metrics_dict.items():
                self.log(metric_name, metric_value, batch_size=batch_size)

            pos_sim = metrics_dict.get(f'contrastive/{stage}_positive_similarity', 0.0)
            neg_sim = metrics_dict.get(f'contrastive/{stage}_negative_similarity', 0.0)
            separation = pos_sim - neg_sim

            self.logging_manager.log_contrastive_statistics(
                pos_sim=pos_sim,
                neg_sim=neg_sim,
                separation=separation
            )
            self.logging_manager.log_temperature_effect(
                temperature=self.temperature,
                separation_score=separation
            )

            # Patch-level scalar metrics (only when head outputs [B, N, D])
            if z1.dim() == 3:
                alignment, uniformity, eff_rank = self._compute_patch_metrics(z1, z2)
                self.log(f"patch/{stage}_alignment",     alignment,  batch_size=batch_size)
                self.log(f"patch/{stage}_uniformity",    uniformity, batch_size=batch_size)
                self.log(f"patch/{stage}_effective_rank", eff_rank,  batch_size=batch_size)

        # Patch-level data for epoch-end visualizations
        z1_patches = z2_patches = per_position_losses = None
        if z1.dim() == 3:
            z1_patches = z1
            z2_patches = z2
            per_position_losses = self.loss.compute_per_position(z1, z2)

        # Create batch log data and run accumulators (always use 2D pooled for logging)
        batch_data = SimCLRBatchLogData(
            z1=z1_log,
            z2=z2_log,
            losses=per_sample_losses,
            metadata=metadata,
            x_raw=x_raw if include_x else None,
            z1_patches=z1_patches,
            z2_patches=z2_patches,
            per_position_losses=per_position_losses,
        )
        self.run_accumulators(stage=stage, data=batch_data)

        return loss

    def training_step(self, batch, batch_idx: int):
        """Training step for SimCLR."""
        return self._common_step(batch, stage="train", include_x=False, log_metrics=(batch_idx % 100 == 0))

    def validation_step(self, batch, batch_idx: int):
        """Validation step for SimCLR."""
        return self._common_step(batch, stage="val", include_x=True, log_metrics=(batch_idx % 100 == 0))

    def test_step(self, batch, batch_idx: int):
        """Test step for SimCLR."""
        return self._common_step(batch, stage="test", include_x=True, log_metrics=(batch_idx % 100 == 0))

    @torch.no_grad()
    def _compute_patch_metrics(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> tuple:
        """Alignment, uniformity, effective rank for patch-level representations."""
        B, N, D = z1.shape
        z1_n = F.normalize(z1, dim=-1)
        z2_n = F.normalize(z2, dim=-1)

        # Alignment: mean squared distance between positive pairs
        alignment = ((z1_n - z2_n) ** 2).sum(dim=-1).mean()

        # Uniformity: log mean Gaussian kernel; subsample tokens for speed
        z_all = z1_n.reshape(B * N, D)
        n_sub = min(512, z_all.shape[0])
        idx = torch.randperm(z_all.shape[0], device=z1.device)[:n_sub]
        z_sub = z_all[idx]
        sq_dists = torch.pdist(z_sub, p=2).pow(2)
        uniformity = sq_dists.mul(-2).exp().mean().log()

        # Effective rank: exp(entropy of eigenvalue spectrum of token covariance)
        z_c = z_all - z_all.mean(0, keepdim=True)
        n_cov = min(1024, z_c.shape[0])
        z_c = z_c[torch.randperm(z_c.shape[0], device=z1.device)[:n_cov]]
        cov = z_c.T @ z_c / z_c.shape[0]   # [D, D]
        eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
        p = eigvals / eigvals.sum().clamp_min(1e-10) + 1e-10
        effective_rank = (-(p * p.log()).sum()).exp()

        return alignment, uniformity, effective_rank

    def run_accumulators(
        self,
        stage: str,
        data: SimCLRBatchLogData,
    ) -> None:
        """
        Run conditional accumulation based on epoch intervals.

        Args:
            stage: Current stage ("train", "val", "test")
            data: Batch data context object
        """
        # Offload detach/cpu transfer here to avoid repetition in steps
        data.detach_cpu()

        # Always accumulate patches for performance tracking
        self.logging_manager.accumulate_patches(data, stage)

        # Conditional accumulation for validation/test
        if stage != "train" and self.accumulate_representations:
            self.logging_manager.log_representation_samples(data, stage)
            self.logging_manager.accumulate_patch_metrics(data, stage)

    def _process_epoch_end(self, stage: Literal["train", "val", "test"]) -> None:
        """Process epoch end for SimCLR."""
        # Log patch performance table
        self.logging_manager.log_patch_performance_table(stage=stage)

        # For val/test, log additional analysis
        if stage != "train":
            if self.accumulate_representations:
                self.logging_manager.log_representation_visualizations(stage=stage)

        # Clear stage-specific data
        if stage == "val":
            self.logging_manager.clear_validation_data()
        elif stage == "test":
            self.logging_manager.clear_test_data()

        logger.info(f"SimCLR {stage} epoch completed")

    def _create_logging_manager(self, config: Config, device: str, trainer, datamodule=None):
        return SimCLRLoggingManager(
            config=config,
            device=device,
            trainer=trainer,
            datamodule=datamodule,
        )

    def _create_metrics_manager(self, config: Config, device: str):
        return SimCLRMetricsManager(device=device)

    def _create_weight_manager(self, config: Config, device: str):
        return WeightManager(
            model=self,
            device=device,
            registry_config=config.train.registry,
            weights_config=config.train.weights
        )
