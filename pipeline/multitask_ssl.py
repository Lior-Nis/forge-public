"""Multi-Task SSL Pipeline for pretext task pretraining."""

import logging
from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from managers.logging.simclr import SimCLRLoggingManager
from managers.metrics.simclr import SimCLRMetricsManager
from managers.weight import WeightManager
from pipeline.base import BasePipeline
from pipeline.config import Config
from pipeline.schemas import SimCLRBatchLogData

logger = logging.getLogger(__name__)


class MultiTaskSSLPipeline(BasePipeline):
    """
    Multi-task self-supervised pipeline.

    Applies random temporal transformations and trains binary classifiers
    to detect each transformation. Three pretext tasks:
    - Arrow of Time (AoT): detect time-reversed signals
    - Segment Permutation: detect shuffled segments
    - Time Warp: detect non-linearly warped signals

    Reuses SimCLR logging/metrics infrastructure.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        D = self.backbone.output_dim
        self.aot_head = nn.Linear(D, 2)
        self.perm_head = nn.Linear(D, 2)
        self.warp_head = nn.Linear(D, 2)
        self.n_segments = 4
        self.warp_min = 0.8
        self.warp_max = 1.2
        logger.info(
            f"MultiTaskSSL initialized: backbone_dim={D}, "
            f"tasks=[aot, permutation, warp], n_segments={self.n_segments}"
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass: preprocess → transform → backbone → pool."""
        if self.preprocessors is not None:
            x = self.preprocessors(x)
        x = self.transform(x)
        if self.spectral_augmentor is not None:
            x = self.spectral_augmentor(x)
        features = self.backbone(x)
        # Pool to [B, D] if needed
        if features.dim() == 3:
            features = features.mean(dim=1)
        return features

    @torch.no_grad()
    def _apply_pretext_tasks(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply random temporal transformations and generate labels.

        Args:
            x: [B, C, T] raw signal

        Returns:
            (transformed_x, labels_dict) where labels_dict maps task name → [B] long tensor
        """
        B, C, T = x.shape
        device = x.device
        x = x.clone()

        labels = {
            'aot': torch.zeros(B, dtype=torch.long, device=device),
            'perm': torch.zeros(B, dtype=torch.long, device=device),
            'warp': torch.zeros(B, dtype=torch.long, device=device),
        }

        # Arrow of Time: reverse signal (50% chance)
        aot_mask = torch.rand(B, device=device) < 0.5
        if aot_mask.any():
            x[aot_mask] = x[aot_mask].flip(-1)
            labels['aot'][aot_mask] = 1

        # Segment Permutation: shuffle segments (50% chance)
        perm_mask = torch.rand(B, device=device) < 0.5
        if perm_mask.any():
            x[perm_mask] = self._permute_segments(x[perm_mask])
            labels['perm'][perm_mask] = 1

        # Time Warp: non-linear warping (50% chance)
        warp_mask = torch.rand(B, device=device) < 0.5
        if warp_mask.any():
            x[warp_mask] = self._time_warp(x[warp_mask])
            labels['warp'][warp_mask] = 1

        return x, labels

    def _permute_segments(self, x: torch.Tensor) -> torch.Tensor:
        """Shuffle temporal segments. Input: [B_sub, C, T]."""
        B, C, T = x.shape
        seg_len = T // self.n_segments
        usable = seg_len * self.n_segments

        segments = x[..., :usable].reshape(B, C, self.n_segments, seg_len)
        # Random permutation per sample
        for i in range(B):
            perm = torch.randperm(self.n_segments, device=x.device)
            segments[i] = segments[i, :, perm]
        result = segments.reshape(B, C, usable)

        if usable < T:
            result = torch.cat([result, x[..., usable:]], dim=-1)
        return result

    def _time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Power-law time warping. Input: [B_sub, C, T]."""
        B, C, T = x.shape
        device = x.device

        warp_factors = torch.rand(B, device=device) * (self.warp_max - self.warp_min) + self.warp_min
        orig_steps = torch.linspace(1e-6, 1, T, device=device).unsqueeze(0)
        warped = torch.pow(orig_steps, warp_factors.unsqueeze(1))
        warped = warped / warped[:, -1:].clamp(min=1e-8)

        grid_x = warped * 2 - 1
        grid_y = torch.zeros_like(grid_x)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)

        x_4d = x.unsqueeze(2)
        x_out = F.grid_sample(x_4d, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return x_out.squeeze(2)

    def _common_step(
        self,
        batch,
        stage: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        """Apply pretext tasks, forward pass, compute multi-task loss."""
        x_raw = batch['input']
        metadata = batch['metadata']
        batch_size = x_raw.shape[0]

        # Apply pretext transformations (before transform/backbone)
        x_transformed, task_labels = self._apply_pretext_tasks(x_raw)

        # Forward pass: preprocess → transform → backbone → pool
        if self.preprocessors is not None:
            x = self.preprocessors(x_transformed)
        else:
            x = x_transformed

        x = self.transform(x)

        if self.spectral_augmentor is not None:
            x = self.spectral_augmentor(x)

        features = self.backbone(x)
        if features.dim() == 3:
            features = features.mean(dim=1)

        # Task-specific predictions
        task_logits = {
            'aot': self.aot_head(features),
            'perm': self.perm_head(features),
            'warp': self.warp_head(features),
        }

        # Compute loss
        loss = self.loss(task_logits, task_labels)
        per_sample_losses = self.loss.compute_per_sample(task_logits, task_labels)
        self._log_loss(loss, stage, batch_size)

        # Log per-task losses and accuracies
        for name in task_logits:
            task_loss = F.cross_entropy(task_logits[name], task_labels[name])
            self.log(f"losses/{stage}_{name}", task_loss, batch_size=batch_size, logger=True)
            task_acc = (task_logits[name].argmax(dim=1) == task_labels[name]).float().mean()
            self.log(f"metrics/{stage}_{name}_acc", task_acc, batch_size=batch_size, logger=True)

        # Reuse SimCLR batch log data (z1=z2=features)
        batch_data = SimCLRBatchLogData(
            z1=features,
            z2=features,
            losses=per_sample_losses,
            metadata=metadata,
        )
        self.run_accumulators(stage=stage, data=batch_data)

        return loss

    def training_step(self, batch, batch_idx: int):
        return self._common_step(batch, stage="train")

    def validation_step(self, batch, batch_idx: int):
        return self._common_step(batch, stage="val")

    def test_step(self, batch, batch_idx: int):
        return self._common_step(batch, stage="test")

    def run_accumulators(self, stage: str, data: SimCLRBatchLogData) -> None:
        data.detach_cpu()
        self.logging_manager.accumulate_patches(data, stage)

    def _process_epoch_end(self, stage: Literal["train", "val", "test"]) -> None:
        self.logging_manager.log_patch_performance_table(stage=stage)
        if stage == "val":
            self.logging_manager.clear_validation_data()
        elif stage == "test":
            self.logging_manager.clear_test_data()
        logger.info(f"MultiTaskSSL {stage} epoch completed")

    def _create_logging_manager(self, config: Config, device: str, trainer, datamodule=None):
        return SimCLRLoggingManager(
            config=config, device=device, trainer=trainer, datamodule=datamodule
        )

    def _create_metrics_manager(self, config: Config, device: str):
        return SimCLRMetricsManager(device=device)

    def _create_weight_manager(self, config: Config, device: str):
        return WeightManager(
            model=self, device=device,
            registry_config=config.train.registry,
            weights_config=config.train.weights,
        )
