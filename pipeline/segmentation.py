"""
Segmentation pipeline for per-timestep FOG detection.

Inherits from ClassificationPipeline. The backbone + BiGRU head produces
[B, seq_len, 2] logits via repeat_interleave upsampling from n_tokens predictions.

Token-level loss: per-timestep labels are aggregated to the token block level
(any-fog within each block → block is positive) before computing loss. This
eliminates conflicting gradients caused by FOG boundaries falling mid-block,
where repeat_interleave would give the same logit opposing labels on both sides.

Metrics are computed at token level (one prediction and one label per token),
which is the true temporal resolution of the model.
"""

import logging
from typing import Literal

import torch

from pipeline.classification import ClassificationPipeline
from pipeline.config import Config
from pipeline.schemas import ClassificationBatchLogData

logger = logging.getLogger(__name__)


class SegmentationPipeline(ClassificationPipeline):

    def __init__(self, config: Config):
        super().__init__(config)
        head_cfg = config.model.head
        if not getattr(head_cfg, "sequence_output", False):
            logger.warning(
                "SegmentationPipeline expects sequence_output=True in head config. "
                "Got sequence_output=False — predictions will be patch-level."
            )
        # n_tokens = number of temporal tokens the GRU sees (= backbone output tokens)
        self._n_tokens = config.model.head.input_seq_len

    def _aggregate_to_tokens(
        self,
        y: torch.Tensor,
        valid_masks: torch.Tensor,
    ):
        """
        Aggregate per-timestep labels and masks to token-block level.

        Args:
            y:           [B, seq_len] per-timestep binary labels
            valid_masks: [B, seq_len] or None

        Returns:
            y_token:     [B, n_tokens] — any-fog within each block
            valid_token: [B, n_tokens] — block valid iff all timesteps valid
            block_size:  int — timesteps per token (seq_len // n_tokens)
        """
        B, seq_len = y.shape
        n = self._n_tokens
        block_size = seq_len // n

        # [B, n_tokens, block_size] → any-fog per block
        y_token = (y.view(B, n, block_size) > 0).any(dim=-1).long()

        if valid_masks is not None:
            valid_token = valid_masks.view(B, n, block_size).all(dim=-1)
        else:
            valid_token = torch.ones(B, n, dtype=torch.bool, device=y.device)

        return y_token, valid_token, block_size

    def _common_step(
        self,
        batch,
        stage: Literal["train", "val", "test"],
        compute_metrics: bool = False,
        include_x: bool = False,
    ) -> torch.Tensor:
        x = batch['x']
        y = batch['y']               # [B, seq_len] per-timestep labels
        valid_masks = batch['valid_mask']  # [B, seq_len]
        patches_metadata = batch['metadata']
        batch_size = x.shape[0]

        logits = self(x)             # [B, seq_len, num_classes] after repeat_interleave

        if valid_masks is None:
            valid_masks = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)

        # Aggregate labels to token level to eliminate intra-block gradient noise
        y_token, valid_token, block_size = self._aggregate_to_tokens(y, valid_masks)

        # Repeat token-level labels back to seq_len so CEFocalLoss shape matches logits
        y_for_loss = y_token.repeat_interleave(block_size, dim=1)         # [B, seq_len]
        valid_for_loss = valid_token.repeat_interleave(block_size, dim=1)  # [B, seq_len]

        losses = self.loss(logits, y_for_loss, valid_for_loss)
        mean_loss = losses.mean()
        self._log_loss(mean_loss, stage, batch_size)

        # Metrics at true token resolution: subsample logits + use token-level labels
        probabilities = None
        if compute_metrics:
            prob_token = torch.softmax(logits[:, ::block_size, :], dim=-1)  # [B, n_tokens, 2]
            self.metrics_manager.update_metrics(
                probabilities=prob_token,
                targets=y_token,
                stage=stage,
                valid_masks=valid_token,
            )
            probabilities = prob_token

        batch_data = ClassificationBatchLogData(
            logits=logits,
            labels=y_for_loss,       # store token-level labels (repeated) for consistency
            losses=losses,
            valid_masks=valid_for_loss,
            patches_metadata=patches_metadata,
            probabilities=probabilities,
            x=x if include_x else None,
        )
        self.run_accumulators(stage=stage, data=batch_data)

        return mean_loss

    def run_accumulators(self, stage: str, data: ClassificationBatchLogData) -> None:
        data.detach_cpu()
        # Skip patch-level and spectral accumulation — both assume [B, C] logits.
        # Timestamp accumulation handles [B, T, C] logits natively.
        if self.accumulate_timestamps and stage != "train":
            self.logging_manager.accumulate_timestamps(data, stage)

    def _process_epoch_end(self, stage: Literal["train", "val", "test"]) -> None:
        # Skip all patch/timestamp visualizations — only log metrics
        if stage != "train":
            metrics = self.metrics_manager.compute_metrics(stage=stage)
            self.log_metrics(metrics, stage=stage)
            self._log_epoch_metrics(stage, metrics)
