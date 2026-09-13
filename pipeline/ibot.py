"""iBOT Pipeline — Image BERT Pre-Training with Online Tokenizer."""

import copy
import logging
import math
from typing import Literal, Tuple

import torch

from managers.logging.ibot import IBOTLoggingManager
from managers.metrics.ibot import IBOTMetricsManager
from managers.weight import WeightManager
from pipeline.base import BasePipeline
from pipeline.config import Config
from pipeline.schemas import IBOTBatchLogData

logger = logging.getLogger(__name__)


class IBOTPipeline(BasePipeline):
    """
    iBOT (Image BERT Pre-Training with Online Tokenizer) pretraining pipeline.

    Student encoder processes a masked input; EMA teacher processes the full input.
    Loss = (1 − cosine_similarity) between student and teacher token embeddings at
    masked positions. The student learns contextualized representations by attending
    to visible patches and matching the teacher's full-context representations.

    Architecture:
        student_emb = backbone(masked_input)       # [B, nW, D]
        teacher_emb = ema_backbone(full_input)     # [B, nW, D]  (no grad)
        loss = (1 - cos_sim(student[mask], teacher[mask])).mean()

    EMA decay follows a cosine schedule from ema_decay → ema_decay_end over all
    training steps (same schedule as I-JEPA).
    """

    def __init__(self, config: Config):
        super().__init__(config)

        if not hasattr(self.backbone, "patch_size"):
            raise ValueError(
                f"iBOT pipeline requires a backbone with 'patch_size' attribute. "
                f"Backbone type {type(self.backbone).__name__} does not support patching."
            )

        self.patch_size: int = self.backbone.patch_size
        self.mask_ratio: float = self.model_cfg.mask_ratio
        self.mask_mode: str = getattr(self.model_cfg, "mask_mode", "temporal")
        self.ema_decay_init: float = self.model_cfg.ema_decay
        self.ema_decay_end: float = self.model_cfg.ema_decay_end

        self.target_backbone = copy.deepcopy(self.backbone)
        for p in self.target_backbone.parameters():
            p.requires_grad = False

        logger.info(
            f"IBOTPipeline | mask_ratio={self.mask_ratio} mask_mode={self.mask_mode} "
            f"ema_decay {self.ema_decay_init} → {self.ema_decay_end} (cosine)"
        )

    # ------------------------------------------------------------------
    # EMA target encoder (identical to I-JEPA)
    # ------------------------------------------------------------------

    def _get_ema_decay(self) -> float:
        """Cosine schedule: ema_decay_init → ema_decay_end over all training steps."""
        if self.trainer is None:
            return self.ema_decay_init
        total_steps = self.trainer.estimated_stepping_batches
        if total_steps <= 0:
            return self.ema_decay_init
        progress = min(self.global_step / max(total_steps, 1), 1.0)
        return self.ema_decay_end - (self.ema_decay_end - self.ema_decay_init) * (
            math.cos(math.pi * progress) + 1
        ) / 2

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """Update EMA target encoder after each optimizer step."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self._update_target_encoder()

    @torch.no_grad()
    def _update_target_encoder(self):
        decay = self._get_ema_decay()
        for student_p, target_p in zip(
            self.backbone.parameters(), self.target_backbone.parameters()
        ):
            target_p.data.mul_(decay).add_(student_p.data, alpha=1 - decay)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _aggregate_freq_tokens(self, emb: torch.Tensor, backbone) -> torch.Tensor:
        """Mean-pool frequency tokens → temporal tokens for SpectralPatchEncoder.

        SpectralPatchEncoder outputs [B, nH*nW, D] freq-major. iBOT masking
        operates on nW temporal patches so we collapse nH → 1.
        For 1D backbones (no _last_nH) the tensor is returned unchanged.
        """
        if hasattr(backbone, "_last_nH") and backbone._last_nH > 1:
            nH = backbone._last_nH
            nW = backbone._last_nW
            B, N, D = emb.shape
            emb = emb.view(B, nH, nW, D).mean(dim=1)  # [B, nW, D]
        return emb

    @torch.no_grad()
    def _create_temporal_mask(
        self, spectral_data: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random temporal patch mask.

        Args:
            spectral_data: [B, C, H, W]
            mask_ratio: fraction of patches to mask

        Returns:
            masked_data: [B, C, H, W] with masked columns zeroed
            patch_mask: [B, num_patches] bool, True = masked
        """
        B, C, H, W = spectral_data.shape
        device = spectral_data.device
        num_patches = W // self.patch_size

        if W % self.patch_size != 0:
            usable_length = num_patches * self.patch_size
            logger.warning(
                f"Sequence length {W} not divisible by patch_size {self.patch_size}. "
                f"Truncating to {usable_length} timesteps."
            )
            spectral_data = spectral_data[..., :usable_length]
            W = usable_length

        num_masked_patches = int(num_patches * mask_ratio)
        patch_mask = torch.zeros(B, num_patches, dtype=torch.bool, device=device)

        masked_patch_indices = torch.stack(
            [torch.randperm(num_patches, device=device)[:num_masked_patches] for _ in range(B)]
        )
        batch_indices = torch.arange(B, device=device).unsqueeze(1)
        patch_mask[batch_indices, masked_patch_indices] = True

        # Expand patch mask to timestep mask
        timestep_starts = masked_patch_indices * self.patch_size
        timestep_offsets = torch.arange(self.patch_size, device=device)
        timestep_indices = (timestep_starts.unsqueeze(-1) + timestep_offsets).reshape(B, -1)
        temporal_mask = torch.ones(B, W, device=device, dtype=spectral_data.dtype)
        batch_idx_exp = torch.arange(B, device=device).unsqueeze(1).expand(
            -1, timestep_indices.shape[1]
        )
        temporal_mask[batch_idx_exp, timestep_indices] = 0

        return spectral_data * temporal_mask.view(B, 1, 1, W), patch_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            student_emb: [B, nW, D] — student backbone output (from masked input)
            teacher_emb: [B, nW, D] — EMA teacher backbone output (from full input)
            patch_mask:  [B, nW] bool — True at masked positions
        """
        if self.preprocessors is not None:
            x = self.preprocessors(x)
        if self.training and self.signal_augmentor is not None:
            x = self.signal_augmentor(x)
        spectral = self.transform(x)

        # Align to patch boundary
        W = spectral.shape[-1]
        num_patches = W // self.patch_size
        if W != num_patches * self.patch_size:
            spectral = spectral[..., : num_patches * self.patch_size]

        masked_spectral, patch_mask = self._create_temporal_mask(spectral, self.mask_ratio)
        if self.training and self.spectral_augmentor is not None:
            masked_spectral = self.spectral_augmentor(masked_spectral)

        # Student: masked input (gradients flow)
        student_emb = self.backbone(masked_spectral)
        student_emb = self._aggregate_freq_tokens(student_emb, self.backbone)  # [B, nW, D]

        # Teacher: full input, float32, no gradient
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            teacher_emb = self.target_backbone(spectral.float())
            teacher_emb = self._aggregate_freq_tokens(teacher_emb, self.target_backbone)

        return student_emb, teacher_emb, patch_mask

    # ------------------------------------------------------------------
    # Common step
    # ------------------------------------------------------------------

    def _common_step(
        self, batch, stage: Literal["train", "val", "test"]
    ) -> torch.Tensor:
        x = batch["input"]
        metadata = batch["metadata"]
        batch_size = x.shape[0]

        student_emb, teacher_emb, patch_mask = self(x)

        per_sample_losses = self.loss.compute_per_sample(student_emb, teacher_emb, patch_mask)
        mean_loss = self.loss(student_emb, teacher_emb, patch_mask)
        self._log_loss(mean_loss, stage, batch_size)

        if stage == "train":
            self.log("ibot/ema_decay", self._get_ema_decay(), prog_bar=False)

        batch_data = IBOTBatchLogData(
            student_tokens=student_emb,
            teacher_tokens=teacher_emb,
            patch_mask=patch_mask,
            losses=per_sample_losses,
            metadata=metadata,
        )
        self.run_accumulators(stage=stage, data=batch_data)
        return mean_loss

    def training_step(self, batch, batch_idx: int):
        return self._common_step(batch, stage="train")

    def validation_step(self, batch, batch_idx: int):
        return self._common_step(batch, stage="val")

    def test_step(self, batch, batch_idx: int):
        return self._common_step(batch, stage="test")

    # ------------------------------------------------------------------
    # Accumulators & epoch end
    # ------------------------------------------------------------------

    def run_accumulators(self, stage: str, data: IBOTBatchLogData) -> None:
        if stage == "train":
            return
        data.detach_cpu()
        self.logging_manager.accumulate_embeddings(data, stage)

    def _process_epoch_end(self, stage: Literal["train", "val", "test"]) -> None:
        if stage != "train":
            self.logging_manager.log_embedding_statistics(stage=stage)
            if stage == "val":
                self.logging_manager.clear_validation_data()
            elif stage == "test":
                self.logging_manager.clear_test_data()
        logger.info(f"iBOT {stage} epoch completed")

    # ------------------------------------------------------------------
    # Manager factories
    # ------------------------------------------------------------------

    def _create_logging_manager(self, config: Config, device: str, trainer, datamodule=None):
        return IBOTLoggingManager(
            config=config, device=device, trainer=trainer, datamodule=datamodule
        )

    def _create_metrics_manager(self, config: Config, device: str):
        return IBOTMetricsManager(device=device)

    def _create_weight_manager(self, config: Config, device: str):
        return WeightManager(
            model=self,
            device=device,
            registry_config=config.train.registry,
            weights_config=config.train.weights,
        )
