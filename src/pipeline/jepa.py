"""JEPA Pipeline — supports both I-JEPA (EMA target encoder) and LeJEPA (SIGReg)."""

import copy
import logging
import math
from typing import Literal, Optional, Tuple

import torch

from managers.logging.jepa import JEPALoggingManager
from managers.metrics.jepa import JEPAMetricsManager
from managers.weight import WeightManager
from pipeline.base import BasePipeline
from pipeline.config import Config
from pipeline.schemas import JEPABatchLogData

logger = logging.getLogger(__name__)


class JEPAPipeline(BasePipeline):
    """
    Unified pipeline for JEPA pretraining.

    Supports two modes via ``model.jepa_mode``:

    **ijepa** (default) — I-JEPA (Assran et al., 2023):
      Student encoder processes masked input; predictor predicts embeddings of
      masked patches as produced by an EMA copy of the student (target encoder).
      Collapse prevented by the EMA asymmetry.

    **lejepa** — LeJEPA (Balestriero & LeCun, arXiv:2511.08544, 2025):
      Single encoder processes both full and masked input; predictor predicts
      masked-position embeddings from the full-input pass. Collapse prevented by
      SIGReg (Sketched Isotropic Gaussian Regularization) in the loss.
      No EMA, no stop-gradient, no momentum schedules.
    """

    def __init__(self, config: Config):
        super().__init__(config)

        if not hasattr(self.backbone, "patch_size"):
            raise ValueError(
                f"JEPA pipeline requires a backbone with 'patch_size' attribute. "
                f"Backbone type {type(self.backbone).__name__} does not support patching."
            )

        self.jepa_mode: str = getattr(self.model_cfg, "jepa_mode", "ijepa")
        self.mask_ratio: float = self.model_cfg.mask_ratio
        self.patch_size: int = self.backbone.patch_size

        if self.jepa_mode == "ijepa":
            self.ema_decay_init: float = self.model_cfg.ema_decay
            self.ema_decay_end: float = self.model_cfg.ema_decay_end
            self.target_backbone = copy.deepcopy(self.backbone)
            for p in self.target_backbone.parameters():
                p.requires_grad = False
            logger.info(
                f"JEPAPipeline mode=ijepa | ema_decay {self.ema_decay_init} → "
                f"{self.ema_decay_end} (step-based cosine)"
            )
        else:
            logger.info("JEPAPipeline mode=lejepa | no EMA — collapse prevention via SIGReg")

        self._validate_mode_loss_compatibility()

    def _validate_mode_loss_compatibility(self):
        from model.losses.lejepa import LeJEPALoss
        if self.jepa_mode == "lejepa" and not isinstance(self.loss, LeJEPALoss):
            raise ValueError(
                f"jepa_mode='lejepa' requires LeJEPALoss but got "
                f"{type(self.loss).__name__}. "
                f"Set train/loss=lejepa in your experiment config."
            )
        if self.jepa_mode == "ijepa" and isinstance(self.loss, LeJEPALoss):
            logger.warning(
                "jepa_mode='ijepa' is paired with LeJEPALoss — SIGReg will run "
                "but full_emb will be None (no-op). Use train/loss=jepa instead."
            )

    # ------------------------------------------------------------------
    # I-JEPA: EMA target encoder update
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
        """After optimizer step, update EMA target encoder (I-JEPA only)."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.jepa_mode == "ijepa":
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

        SpectralPatchEncoder outputs [B, nH*nW, D] freq-major. JEPA masking
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
    # Forward (dispatches to mode-specific implementation)
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            predicted:     [B, num_masked, D]
            target_masked: [B, num_masked, D]
            patch_mask:    [B, num_patches] bool
            full_emb:      [B, N, D] full encoder output (LeJEPA only, else None)
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

        if self.jepa_mode == "ijepa":
            return self._forward_ijepa(spectral)
        return self._forward_lejepa(spectral)

    def _forward_ijepa(
        self, spectral: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        # Target encoder: full input, float32 (avoid AMP overflow in frozen BN etc.)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            target_emb = self.target_backbone(spectral.float())
            target_emb = self._aggregate_freq_tokens(target_emb, self.target_backbone)

        # Student: masked input
        masked_spectral, patch_mask = self._create_temporal_mask(spectral, self.mask_ratio)
        if self.training and self.spectral_augmentor is not None:
            masked_spectral = self.spectral_augmentor(masked_spectral)

        student_emb = self.backbone(masked_spectral)
        student_emb = self._aggregate_freq_tokens(student_emb, self.backbone)

        B = student_emb.shape[0]
        visible_mask = ~patch_mask
        num_visible = visible_mask.sum(dim=1)[0].int().item()
        visible_emb = student_emb[visible_mask].reshape(B, num_visible, -1)
        predicted = self.head(visible_emb, patch_mask)

        num_masked = patch_mask.sum(dim=1)[0].int().item()
        target_masked = target_emb[patch_mask].reshape(B, num_masked, -1)

        return predicted, target_masked, patch_mask, None

    def _forward_lejepa(
        self, spectral: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Full encoder pass — gradients flow through (no stop-gradient per LeJEPA design)
        full_emb = self.backbone(spectral)
        full_emb = self._aggregate_freq_tokens(full_emb, self.backbone)  # [B, N, D]

        # Student / context pass — same backbone, masked input
        masked_spectral, patch_mask = self._create_temporal_mask(spectral, self.mask_ratio)
        if self.training and self.spectral_augmentor is not None:
            masked_spectral = self.spectral_augmentor(masked_spectral)

        context_emb = self.backbone(masked_spectral)
        context_emb = self._aggregate_freq_tokens(context_emb, self.backbone)

        B = context_emb.shape[0]
        visible_mask = ~patch_mask
        num_visible = visible_mask.sum(dim=1)[0].int().item()
        visible_emb = context_emb[visible_mask].reshape(B, num_visible, -1)
        predicted = self.head(visible_emb, patch_mask)

        # Target: full-input encoder output at masked positions
        num_masked = patch_mask.sum(dim=1)[0].int().item()
        target_masked = full_emb[patch_mask].reshape(B, num_masked, -1)

        return predicted, target_masked, patch_mask, full_emb

    # ------------------------------------------------------------------
    # Common step
    # ------------------------------------------------------------------

    def _common_step(
        self, batch, stage: Literal["train", "val", "test"]
    ) -> torch.Tensor:
        x = batch["input"]
        metadata = batch["metadata"]
        batch_size = x.shape[0]

        predicted, target_masked, patch_mask, full_emb = self(x)

        per_sample_losses = self.loss.compute_per_sample(predicted, target_masked)

        # Full loss (includes SIGReg for LeJEPA when loss supports it)
        mean_loss = self.loss(predicted, target_masked, full_emb=full_emb)
        self._log_loss(mean_loss, stage, batch_size)

        if stage == "train":
            if self.jepa_mode == "ijepa":
                self.log("jepa/ema_decay", self._get_ema_decay(), prog_bar=False)
            else:
                self.log("jepa/mode", 0.0, prog_bar=False)  # sentinel so WandB shows mode

        batch_data = JEPABatchLogData(
            predicted=predicted,
            target=target_masked,
            losses=per_sample_losses,
            metadata=metadata,
            patch_mask=patch_mask,
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

    def run_accumulators(self, stage: str, data: JEPABatchLogData) -> None:
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
        logger.info(f"JEPA[{self.jepa_mode}] {stage} epoch completed")

    # ------------------------------------------------------------------
    # Manager factories
    # ------------------------------------------------------------------

    def _create_logging_manager(self, config: Config, device: str, trainer, datamodule=None):
        return JEPALoggingManager(
            config=config, device=device, trainer=trainer, datamodule=datamodule
        )

    def _create_metrics_manager(self, config: Config, device: str):
        return JEPAMetricsManager(device=device)

    def _create_weight_manager(self, config: Config, device: str):
        return WeightManager(
            model=self,
            device=device,
            registry_config=config.train.registry,
            weights_config=config.train.weights,
        )
