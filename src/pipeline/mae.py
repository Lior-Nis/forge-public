"""MAE Pipeline for masked autoencoder pretraining."""

import gc
import logging
from typing import Dict, List, Literal, Optional, Tuple

import torch

from managers.logging.mae import MAELoggingManager
from managers.metrics.mae import MAEMetricsManager
from managers.weight import WeightManager
from pipeline.base import BasePipeline
from pipeline.config import Config
from pipeline.schemas import MAEBatchLogData

logger = logging.getLogger(__name__)


class MAEPipeline(BasePipeline):
    """
    Pipeline for Masked Autoencoder (MAE) pretraining.

    Handles:
    - Masked autoencoder reconstruction
    - MSE/MAE/SSIM loss computation via loss module
    - Reconstruction quality metrics
    - Masking strategy management
    """

    def __init__(self, config: Config):
        """Initialize MAE pipeline."""
        super().__init__(config)

        if not hasattr(self.backbone, 'patch_size'):
            raise ValueError(
                f"MAE pipeline requires a backbone with 'patch_size' attribute. "
                f"Backbone type {type(self.backbone).__name__} does not support patching. "
                f"Use FogFormer or another patch-based backbone."
            )

        self.mask_ratio = self.model_cfg.mask_ratio
        self.mask_mode = getattr(self.model_cfg, 'mask_mode', 'temporal')
        self.patch_size = self.backbone.patch_size
        self.accumulate_reconstructions = False
        self.should_accumulate_patches = False

    @torch.no_grad()
    def _create_fogformer_temporal_mask(
        self,
        spectral_data: torch.Tensor,
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        masked_patch_indices = torch.stack([
            torch.randperm(num_patches, device=device)[:num_masked_patches]
            for _ in range(B)
        ])

        batch_indices = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
        patch_mask[batch_indices, masked_patch_indices] = True
        timestep_starts = masked_patch_indices * self.patch_size  # [B, num_masked_patches]
        timestep_offsets = torch.arange(self.patch_size, device=device)  # [patch_size]
        timestep_indices = (timestep_starts.unsqueeze(-1) + timestep_offsets).reshape(B, -1)
        temporal_mask = torch.ones(B, W, device=device, dtype=spectral_data.dtype)
        batch_idx_expanded = torch.arange(B, device=device).unsqueeze(1).expand(-1, timestep_indices.shape[1])
        temporal_mask[batch_idx_expanded, timestep_indices] = 0
        masked_data = spectral_data * temporal_mask.view(B, 1, 1, W)
        return masked_data, patch_mask

    @torch.no_grad()
    def _create_causal_temporal_mask(
        self,
        spectral_data: torch.Tensor,
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Causal temporal masking: always predict future patches from past context.

        For each sample a random split point K is drawn. Patches 0..K-1 are visible
        (past context); patches K..nW-1 are masked (future to predict). The random
        split prevents memorising a fixed location and forces generalisation across
        context lengths.

        On average, mask_ratio fraction of patches are masked.

        Args:
            spectral_data: [B, C, H, W]
            mask_ratio: target fraction of patches to mask (future)

        Returns:
            (masked_data [B,C,H,W], patch_mask [B, nW] bool — True = masked/future)
        """
        B, C, H, W = spectral_data.shape
        device = spectral_data.device

        if W % self.patch_size != 0:
            usable_length = (W // self.patch_size) * self.patch_size
            logger.warning(
                f"Sequence length {W} not divisible by patch_size {self.patch_size}. "
                f"Truncating to {usable_length} timesteps."
            )
            spectral_data = spectral_data[..., :usable_length]
            W = usable_length

        num_patches = W // self.patch_size

        # Split range: on average (1-mask_ratio)*nW patches are visible.
        # Allow ±50 % variation around that mean for diversity.
        mean_visible = max(1, int(num_patches * (1 - mask_ratio)))
        lo = max(1, mean_visible // 2)
        hi = min(num_patches - 1, mean_visible + mean_visible // 2)
        if lo >= hi:
            hi = lo + 1  # guarantee at least one future patch

        # Per-sample split points [B]: number of visible (past) patches
        split_points = torch.randint(lo, hi + 1, (B,), device=device)

        # patch_mask[b, t] = True iff patch t is in the future for sample b
        patch_indices = torch.arange(num_patches, device=device).unsqueeze(0)  # [1, nW]
        patch_mask = patch_indices >= split_points.unsqueeze(1)  # [B, nW]

        # Expand patch mask to pixel level [B, W] via repeat_interleave
        pixel_mask = patch_mask.repeat_interleave(self.patch_size, dim=1)  # [B, W]
        temporal_mask = (~pixel_mask).to(spectral_data.dtype)               # 1=visible

        masked_data = spectral_data * temporal_mask.view(B, 1, 1, W)
        return masked_data, patch_mask

    @torch.no_grad()
    def _create_frequency_mask(
        self,
        spectral_data: torch.Tensor,
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask random frequency bands along the H dimension.

        Args:
            spectral_data: [B, C, H, W] spectral representation
            mask_ratio: Fraction of frequency bands to mask

        Returns:
            (masked_data, freq_mask) where freq_mask is [B, H] boolean (True=masked)
        """
        B, C, H, W = spectral_data.shape
        device = spectral_data.device

        num_masked_bands = max(1, int(H * mask_ratio))
        freq_mask = torch.zeros(B, H, dtype=torch.bool, device=device)

        masked_band_indices = torch.stack([
            torch.randperm(H, device=device)[:num_masked_bands]
            for _ in range(B)
        ])

        batch_indices = torch.arange(B, device=device).unsqueeze(1)
        freq_mask[batch_indices, masked_band_indices] = True

        # Zero out masked frequency bands: invert mask so True=masked becomes 0
        visible_mask = (~freq_mask).float().view(B, 1, H, 1)
        masked_data = spectral_data * visible_mask

        return masked_data, freq_mask

    @torch.no_grad()
    def _create_2d_patch_mask(
        self,
        spectral_data: torch.Tensor,
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask random 2D patches in time-frequency space.

        Divides the [H, W] spectrogram into a grid of (freq_patch_size × patch_size)
        blocks and randomly masks mask_ratio of them. This prevents the interpolation
        shortcut: a masked patch cannot be recovered by interpolating along either
        the time or frequency axis independently.

        Args:
            spectral_data: [B, C, H, W] spectral representation
            mask_ratio: Fraction of 2D patches to mask

        Returns:
            (masked_data, patch_mask_2d) where patch_mask_2d is [B, nH, nW] boolean
            (True = masked)
        """
        B, C, H, W = spectral_data.shape
        device = spectral_data.device

        # Use backbone's freq_patch if available (e.g. SpectralPatchEncoder);
        # otherwise fall back to ~8 equal-height bands.
        freq_patch_size = getattr(self.backbone, 'freq_patch', max(1, H // 8))
        time_patch_size = self.patch_size  # = backbone.time_patch for SpectralPatchEncoder

        nH = H // freq_patch_size
        nW = W // time_patch_size
        num_patches = nH * nW
        num_masked = max(1, int(num_patches * mask_ratio))

        # Sample random patch indices per sample
        patch_mask_2d = torch.zeros(B, nH, nW, dtype=torch.bool, device=device)
        flat_indices = torch.stack([
            torch.randperm(num_patches, device=device)[:num_masked]
            for _ in range(B)
        ])  # [B, num_masked]
        freq_idx = flat_indices // nW   # [B, num_masked]
        time_idx = flat_indices % nW    # [B, num_masked]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(freq_idx)
        patch_mask_2d[batch_idx, freq_idx, time_idx] = True

        # Expand patch mask to pixel-level: [B, nH, nW] → [B, H', W']
        pixel_mask = patch_mask_2d \
            .repeat_interleave(freq_patch_size, dim=1) \
            .repeat_interleave(time_patch_size, dim=2)  # [B, nH*fps, nW*tps]

        # Crop to actual H, W (in case H/W not exactly divisible)
        h_crop = min(pixel_mask.shape[1], H)
        w_crop = min(pixel_mask.shape[2], W)
        pixel_mask = pixel_mask[:, :h_crop, :w_crop]

        # Build full spatial mask [B, 1, H, W] (zero = keep, zero-out masked patches)
        visible_mask = (~pixel_mask[:, :h_crop, :w_crop]).float().unsqueeze(1)  # [B,1,H',W']

        # Pad back to original H, W if needed
        if h_crop < H or w_crop < W:
            import torch.nn.functional as F_pad
            visible_mask = F_pad.pad(visible_mask, (0, W - w_crop, 0, H - h_crop), value=1.0)

        masked_data = spectral_data * visible_mask
        return masked_data, patch_mask_2d

    def forward(self, x: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.preprocessors is not None:
            x = self.preprocessors(x)
        if self.training and self.signal_augmentor is not None:
            x = self.signal_augmentor(x)
        spectral = self.transform(x)

        if self.mask_mode == "frequency":
            masked_x, spectral_mask = self._create_frequency_mask(spectral, self.mask_ratio)
        elif self.mask_mode == "2d_patch":
            masked_x, spectral_mask = self._create_2d_patch_mask(spectral, self.mask_ratio)
        elif self.mask_mode == "causal":
            masked_x, spectral_mask = self._create_causal_temporal_mask(spectral, self.mask_ratio)
        else:
            masked_x, spectral_mask = self._create_fogformer_temporal_mask(spectral, self.mask_ratio)

        # Ensure original matches masked_x length (temporal mask may truncate)
        if spectral.shape != masked_x.shape:
            spectral = spectral[..., :masked_x.shape[-1]]

        if self.training and self.spectral_augmentor is not None:
            masked_x = self.spectral_augmentor(masked_x)

        # Causal MAE must prevent each token from attending to future time patches.
        encoded = self.backbone(masked_x, causal=(self.mask_mode == 'causal'))

        # For 2D-patch backbones (SpectralPatchEncoder), provide grid layout so
        # the decoder can reconstruct spatial patches correctly.
        grid_shape = None
        if hasattr(self.backbone, 'freq_patch'):
            fp = self.backbone.freq_patch
            tp = self.backbone.time_patch
            H_sp = masked_x.shape[2]
            W_sp = masked_x.shape[3]
            grid_shape = (H_sp // fp, W_sp // tp)

        x_decoded = self.head(encoded, grid_shape=grid_shape)

        # Align original with decoder output (patch rounding may trim H and/or W)
        if x_decoded.shape != spectral.shape:
            spectral = spectral[:, :, :x_decoded.shape[2], :x_decoded.shape[3]]

        return x_decoded, spectral, spectral_mask

    def on_validation_start(self):
        """Called when validation starts - setup accumulation flags."""
        super().on_validation_start()
        lm = self.logging_manager
        epoch = self.current_epoch
        self.accumulate_reconstructions = (epoch % lm.reconstruction_interval == 0)
        patch_interval = lm.config.train.logging.accumulation_intervals["patches"]
        self.should_accumulate_patches = (epoch % patch_interval == 0)

    def on_test_start(self):
        """Called when test starts - enable all accumulation."""
        super().on_test_start()
        self.accumulate_reconstructions = True

    def _common_step(self, batch, stage: Literal["train", "val", "test"]) -> torch.Tensor:
        """
        Common step logic shared across training, validation, and test steps.

        Args:
            batch: Input batch from dataloader
            stage: Current stage ("train", "val", or "test")

        Returns:
            Mean loss tensor for this batch
        """
        # Parse batch
        x = batch['input']
        metadata = batch['metadata']
        batch_size = x.shape[0]

        reconstructed, original, spectral_mask = self(x)

        per_sample_losses = self.loss.compute_per_sample(
            reconstructed, original, spectral_mask,
            patch_size=self.patch_size if spectral_mask is not None else None,
            mask_mode=self.mask_mode
        )
        mean_loss = per_sample_losses.mean()

        self._log_loss(mean_loss, stage, batch_size)

        batch_data = MAEBatchLogData(
            reconstructed=reconstructed,
            original=original,
            spectral_mask=spectral_mask,
            losses=per_sample_losses,
            metadata=metadata
        )
        self.run_accumulators(stage=stage, data=batch_data)

        return mean_loss

    def training_step(self, batch, batch_idx: int):
        """Training step for MAE."""
        return self._common_step(batch, stage="train")

    def validation_step(self, batch, batch_idx: int):
        """Validation step for MAE."""
        return self._common_step(batch, stage="val")

    def test_step(self, batch, batch_idx: int):
        """Test step for MAE."""
        return self._common_step(batch, stage="test")

    def run_accumulators(self, stage: str, data: MAEBatchLogData) -> None:
        """
        Run conditional accumulation based on epoch intervals.

        Args:
            stage: Current stage ("train", "val", "test")
            data: Batch data context object
        """
        if stage == "train":
            return

        data.detach_cpu()

        if self.should_accumulate_patches:
            self.logging_manager.accumulate_patches(data, stage)
        if self.accumulate_reconstructions:
            self.logging_manager.log_reconstruction_samples(data, stage)

        torch.cuda.empty_cache()

    def _process_epoch_end(self, stage: Literal["train", "val", "test"]) -> None:
        """Process epoch end for MAE."""
        # Log patch performance table
        self.logging_manager.log_patch_performance_table(stage=stage)

        if stage != "train" and self.accumulate_reconstructions:
            self.logging_manager.log_reconstruction_visualizations(stage=stage)

            torch.cuda.empty_cache()
            gc.collect()

        # Clear stage-specific data
        if stage == "val":
            self.logging_manager.clear_validation_data()
        elif stage == "test":
            self.logging_manager.clear_test_data()

        logger.info(f"MAE {stage} epoch completed")

    def _create_logging_manager(self, config: Config, device: str, trainer, datamodule=None):
        return MAELoggingManager(
            config=config,
            device=device,
            trainer=trainer,
            datamodule=datamodule,
        )

    def _create_metrics_manager(self, config: Config, device: str):
        return MAEMetricsManager(device=device)

    def _create_weight_manager(self, config: Config, device: str):
        return WeightManager(
            model=self,
            device=device,
            registry_config=config.train.registry,
            weights_config=config.train.weights
        )
