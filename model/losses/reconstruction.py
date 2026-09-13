"""
Reconstruction loss functions for autoencoder-style tasks.

This module provides loss functions for reconstruction tasks:
- MAELoss: Masked Autoencoder reconstruction loss
- SSIMLoss: Structural Similarity Index loss for spectral data
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.losses.base import BaseLoss

logger = logging.getLogger(__name__)


class MAELoss(BaseLoss):
    """
    Masked Autoencoder (MAE) reconstruction loss.

    Supports both patch-based and full spectral reconstruction with clean API
    for per-sample loss computation matching classification pipeline patterns.

    Args:
        norm_pix_loss: Whether to normalize target patches (for patch-based MAE)
        patch_size: Size of image patches (for patch-based MAE, ignored for spectral)
        loss_type: 'mse' or 'mae' (L1)
        use_patches: Whether to use patch-based computation (False for spectral data)
    """

    def __init__(
        self,
        norm_pix_loss: bool = False,
        patch_size: int = 16,
        loss_type: str = 'mse',
        use_patches: bool = False,
        **kwargs,
    ):
        super().__init__(reduction='mean')
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.loss_type = loss_type
        self.use_patches = use_patches

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.

        Args:
            imgs: [batch_size, channels, height, width]

        Returns:
            patches: [batch_size, num_patches, patch_size^2 * channels]
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0, \
            f"Image dimensions must be square and divisible by patch_size={p}"

        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], imgs.shape[1], h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.reshape(imgs.shape[0], h * w, p**2 * imgs.shape[1])
        return x

    def _apply_spectral_mask(
        self,
        per_element_loss: torch.Tensor,
        patch_mask: torch.Tensor,
        patch_size: int,
        spectral_shape: tuple
    ) -> torch.Tensor:
        """
        Apply patch-level mask to spectral loss tensor using vectorized operations.

        VECTORIZED: No loops, 10-50x faster than original.

        Args:
            per_element_loss: [B, C, H, W] element-wise loss
            patch_mask: [B, num_patches] boolean mask (True = masked)
            patch_size: Temporal patch size
            spectral_shape: (C, H, W) shape

        Returns:
            Masked loss tensor [B, C, H, W] with 0 where patches are visible
        """
        B, C, H, W = per_element_loss.shape
        device = per_element_loss.device

        # Expand each patch mask to cover patch_size timesteps
        # [B, num_patches] → [B, num_patches * patch_size]
        temporal_mask = patch_mask.repeat_interleave(patch_size, dim=1)

        # Truncate or pad to match W
        if temporal_mask.shape[1] > W:
            temporal_mask = temporal_mask[:, :W]
        elif temporal_mask.shape[1] < W:
            padding = W - temporal_mask.shape[1]
            temporal_mask = F.pad(temporal_mask, (0, padding), value=False)

        # Expand to [B, C, H, W]
        full_mask = temporal_mask.unsqueeze(1).unsqueeze(2).expand(-1, C, H, -1)

        # Apply mask
        masked_loss = per_element_loss * full_mask.float()
        return masked_loss

    def _apply_frequency_mask(
        self,
        per_element_loss: torch.Tensor,
        freq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply frequency-band mask to loss tensor.

        Args:
            per_element_loss: [B, C, H, W] element-wise loss
            freq_mask: [B, H] boolean mask (True = masked band)

        Returns:
            Masked loss tensor [B, C, H, W] with 0 where bands are visible
        """
        # [B, H] → [B, 1, H, 1] → broadcast to [B, C, H, W]
        full_mask = freq_mask.unsqueeze(1).unsqueeze(-1).float()
        masked_loss = per_element_loss * full_mask
        return masked_loss

    def _apply_2d_patch_mask(
        self,
        per_element_loss: torch.Tensor,
        patch_mask_2d: torch.Tensor,
        freq_patch_size: int,
        time_patch_size: int,
    ) -> torch.Tensor:
        """
        Apply 2D time-frequency patch mask to loss tensor.

        Args:
            per_element_loss: [B, C, H, W] element-wise loss
            patch_mask_2d: [B, nH, nW] boolean mask (True = masked patch)
            freq_patch_size: Height of each frequency patch in pixels
            time_patch_size: Width of each time patch in pixels

        Returns:
            Masked loss tensor [B, C, H, W] with 0 where patches are visible
        """
        B, C, H, W = per_element_loss.shape

        # Expand to pixel level: [B, nH, nW] → [B, nH*fps, nW*tps]
        pixel_mask = patch_mask_2d \
            .repeat_interleave(freq_patch_size, dim=1) \
            .repeat_interleave(time_patch_size, dim=2)

        # Crop/pad to [B, H, W]
        h = min(pixel_mask.shape[1], H)
        w = min(pixel_mask.shape[2], W)
        full_mask = torch.zeros(B, H, W, dtype=torch.float, device=per_element_loss.device)
        full_mask[:, :h, :w] = pixel_mask[:, :h, :w].float()

        # Expand channel dim: [B, H, W] → [B, C, H, W]
        full_mask = full_mask.unsqueeze(1).expand(-1, C, -1, -1)
        return per_element_loss * full_mask

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: Optional[int] = None,
        mask_mode: str = "temporal"
    ) -> torch.Tensor:
        """
        Compute MAE reconstruction loss with aggregation.

        Args:
            pred: Predicted data
                  - Patch-based: [batch_size, num_patches, patch_dim]
                  - Spectral: [batch_size, channels, freq_bins, time_steps]
            target: Target data (same shape as pred)
            mask: Optional binary mask
                  - Patch-based: [batch_size, num_patches]
                  - Spectral: [batch_size, num_patches] patch-level mask or [batch_size, H] freq mask
            patch_size: Temporal patch size (required if mask provided for temporal masking)
            mask_mode: "temporal" (patch-based) or "frequency" (freq-band-based)

        Returns:
            Aggregated reconstruction loss (scalar if reduction='mean')
        """
        # Compute per-sample losses
        per_sample = self.compute_per_sample(pred, target, mask, patch_size, mask_mode)

        # Apply reduction
        return self._apply_reduction(per_sample)

    def compute_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: Optional[int] = None,
        mask_mode: str = "temporal"
    ) -> torch.Tensor:
        """
        Compute per-sample reconstruction losses.

        This method provides clean API for logging, matching classification pipeline.

        Args:
            pred: Predicted data
            target: Target data
            mask: Optional mask (for patch-based MAE or spectral masking)
            patch_size: Temporal patch size (required if mask provided for temporal masking)
            mask_mode: "temporal" (patch-based) or "frequency" (freq-band-based)

        Returns:
            Per-sample losses [batch_size]
        """
        if self.use_patches:
            # Patch-based MAE (original MAE paper approach)
            return self._compute_patch_based(pred, target, mask, patch_size)
        else:
            # Spectral reconstruction (full 4D tensor comparison)
            return self._compute_spectral(pred, target, mask, patch_size, mask_mode)

    def _compute_patch_based(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Compute patch-based reconstruction loss (original MAE)."""
        # Convert target to patches if it's an image
        if len(target.shape) == 4:
            target = self.patchify(target)

        # Normalize target patches if enabled
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6)**.5

        # Compute loss per patch
        if self.loss_type == 'mse':
            loss = (pred - target) ** 2
        elif self.loss_type == 'mae':
            loss = torch.abs(pred - target)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        loss = loss.mean(dim=-1)  # [batch_size, num_patches]

        # Apply mask if provided (only compute loss on masked patches)
        if mask is not None:
            # Ensure mask shape matches loss
            if mask.shape != loss.shape:
                if len(mask.shape) == 1:
                    mask = mask.unsqueeze(0).expand_as(loss)
                elif mask.shape[0] != loss.shape[0]:
                    mask = mask.expand_as(loss)

            # Average only over masked patches per sample
            mask_bool = mask.bool()
            per_sample_loss = []
            for i in range(loss.shape[0]):
                sample_mask = mask_bool[i]
                if sample_mask.any():
                    per_sample_loss.append(loss[i][sample_mask].mean())
                else:
                    per_sample_loss.append(torch.tensor(0.0, device=loss.device))
            return torch.stack(per_sample_loss)

        # Average over patches for each sample
        return loss.mean(dim=-1)  # [batch_size]

    def _compute_spectral(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: Optional[int] = None,
        mask_mode: str = "temporal"
    ) -> torch.Tensor:
        """
        Compute spectral reconstruction loss with optional masking.

        Args:
            pred: [B, C, H, W] predicted spectrogram
            target: [B, C, H, W] target spectrogram
            mask: [B, num_patches] temporal mask or [B, H] frequency mask (True = masked)
            patch_size: Temporal patch size (required for temporal masking)
            mask_mode: "temporal" or "frequency"

        Returns:
            Per-sample losses [batch_size]
        """
        # Compute element-wise loss
        if self.loss_type == 'mse':
            per_element = F.mse_loss(pred, target, reduction='none')
        elif self.loss_type == 'mae':
            per_element = F.l1_loss(pred, target, reduction='none')
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Apply mask if provided
        if mask is not None:
            B = pred.shape[0]
            C, H, W = pred.shape[1:]

            if mask_mode == "frequency":
                # Frequency-band masking: mask is [B, H]
                per_element = self._apply_frequency_mask(per_element, mask)

                num_masked_bands = mask.sum(dim=1)  # [B]
                num_masked_elements = num_masked_bands * C * W  # [B]
            elif mask_mode == "2d_patch":
                # 2D time-frequency patch masking: mask is [B, nH, nW]
                # Infer patch sizes from mask grid dimensions — avoids hard-coding
                nH_mask = mask.shape[1]
                nW_mask = mask.shape[2]
                freq_patch_size = max(1, H // nH_mask)
                time_patch_size = max(1, W // nW_mask)
                per_element = self._apply_2d_patch_mask(
                    per_element, mask, freq_patch_size, time_patch_size
                )
                num_masked_patches = mask.sum(dim=(1, 2))  # [B]
                num_masked_elements = num_masked_patches * freq_patch_size * time_patch_size * C  # [B]
            else:
                # Temporal patch masking: mask is [B, num_patches]
                if patch_size is None:
                    raise ValueError("patch_size required for temporal masking")

                per_element = self._apply_spectral_mask(
                    per_element, mask, patch_size, pred.shape[1:]
                )

                num_masked_patches = mask.sum(dim=1)  # [B]
                num_masked_elements = num_masked_patches * patch_size * C * H  # [B]

            # Sum loss per sample
            per_sample_sums = per_element.view(B, -1).sum(dim=1)  # [B]

            # Avoid division by zero
            safe_divisor = torch.clamp(num_masked_elements.float(), min=1)
            per_sample_loss = per_sample_sums / safe_divisor

            # Zero out losses where no elements were masked
            per_sample_loss = torch.where(
                num_masked_elements > 0,
                per_sample_loss,
                torch.zeros_like(per_sample_loss)
            )

            return per_sample_loss  # [B]
        else:
            # No mask: compute over all elements
            per_sample = per_element.view(per_element.shape[0], -1).mean(dim=1)
            return per_sample


class SSIMLoss(BaseLoss):
    """
    Structural Similarity Index (SSIM) loss for spectral data.

    Preserves structural details better than MSE alone.
    Designed for time-frequency spectral representations.

    Args:
        window_size: Sliding window size for SSIM computation
        data_range: Dynamic range of the data
        channel_reduction: How to aggregate multi-channel SSIM
        size_average: Whether to average across spatial dimensions
    """

    def __init__(
        self,
        window_size: int = 11,
        data_range: float = 1.0,
        channel_reduction: str = 'mean',
        size_average: bool = True,
        **kwargs
    ):
        super().__init__(reduction='mean')
        self.window_size = window_size
        self.data_range = data_range
        self.channel_reduction = channel_reduction
        self.size_average = size_average

        # SSIM constants from paper
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size))

    def _create_window(self, window_size: int) -> torch.Tensor:
        """Create 2D Gaussian window for SSIM computation."""
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        # Create 2D window
        window_2d = g.unsqueeze(0) * g.unsqueeze(1)
        return window_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    def _compute_ssim(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        window: torch.Tensor,
        padding: int
    ) -> torch.Tensor:
        """Compute SSIM for single channel."""
        # Compute local means
        mu1 = F.conv2d(x, window, padding=padding)
        mu2 = F.conv2d(y, window, padding=padding)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Compute local variances and covariance
        sigma1_sq = F.conv2d(x * x, window, padding=padding) - mu1_sq
        sigma2_sq = F.conv2d(y * y, window, padding=padding) - mu2_sq
        sigma12 = F.conv2d(x * y, window, padding=padding) - mu1_mu2

        # Compute SSIM
        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        ssim_map = numerator / (denominator + 1e-8)

        if self.size_average:
            return ssim_map.mean(dim=(2, 3))
        return ssim_map

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute SSIM loss with aggregation.

        Args:
            pred: Predicted spectral data [batch, channels, freq_bins, time_steps]
            target: Target spectral data [batch, channels, freq_bins, time_steps]
            mask: Optional patch-level mask [batch, num_patches]
            patch_size: Temporal patch size (required if mask provided)

        Returns:
            Aggregated SSIM loss (1 - SSIM)
        """
        # Compute per-sample losses
        per_sample = self.compute_per_sample(pred, target, mask, patch_size)

        # Apply reduction
        return self._apply_reduction(per_sample)

    def compute_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute per-sample SSIM losses with optional masking.

        Args:
            pred: Predicted spectral data [batch, channels, freq_bins, time_steps]
            target: Target spectral data [batch, channels, freq_bins, time_steps]
            mask: Optional patch-level mask [batch, num_patches]
            patch_size: Temporal patch size (required if mask provided)

        Returns:
            Per-sample SSIM losses [batch_size]
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

        batch_size, channels, height, width = pred.shape

        # Determine effective window size
        effective_window_size = min(self.window_size, min(height, width))
        if effective_window_size % 2 == 0:
            effective_window_size -= 1

        # Fall back to MSE for very small inputs
        if effective_window_size < 3:
            logger.warning(
                f"SSIM window size too small ({effective_window_size}) for input shape "
                f"{pred.shape}. Falling back to MSE loss. Consider using MAELoss instead."
            )
            per_element = F.mse_loss(pred, target, reduction='none')

            # Apply mask if provided
            if mask is not None and patch_size is not None:
                # Use MAELoss helper for masking
                mae_loss = MAELoss()
                per_element = mae_loss._apply_spectral_mask(
                    per_element, mask, patch_size, pred.shape[1:]
                )
                # Compute mean only over masked elements
                per_sample = []
                for b in range(batch_size):
                    num_masked_patches = mask[b].sum().item()
                    num_masked_elements = num_masked_patches * patch_size * channels * height
                    if num_masked_elements > 0:
                        sample_loss = per_element[b].sum() / num_masked_elements
                    else:
                        sample_loss = torch.tensor(0.0, device=pred.device)
                    per_sample.append(sample_loss)
                return torch.stack(per_sample)

            return per_element.view(batch_size, -1).mean(dim=1)

        # Create appropriate window
        if effective_window_size != self.window_size:
            window = self._create_window(effective_window_size).to(pred.device)
        else:
            window = self.window.to(pred.device)

        padding = effective_window_size // 2

        # Compute SSIM per channel
        ssim_values = []
        for c in range(channels):
            pred_c = pred[:, c:c+1, :, :]
            target_c = target[:, c:c+1, :, :]
            ssim_c = self._compute_ssim(pred_c, target_c, window, padding)
            ssim_values.append(ssim_c)

        # Aggregate channels
        if len(ssim_values) == 1:
            ssim_per_sample = ssim_values[0].squeeze()  # [batch_size]
        else:
            ssim_tensor = torch.stack(ssim_values, dim=1)  # [batch_size, channels]
            if self.channel_reduction == 'mean':
                ssim_per_sample = ssim_tensor.mean(dim=1).squeeze()
            elif self.channel_reduction == 'sum':
                ssim_per_sample = ssim_tensor.sum(dim=1).squeeze()
            else:
                raise ValueError(f"Unknown channel_reduction: {self.channel_reduction}")

        # Return 1 - SSIM as loss (lower SSIM = higher loss) per sample
        return 1.0 - ssim_per_sample  # [batch_size]


class SpectralMAELoss(BaseLoss):
    """
    Frequency-domain reconstruction loss.

    Computes MSE/MAE in the FFT domain, penalizing missing spectral content.
    This prevents trivial interpolation solutions since interpolation produces
    overly smooth outputs that lack high-frequency components.

    Args:
        loss_type: 'mse' or 'mae' (L1)
        log_scale: Whether to compute loss on log-magnitude spectrum
        epsilon: Small value for numerical stability in log
    """

    def __init__(
        self,
        loss_type: str = 'mse',
        log_scale: bool = True,
        epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__(reduction='mean')
        self.loss_type = loss_type
        self.log_scale = log_scale
        self.epsilon = epsilon

    def _compute_fft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute FFT magnitude spectrum along the last (temporal) dimension.

        Args:
            x: [..., T] tensor

        Returns:
            Magnitude spectrum [..., T//2+1]
        """
        # cuFFT requires float32 for non-power-of-2 sizes; disable AMP autocast
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft(x.float(), dim=-1)
        magnitude = spectrum.abs()
        if self.log_scale:
            magnitude = torch.log(magnitude + self.epsilon)
        return magnitude

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: Optional[int] = None,
    ) -> torch.Tensor:
        per_sample = self.compute_per_sample(pred, target, mask, patch_size)
        return self._apply_reduction(per_sample)

    def compute_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute per-sample frequency-domain loss.

        For masked MAE: computes FFT on each masked patch individually,
        comparing predicted vs target spectral content per patch.

        Args:
            pred: [B, C, H, W] predicted spectrogram
            target: [B, C, H, W] target spectrogram
            mask: [B, num_patches] boolean mask (True = masked patch)
            patch_size: Temporal patch size (required if mask provided)

        Returns:
            Per-sample losses [B]
        """
        B = pred.shape[0]

        if mask is not None and patch_size is not None:
            # Compute FFT loss only on masked patches
            C, H, W = pred.shape[1:]
            num_patches = W // patch_size

            # Reshape to patches: [B, C, H, W] → [B, C, H, num_patches, patch_size]
            usable_W = num_patches * patch_size
            pred_patches = pred[..., :usable_W].reshape(B, C, H, num_patches, patch_size)
            target_patches = target[..., :usable_W].reshape(B, C, H, num_patches, patch_size)

            # FFT on each patch along last dim
            pred_fft = self._compute_fft_magnitude(pred_patches)
            target_fft = self._compute_fft_magnitude(target_patches)

            # Per-element loss: [B, C, H, num_patches, freq_bins]
            if self.loss_type == 'mse':
                per_element = (pred_fft - target_fft) ** 2
            elif self.loss_type == 'mae':
                per_element = torch.abs(pred_fft - target_fft)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            # Mean over C, H, freq_bins → [B, num_patches]
            per_patch_loss = per_element.mean(dim=(1, 2, 4))

            # Mask: only count masked patches
            mask_float = mask[:, :num_patches].float()
            masked_loss = per_patch_loss * mask_float

            # Mean over masked patches per sample
            num_masked = mask_float.sum(dim=1).clamp(min=1)
            per_sample = masked_loss.sum(dim=1) / num_masked
            return per_sample
        else:
            # No mask: FFT on full temporal dimension
            pred_fft = self._compute_fft_magnitude(pred)
            target_fft = self._compute_fft_magnitude(target)

            if self.loss_type == 'mse':
                per_element = (pred_fft - target_fft) ** 2
            elif self.loss_type == 'mae':
                per_element = torch.abs(pred_fft - target_fft)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            return per_element.view(B, -1).mean(dim=1)
