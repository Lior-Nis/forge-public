"""
Hierarchical Normalization Modules for FoG Detection.

This module implements multi-stage normalization following state-of-the-art approaches
for handling bias at different levels:
1. Protocol-level: Hardware/sampling harmonization
2. Patient-level: Inter-patient variability reduction
3. Session-level: Intra-patient session-specific adaptation

References:
- RevIN: "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift" (ICLR 2022)
- FAN: "Frequency Adaptive Normalization for Non-stationary Time Series Forecasting" (NeurIPS 2023)
"""

import logging
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

logger = logging.getLogger(__name__)


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) for time series.

    RevIN normalizes each instance (session) independently and stores the statistics
    to reverse the normalization after prediction. This is crucial for:
    - Handling session-specific distribution shifts
    - Maintaining interpretability of predictions
    - Preserving absolute scale information when needed

    Key features:
    - Per-instance (per-channel) normalization
    - Learnable affine transformation (optional)
    - Reversible for post-processing

    Args:
        num_features: Number of input channels (e.g., 3 for AccV, AccML, AccAP)
        eps: Small constant for numerical stability
        affine: If True, apply learnable affine transformation after normalization
        subtract_last: If True, use last value as reference instead of mean (for non-stationary data)

    Shape:
        - Input: (batch_size, num_features, seq_len)
        - Output: (batch_size, num_features, seq_len)

    Example:
        >>> revin = RevIN(num_features=3, affine=True)
        >>> x = torch.randn(32, 3, 200)  # Batch of 32 sessions
        >>> x_norm = revin(x, mode='norm')
        >>> # ... model processing ...
        >>> x_denorm = revin(x_norm, mode='denorm')
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
        **kwargs
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        # Learnable affine parameters (optional)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        # Buffers to store normalization statistics for reversal
        self.register_buffer('mean', torch.zeros(1, num_features, 1))
        self.register_buffer('stdev', torch.ones(1, num_features, 1))

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        mode: Literal['norm', 'denorm'] = 'norm'
    ) -> torch.Tensor:
        """
        Apply or reverse instance normalization.

        Args:
            x: Input tensor (batch_size, num_features, seq_len)
            mode: 'norm' to normalize, 'denorm' to reverse normalization

        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Mode must be 'norm' or 'denorm', got {mode}")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input using instance statistics."""
        if self.subtract_last:
            # Use last value as reference (good for trend-heavy data)
            self.mean = x[:, :, -1:].detach()
        else:
            # Use mean as reference (standard approach)
            self.mean = x.mean(dim=2, keepdim=True).detach()

        # Compute standard deviation
        x_centered = x - self.mean
        variance = x_centered.pow(2).mean(dim=2, keepdim=True)
        self.stdev = torch.sqrt(variance + self.eps).detach()

        # Normalize
        x_norm = x_centered / self.stdev

        # Apply learnable affine transformation
        if self.affine:
            # Reshape affine parameters for broadcasting
            weight = self.affine_weight.view(1, -1, 1)
            bias = self.affine_bias.view(1, -1, 1)
            x_norm = x_norm * weight + bias

        return x_norm

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the normalization using stored statistics."""
        if self.affine:
            # Reverse affine transformation
            weight = self.affine_weight.view(1, -1, 1)
            bias = self.affine_bias.view(1, -1, 1)
            x = (x - bias) / weight

        # Reverse normalization
        x_denorm = x * self.stdev + self.mean

        return x_denorm


class FAN(nn.Module):
    """
    Frequency Adaptive Normalization (FAN) for non-stationary time series.

    FAN normalizes in the frequency domain, which is particularly effective for
    signals with frequency-varying characteristics like FoG detection where:
    - Locomotion frequencies: 0.5-3 Hz
    - FoG frequencies: 3-8 Hz

    The normalization is frequency-band specific, allowing the model to handle
    different frequency components independently.

    Args:
        num_features: Number of input channels
        freq_bands: List of (low, high) frequency ranges in Hz to normalize separately
        sample_rate: Sampling rate in Hz
        eps: Small constant for numerical stability
        affine: If True, apply learnable affine transformation per frequency band

    Shape:
        - Input: (batch_size, num_features, seq_len)
        - Output: (batch_size, num_features, seq_len)

    Example:
        >>> fan = FAN(num_features=3, freq_bands=[(0.5, 3.0), (3.0, 8.0)], sample_rate=100)
        >>> x = torch.randn(32, 3, 200)
        >>> x_norm = fan(x, mode='norm')
    """

    def __init__(
        self,
        num_features: int,
        freq_bands: list = [(0.5, 3.0), (3.0, 8.0), (8.0, 20.0)],
        sample_rate: float = 100.0,
        eps: float = 1e-5,
        affine: bool = True,
        **kwargs
    ):
        super().__init__()
        self.num_features = num_features
        self.freq_bands = freq_bands
        self.sample_rate = sample_rate
        self.eps = eps
        self.affine = affine
        self.num_bands = len(freq_bands)

        # Learnable affine parameters per frequency band (optional)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features, self.num_bands))
            self.affine_bias = nn.Parameter(torch.zeros(num_features, self.num_bands))

        # Buffers to store normalization statistics
        self.register_buffer('band_means', torch.zeros(1, num_features, self.num_bands))
        self.register_buffer('band_stds', torch.ones(1, num_features, self.num_bands))

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        mode: Literal['norm', 'denorm'] = 'norm'
    ) -> torch.Tensor:
        """
        Apply frequency-adaptive normalization.

        Args:
            x: Input tensor (batch_size, num_features, seq_len)
            mode: 'norm' to normalize, 'denorm' to reverse normalization

        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Mode must be 'norm' or 'denorm', got {mode}")

    def _get_freq_bins(self, seq_len: int) -> Tuple[torch.Tensor, list]:
        """Get frequency bins and masks for each frequency band."""
        # Compute frequency bins
        freqs = torch.fft.rfftfreq(seq_len, d=1.0/self.sample_rate)

        # Create masks for each frequency band
        band_masks = []
        for low, high in self.freq_bands:
            mask = (freqs >= low) & (freqs <= high)
            band_masks.append(mask)

        return freqs, band_masks

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize in frequency domain per band."""
        batch_size, num_features, seq_len = x.shape
        device = x.device

        # Get frequency bins and band masks
        freqs, band_masks = self._get_freq_bins(seq_len)

        # Transform to frequency domain
        x_fft = torch.fft.rfft(x, dim=2)
        x_magnitude = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)

        # Normalize magnitude per frequency band
        self.band_means = torch.zeros(batch_size, num_features, self.num_bands, device=device)
        self.band_stds = torch.ones(batch_size, num_features, self.num_bands, device=device)

        x_magnitude_norm = x_magnitude.clone()

        for band_idx, mask in enumerate(band_masks):
            if mask.sum() > 0:
                # Extract magnitudes for this frequency band
                band_magnitudes = x_magnitude[:, :, mask]

                # Compute band statistics
                band_mean = band_magnitudes.mean(dim=2, keepdim=True)
                band_std = band_magnitudes.std(dim=2, keepdim=True).clamp_min(self.eps)

                # Store statistics
                self.band_means[:, :, band_idx] = band_mean.squeeze(-1)
                self.band_stds[:, :, band_idx] = band_std.squeeze(-1)

                # Normalize band magnitudes
                band_magnitudes_norm = (band_magnitudes - band_mean) / band_std

                # Apply learnable affine transformation
                if self.affine:
                    weight = self.affine_weight[:, band_idx].view(1, -1, 1)
                    bias = self.affine_bias[:, band_idx].view(1, -1, 1)
                    band_magnitudes_norm = band_magnitudes_norm * weight + bias

                # Update normalized magnitudes
                x_magnitude_norm[:, :, mask] = band_magnitudes_norm

        # Reconstruct complex spectrum
        x_fft_norm = x_magnitude_norm * torch.exp(1j * x_phase)

        # Transform back to time domain
        x_norm = torch.fft.irfft(x_fft_norm, n=seq_len, dim=2)

        return x_norm

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse frequency-domain normalization."""
        batch_size, num_features, seq_len = x.shape

        # Get frequency bins and band masks
        freqs, band_masks = self._get_freq_bins(seq_len)

        # Transform to frequency domain
        x_fft = torch.fft.rfft(x, dim=2)
        x_magnitude = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)

        # Denormalize magnitude per frequency band
        x_magnitude_denorm = x_magnitude.clone()

        for band_idx, mask in enumerate(band_masks):
            if mask.sum() > 0:
                # Extract magnitudes for this frequency band
                band_magnitudes = x_magnitude[:, :, mask]

                # Reverse affine transformation
                if self.affine:
                    weight = self.affine_weight[:, band_idx].view(1, -1, 1)
                    bias = self.affine_bias[:, band_idx].view(1, -1, 1)
                    band_magnitudes = (band_magnitudes - bias) / weight

                # Retrieve stored statistics
                band_mean = self.band_means[:, :, band_idx].unsqueeze(-1)
                band_std = self.band_stds[:, :, band_idx].unsqueeze(-1)

                # Denormalize
                band_magnitudes_denorm = band_magnitudes * band_std + band_mean

                # Update denormalized magnitudes
                x_magnitude_denorm[:, :, mask] = band_magnitudes_denorm

        # Reconstruct complex spectrum
        x_fft_denorm = x_magnitude_denorm * torch.exp(1j * x_phase)

        # Transform back to time domain
        x_denorm = torch.fft.irfft(x_fft_denorm, n=seq_len, dim=2)

        return x_denorm


class PatientLevelNormalizer(nn.Module):
    """
    Patient-level normalization using pre-computed statistics from training data.

    This normalizer applies global z-score normalization per channel using statistics
    computed ONLY from the training set to prevent data leakage. These statistics
    represent the population-level distribution and help remove inter-patient variability.

    Key features:
    - Uses training-only statistics (no leakage)
    - Per-channel normalization
    - Supports both z-score and robust (median/MAD) normalization

    Args:
        num_features: Number of input channels
        stats: Dictionary containing 'mean' and 'std' (or 'median' and 'mad') per channel
        method: Normalization method ('zscore' or 'robust')
        eps: Small constant for numerical stability

    Shape:
        - Input: (batch_size, num_features, seq_len)
        - Output: (batch_size, num_features, seq_len)

    Example:
        >>> stats = {'mean': torch.tensor([0.1, 0.2, 0.3]), 'std': torch.tensor([1.0, 1.1, 0.9])}
        >>> normalizer = PatientLevelNormalizer(num_features=3, stats=stats)
        >>> x = torch.randn(32, 3, 200)
        >>> x_norm = normalizer(x)
    """

    def __init__(
        self,
        num_features: int,
        stats: Optional[Dict[str, torch.Tensor]] = None,
        method: Literal['zscore', 'robust'] = 'zscore',
        eps: float = 1e-8,
        **kwargs
    ):
        super().__init__()
        self.num_features = num_features
        self.method = method
        self.eps = eps

        # Register normalization statistics as buffers (not trainable)
        if stats is not None:
            if method == 'zscore':
                if 'mean' not in stats or 'std' not in stats:
                    raise ValueError("Stats dict must contain 'mean' and 'std' for zscore method")
                self.register_buffer('mean', stats['mean'].view(1, -1, 1))
                self.register_buffer('std', stats['std'].view(1, -1, 1))
            elif method == 'robust':
                if 'median' not in stats or 'mad' not in stats:
                    raise ValueError("Stats dict must contain 'median' and 'mad' for robust method")
                self.register_buffer('median', stats['median'].view(1, -1, 1))
                self.register_buffer('mad', stats['mad'].view(1, -1, 1))
        else:
            # Initialize with identity (no normalization)
            if method == 'zscore':
                self.register_buffer('mean', torch.zeros(1, num_features, 1))
                self.register_buffer('std', torch.ones(1, num_features, 1))
            elif method == 'robust':
                self.register_buffer('median', torch.zeros(1, num_features, 1))
                self.register_buffer('mad', torch.ones(1, num_features, 1))

        logger.info(f"Initialized PatientLevelNormalizer with method={method}, stats={'provided' if stats else 'identity'}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply patient-level normalization.

        Args:
            x: Input tensor (batch_size, num_features, seq_len)

        Returns:
            Normalized tensor with same shape
        """
        if self.method == 'zscore':
            x_norm = (x - self.mean) / (self.std + self.eps)
        elif self.method == 'robust':
            x_norm = (x - self.median) / (self.mad + self.eps)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        return x_norm


# Registry for Hydra instantiation
HIERARCHICAL_NORMALIZERS_MAP = {
    "RevIN": RevIN,
    "FAN": FAN,
    "PatientLevelNormalizer": PatientLevelNormalizer,
}
