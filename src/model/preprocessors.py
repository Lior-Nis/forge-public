"""
Signal preprocessing modules for FoG detection.

This module provides modular preprocessing stages that are applied before
augmentations in the pipeline. Preprocessors handle signal conditioning
tasks like detrending, filtering, normalization, and baseline correction.

The architecture mirrors the augmentors system with:
- Individual preprocessor classes following the same interface
- SignalPreprocessor chaining class for sequential application
- Factory pattern for Hydra configuration integration
- GPU/AMP compatibility with @torch.no_grad() decorators

Execution Order:
    preprocessors → augmentations → transform → backbone → head

Example Usage:
    ```python
    from model.preprocessors import SignalPreprocessor

    # Single preprocessor
    detrend = DetrendPreprocessor(method='linear')
    x_processed = detrend(x)

    # Chained preprocessors (typical usage)
    preprocessors = SignalPreprocessor([
        DetrendPreprocessor(method='linear'),
        BandpassFilterPreprocessor(low_freq=0.5, high_freq=20.0),
        NormalizePreprocessor(method='zscore')
    ])
    x_processed = preprocessors(x)
    ```
"""

import logging
import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as audio_F
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


class DetrendPreprocessor(nn.Module):
    """Remove trends from accelerometer signals to improve stationarity."""

    def __init__(
        self,
        method: Literal['linear', 'constant', 'polynomial'] = 'linear',
        poly_order: int = 2,
        **kwargs
    ):
        super().__init__()
        self.method = method
        self.poly_order = poly_order

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove trends from input signals.

        Args:
            x: Tensor of shape [batch_size, channels, seq_len]
        Returns:
            Detrended tensor of same shape
        """
        batch_size, channels, seq_len = x.shape
        device = x.device

        # Process on CPU for scipy compatibility, then return to original device
        x_cpu = x.cpu().numpy()
        x_detrended = np.zeros_like(x_cpu)

        for b in range(batch_size):
            for c in range(channels):
                signal = x_cpu[b, c, :]

                if self.method == 'linear':
                    # Remove linear trend
                    x_detrended[b, c, :] = scipy_signal.detrend(signal, type='linear')
                elif self.method == 'constant':
                    # Remove DC component (mean)
                    x_detrended[b, c, :] = scipy_signal.detrend(signal, type='constant')
                elif self.method == 'polynomial':
                    # Remove polynomial trend
                    time_points = np.arange(seq_len)
                    poly_coeffs = np.polyfit(time_points, signal, self.poly_order)
                    trend = np.polyval(poly_coeffs, time_points)
                    x_detrended[b, c, :] = signal - trend
                else:
                    # No detrending, pass through
                    x_detrended[b, c, :] = signal

        return torch.from_numpy(x_detrended).float().to(device)


class LowpassFilterPreprocessor(nn.Module):
    """Apply Butterworth lowpass filter to remove high-frequency noise (GPU-accelerated)."""

    def __init__(
        self,
        cutoff_freq: float = 20.0,
        order: int = 4,
        sample_rate: int = 100,
        **kwargs
    ):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.order = order
        self.sample_rate = sample_rate

        # Pre-compute filter coefficients using scipy (one-time CPU operation)
        sos = scipy_signal.butter(
            self.order,
            self.cutoff_freq,
            btype='low',
            fs=self.sample_rate,
            output='sos'
        )

        # Convert SOS to transfer function (ba) format for torchaudio
        b, a = scipy_signal.sos2tf(sos)

        # Store as buffers (will move to GPU with model)
        self.register_buffer('a_coeffs', torch.tensor(a, dtype=torch.float32))
        self.register_buffer('b_coeffs', torch.tensor(b, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply lowpass filter to input signals (GPU-native, batched).

        Args:
            x: Tensor of shape [batch_size, channels, seq_len]
        Returns:
            Filtered tensor of same shape
        """
        batch_size, channels, seq_len = x.shape
        original_device = x.device
        original_dtype = x.dtype

        # Reshape to [batch_size * channels, seq_len] for torchaudio
        x_reshaped = x.reshape(batch_size * channels, seq_len)

        try:
            # filtfilt requires float32; disable autocast to prevent AMP recasting
            with torch.amp.autocast(device_type=original_device.type, enabled=False):
                x_f32 = x_reshaped.float()

                if x.device.type == 'mps':
                    x_cpu = x_f32.cpu()
                    x_filtered = audio_F.filtfilt(x_cpu, self.a_coeffs.cpu(), self.b_coeffs.cpu(), clamp=False)
                    x_filtered = x_filtered.to(original_device)
                else:
                    x_filtered = audio_F.filtfilt(x_f32, self.a_coeffs, self.b_coeffs, clamp=False)

            return x_filtered.to(original_dtype).reshape(batch_size, channels, seq_len)
        except Exception as e:
            warnings.warn(f"Lowpass filtering failed: {e}, using original signal")
            return x


class HighpassFilterPreprocessor(nn.Module):
    """Apply Butterworth highpass filter to remove low-frequency drift (GPU-accelerated)."""

    def __init__(
        self,
        cutoff_freq: float = 0.5,
        order: int = 4,
        sample_rate: int = 100,
        **kwargs
    ):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.order = order
        self.sample_rate = sample_rate

        # Pre-compute filter coefficients using scipy (one-time CPU operation)
        sos = scipy_signal.butter(
            self.order,
            self.cutoff_freq,
            btype='high',
            fs=self.sample_rate,
            output='sos'
        )

        # Convert SOS to transfer function (ba) format for torchaudio
        b, a = scipy_signal.sos2tf(sos)

        # Store as buffers (will move to GPU with model)
        self.register_buffer('a_coeffs', torch.tensor(a, dtype=torch.float32))
        self.register_buffer('b_coeffs', torch.tensor(b, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply highpass filter to input signals (GPU-native, batched).

        Args:
            x: Tensor of shape [batch_size, channels, seq_len]
        Returns:
            Filtered tensor of same shape
        """
        batch_size, channels, seq_len = x.shape
        original_device = x.device
        original_dtype = x.dtype

        # Reshape to [batch_size * channels, seq_len] for torchaudio
        x_reshaped = x.reshape(batch_size * channels, seq_len)

        try:
            # filtfilt requires float32; disable autocast to prevent AMP recasting
            with torch.amp.autocast(device_type=original_device.type, enabled=False):
                x_f32 = x_reshaped.float()

                if x.device.type == 'mps':
                    x_cpu = x_f32.cpu()
                    x_filtered = audio_F.filtfilt(x_cpu, self.a_coeffs.cpu(), self.b_coeffs.cpu(), clamp=False)
                    x_filtered = x_filtered.to(original_device)
                else:
                    x_filtered = audio_F.filtfilt(x_f32, self.a_coeffs, self.b_coeffs, clamp=False)

            return x_filtered.to(original_dtype).reshape(batch_size, channels, seq_len)
        except Exception as e:
            warnings.warn(f"Highpass filtering failed: {e}, using original signal")
            return x


class BandpassFilterPreprocessor(nn.Module):
    """Apply Butterworth bandpass filter to remove noise outside frequency range of interest (GPU-accelerated)."""

    def __init__(
        self,
        low_freq: float = 0.5,
        high_freq: float = 20.0,
        order: int = 4,
        sample_rate: int = 100,
        **kwargs
    ):
        super().__init__()
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order
        self.sample_rate = sample_rate

        # Pre-compute filter coefficients using scipy (one-time CPU operation)
        sos = scipy_signal.butter(
            self.order,
            [self.low_freq, self.high_freq],
            btype='band',
            fs=self.sample_rate,
            output='sos'
        )

        # Convert SOS to transfer function (ba) format for torchaudio
        b, a = scipy_signal.sos2tf(sos)

        # Store as buffers (will move to GPU with model)
        self.register_buffer('a_coeffs', torch.tensor(a, dtype=torch.float32))
        self.register_buffer('b_coeffs', torch.tensor(b, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply bandpass filter to input signals (GPU-native, batched).

        Args:
            x: Tensor of shape [batch_size, channels, seq_len]
        Returns:
            Filtered tensor of same shape
        """
        batch_size, channels, seq_len = x.shape
        original_device = x.device
        original_dtype = x.dtype

        # Reshape to [batch_size * channels, seq_len] for torchaudio
        x_reshaped = x.reshape(batch_size * channels, seq_len)

        try:
            # filtfilt requires float32; disable autocast to prevent AMP recasting
            with torch.amp.autocast(device_type=original_device.type, enabled=False):
                x_f32 = x_reshaped.float()

                if x.device.type == 'mps':
                    x_cpu = x_f32.cpu()
                    x_filtered = audio_F.filtfilt(x_cpu, self.a_coeffs.cpu(), self.b_coeffs.cpu(), clamp=False)
                    x_filtered = x_filtered.to(original_device)
                else:
                    x_filtered = audio_F.filtfilt(x_f32, self.a_coeffs, self.b_coeffs, clamp=False)

            return x_filtered.to(original_dtype).reshape(batch_size, channels, seq_len)
        except Exception as e:
            warnings.warn(f"Bandpass filtering failed: {e}, using original signal")
            return x


class NormalizePreprocessor(nn.Module):
    """Normalize signals using various strategies for consistent scaling."""

    def __init__(
        self,
        method: Literal['zscore', 'minmax', 'robust', 'unit_norm'] = 'zscore',
        per_channel: bool = True,
        epsilon: float = 1e-8,
        **kwargs
    ):
        super().__init__()
        self.method = method
        self.per_channel = per_channel
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input signals.

        Args:
            x: Tensor of shape [batch_size, channels, seq_len]
        Returns:
            Normalized tensor of same shape
        """
        if self.method == 'zscore':
            return self._zscore_normalize(x)
        elif self.method == 'minmax':
            return self._minmax_normalize(x)
        elif self.method == 'robust':
            return self._robust_normalize(x)
        elif self.method == 'unit_norm':
            return self._unit_norm_normalize(x)
        else:
            return x

    def _zscore_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Z-score normalization (mean=0, std=1)."""
        if self.per_channel:
            # Normalize each sample and channel independently over time only.
            # dim=2 avoids batch-dependent statistics that would cause
            # non-deterministic outputs when batch size changes at inference.
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True).clamp_min(self.epsilon)
        else:
            # Normalize each sample independently over all channels and time
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True).clamp_min(self.epsilon)

        return (x - mean) / std

    def _minmax_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Min-max normalization to [0, 1] range."""
        if self.per_channel:
            # Per-sample, per-channel: reduce over time only
            x_min = x.amin(dim=2, keepdim=True)
            x_max = x.amax(dim=2, keepdim=True)
        else:
            # Per-sample: reduce over channels and time
            x_min = x.amin(dim=(1, 2), keepdim=True)
            x_max = x.amax(dim=(1, 2), keepdim=True)

        # Avoid division by zero
        x_range = (x_max - x_min).clamp_min(self.epsilon)
        return (x - x_min) / x_range

    def _robust_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Robust normalization using median and MAD (less sensitive to outliers)."""
        if self.per_channel:
            # Calculate median and MAD per channel
            median = x.median(dim=2, keepdim=True)[0].median(dim=0, keepdim=True)[0]
            mad = torch.median(torch.abs(x - median), dim=2, keepdim=True)[0].median(dim=0, keepdim=True)[0]
        else:
            median = x.median()
            mad = torch.median(torch.abs(x - median))

        mad = mad.clamp_min(self.epsilon)
        return (x - median) / mad

    def _unit_norm_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """L2 unit norm normalization."""
        if self.per_channel:
            # Normalize each channel to unit norm
            norm = torch.norm(x, dim=2, keepdim=True).clamp_min(self.epsilon)
        else:
            norm = torch.norm(x).clamp_min(self.epsilon)

        return x / norm


class BaselineCorrectPreprocessor(nn.Module):
    """Remove baseline drift and DC offset from signals."""

    def __init__(
        self,
        method: Literal['mean', 'median', 'highpass'] = 'mean',
        window_size: Optional[int] = None,
        cutoff_freq: float = 0.1,
        sample_rate: int = 100,
        **kwargs
    ):
        super().__init__()
        self.method = method
        self.window_size = window_size
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate

        if method == 'highpass':
            # Pre-compute highpass filter coefficients
            self.sos = scipy_signal.butter(
                2, self.cutoff_freq, btype='high', fs=self.sample_rate, output='sos'
            )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply baseline correction to input signals.

        Args:
            x: Tensor of shape [batch_size, channels, seq_len]
        Returns:
            Baseline-corrected tensor of same shape
        """
        if self.method == 'mean':
            return self._mean_baseline_correct(x)
        elif self.method == 'median':
            return self._median_baseline_correct(x)
        elif self.method == 'highpass':
            return self._highpass_baseline_correct(x)
        else:
            return x

    def _mean_baseline_correct(self, x: torch.Tensor) -> torch.Tensor:
        """Remove mean baseline (DC offset)."""
        if self.window_size is None:
            # Global mean removal
            baseline = x.mean(dim=2, keepdim=True)
        else:
            # Rolling mean baseline
            baseline = F.avg_pool1d(
                x, kernel_size=self.window_size, stride=1,
                padding=self.window_size//2
            )
        return x - baseline

    def _median_baseline_correct(self, x: torch.Tensor) -> torch.Tensor:
        """Remove median baseline (robust to outliers)."""
        baseline = x.median(dim=2, keepdim=True)[0]
        return x - baseline

    def _highpass_baseline_correct(self, x: torch.Tensor) -> torch.Tensor:
        """Use high-pass filter to remove low-frequency drift."""
        batch_size, channels, seq_len = x.shape
        device = x.device

        x_cpu = x.cpu().numpy()
        x_corrected = np.zeros_like(x_cpu)

        for b in range(batch_size):
            for c in range(channels):
                signal = x_cpu[b, c, :]
                try:
                    x_corrected[b, c, :] = scipy_signal.sosfiltfilt(self.sos, signal)
                except Exception as e:
                    warnings.warn(f"High-pass filtering failed: {e}, using original signal")
                    x_corrected[b, c, :] = signal

        return torch.from_numpy(x_corrected).float().to(device)


class ResamplePreprocessor(nn.Module):
    """Resample signals to different sampling rates."""

    def __init__(
        self,
        target_rate: int,
        original_rate: int = 100,
        method: Literal['linear', 'nearest'] = 'linear',
        **kwargs
    ):
        super().__init__()
        self.target_rate = target_rate
        self.original_rate = original_rate
        self.method = method
        self.resample_ratio = target_rate / original_rate

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resample input signals.

        Args:
            x: Tensor of shape [batch_size, channels, seq_len]
        Returns:
            Resampled tensor of shape [batch_size, channels, new_seq_len]
        """
        if abs(self.resample_ratio - 1.0) < 1e-6:
            # No resampling needed
            return x

        batch_size, channels, seq_len = x.shape
        new_seq_len = int(seq_len * self.resample_ratio)

        # Use PyTorch interpolation for GPU compatibility
        x_reshaped = x.view(batch_size * channels, 1, seq_len)
        x_resampled = F.interpolate(
            x_reshaped,
            size=new_seq_len,
            mode=self.method,
            align_corners=False
        )

        return x_resampled.view(batch_size, channels, new_seq_len)


class SmoothingPreprocessor(nn.Module):
    """Apply smoothing filters to reduce high-frequency noise."""

    def __init__(
        self,
        method: Literal['gaussian', 'median', 'moving_average'] = 'gaussian',
        kernel_size: int = 5,
        sigma: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.method = method
        self.kernel_size = kernel_size
        self.sigma = sigma

        if method == 'gaussian':
            # Pre-compute Gaussian kernel
            self.register_buffer('gaussian_kernel', self._create_gaussian_kernel())

    def _create_gaussian_kernel(self) -> torch.Tensor:
        """Create 1D Gaussian kernel for convolution."""
        x = torch.arange(self.kernel_size, dtype=torch.float32)
        x = x - (self.kernel_size - 1) / 2
        kernel = torch.exp(-(x**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, kernel_size]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply smoothing to input signals.

        Args:
            x: Tensor of shape [batch_size, channels, seq_len]
        Returns:
            Smoothed tensor of same shape
        """
        if self.method == 'gaussian':
            return self._gaussian_smooth(x)
        elif self.method == 'median':
            return self._median_smooth(x)
        elif self.method == 'moving_average':
            return self._moving_average_smooth(x)
        else:
            return x

    def _gaussian_smooth(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing."""
        batch_size, channels, seq_len = x.shape

        # Reshape for group convolution (one filter per channel)
        x_reshaped = x.view(batch_size, channels, seq_len)

        # Apply convolution with proper padding
        padding = self.kernel_size // 2
        x_smoothed = F.conv1d(
            x_reshaped,
            self.gaussian_kernel.repeat(channels, 1, 1),
            padding=padding,
            groups=channels
        )

        return x_smoothed

    def _median_smooth(self, x: torch.Tensor) -> torch.Tensor:
        """Apply median filtering (requires manual implementation)."""
        batch_size, channels, seq_len = x.shape
        x_smoothed = torch.zeros_like(x)

        padding = self.kernel_size // 2
        x_padded = F.pad(x, (padding, padding), mode='reflect')

        for i in range(seq_len):
            window = x_padded[:, :, i:i + self.kernel_size]
            x_smoothed[:, :, i] = torch.median(window, dim=2)[0]

        return x_smoothed

    def _moving_average_smooth(self, x: torch.Tensor) -> torch.Tensor:
        """Apply moving average smoothing."""
        # Create uniform kernel
        kernel = torch.ones(1, 1, self.kernel_size, device=x.device) / self.kernel_size

        batch_size, channels, seq_len = x.shape
        x_reshaped = x.view(batch_size, channels, seq_len)

        padding = self.kernel_size // 2
        x_smoothed = F.conv1d(
            x_reshaped,
            kernel.repeat(channels, 1, 1),
            padding=padding,
            groups=channels
        )

        return x_smoothed


class UnitConversionPreprocessor(nn.Module):
    """Convert acceleration units (e.g., g to m/s², m/s² to g)."""

    def __init__(
        self,
        from_unit: Literal['g', 'm/s2', 'mg'] = 'g',
        to_unit: Literal['g', 'm/s2', 'mg'] = 'm/s2',
        **kwargs
    ):
        super().__init__()
        self.from_unit = from_unit
        self.to_unit = to_unit

        # Conversion factors (all to m/s²)
        to_mps2 = {
            'g': 9.80665,
            'm/s2': 1.0,
            'mg': 0.00980665,
        }

        # Calculate conversion factor
        self.conversion_factor = to_mps2[from_unit] / to_mps2[to_unit]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert acceleration units.

        Args:
            x: Tensor of shape [batch_size, channels, seq_len]
        Returns:
            Converted tensor of same shape
        """
        return x * self.conversion_factor


class SensorCalibrationPreprocessor(nn.Module):
    """Apply sensor calibration using calibration matrix and bias vector."""

    def __init__(
        self,
        calibration_matrix: Optional[List[List[float]]] = None,
        bias_vector: Optional[List[float]] = None,
        apply_per_session: bool = False,
        **kwargs
    ):
        super().__init__()
        self.apply_per_session = apply_per_session

        # Default to identity calibration
        if calibration_matrix is None:
            calibration_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        if bias_vector is None:
            bias_vector = [0.0, 0.0, 0.0]

        # Register calibration parameters as buffers
        self.register_buffer('calib_matrix', torch.tensor(calibration_matrix, dtype=torch.float32))
        self.register_buffer('bias', torch.tensor(bias_vector, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sensor calibration.

        Args:
            x: Tensor of shape [batch_size, channels, seq_len]
        Returns:
            Calibrated tensor of same shape
        """
        batch_size, channels, seq_len = x.shape

        # Reshape to (batch_size, seq_len, channels) for matrix multiplication
        x_t = x.permute(0, 2, 1)

        # Apply calibration: x_calibrated = (x - bias) @ calib_matrix^T
        x_centered = x_t - self.bias.view(1, 1, -1)
        x_calibrated = torch.matmul(x_centered, self.calib_matrix.t())

        # Reshape back to (batch_size, channels, seq_len)
        x_out = x_calibrated.permute(0, 2, 1)

        return x_out


class PatientNormalizationPreprocessor(nn.Module):
    """
    Preprocessor wrapper for patient-level normalization.

    Stats are automatically loaded from Zarr dataset and aggregated in DataModule,
    then injected into this preprocessor by the pipeline at training start.
    This ensures no data leakage (only training fold stats are used).

    This must be the SECOND normalization stage (after protocol harmonization).
    """

    def __init__(
        self,
        num_features: int = 3,
        stats: Optional[Dict[str, torch.Tensor]] = None,
        method: Literal['zscore', 'robust'] = 'zscore',
        eps: float = 1e-8,
        **kwargs
    ):
        super().__init__()

        # Import and instantiate normalizer
        from model.hierarchical_normalizers import PatientLevelNormalizer
        self.normalizer = PatientLevelNormalizer(
            num_features=num_features,
            stats=stats,  # Will be None initially, injected later by pipeline
            method=method,
            eps=eps
        )
        self.method = method
        self.num_features = num_features
        self.eps = eps

    def set_stats(self, stats: Dict[str, torch.Tensor]) -> None:
        """
        Set normalization stats from datamodule (called by pipeline before training).

        Args:
            stats: Dictionary with 'mean' and 'std' (or 'median' and 'mad') tensors
        """
        if self.method == 'zscore':
            if 'mean' not in stats or 'std' not in stats:
                raise ValueError("Stats dict must contain 'mean' and 'std' for zscore method")
            self.normalizer.mean = stats['mean'].view(1, -1, 1).to(self.normalizer.mean.device)
            self.normalizer.std = stats['std'].view(1, -1, 1).to(self.normalizer.std.device)
            logger.info(f"Injected patient normalization stats (zscore): mean={stats['mean'].numpy()}, std={stats['std'].numpy()}")
        elif self.method == 'robust':
            if 'median' not in stats or 'mad' not in stats:
                raise ValueError("Stats dict must contain 'median' and 'mad' for robust method")
            self.normalizer.median = stats['median'].view(1, -1, 1).to(self.normalizer.median.device)
            self.normalizer.mad = stats['mad'].view(1, -1, 1).to(self.normalizer.mad.device)
            logger.info(f"Injected patient normalization stats (robust): median={stats['median'].numpy()}, mad={stats['mad'].numpy()}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply patient-level normalization."""
        return self.normalizer(x)


class SessionNormalizationPreprocessor(nn.Module):
    """
    Preprocessor wrapper for session-level normalization.

    Applies per-session adaptive normalization (RevIN or FAN).
    This must be the THIRD normalization stage (after patient-level normalization).
    """

    def __init__(
        self,
        num_features: int = 3,
        method: Literal['revin', 'fan', 'none'] = 'revin',
        eps: float = 1e-5,
        affine: bool = True,
        # RevIN-specific params
        subtract_last: bool = False,
        # FAN-specific params
        freq_bands: Optional[List[Tuple[float, float]]] = None,
        sample_rate: float = 100.0,
        **kwargs
    ):
        super().__init__()
        self.method = method

        if method == 'revin':
            from model.hierarchical_normalizers import RevIN
            self.normalizer = RevIN(
                num_features=num_features,
                eps=eps,
                affine=affine,
                subtract_last=subtract_last
            )
        elif method == 'fan':
            from model.hierarchical_normalizers import FAN
            if freq_bands is None:
                freq_bands = [(0.5, 3.0), (3.0, 8.0), (8.0, 20.0)]
            self.normalizer = FAN(
                num_features=num_features,
                freq_bands=freq_bands,
                sample_rate=sample_rate,
                eps=eps,
                affine=affine
            )
        elif method == 'none':
            self.normalizer = nn.Identity()
        else:
            raise ValueError(f"Unknown session normalization method: {method}")

        logger.info(f"Initialized SessionNormalizationPreprocessor with method={method}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply session-level normalization."""
        if self.method == 'none':
            return x
        return self.normalizer(x, mode='norm')


class SignalPreprocessor(nn.Module):
    """
    Main preprocessing module that applies multiple preprocessors sequentially.
    Mirrors the SignalAugmenter pattern from augmentors.py.
    """

    def __init__(self, preprocessors: List[nn.Module]):
        """
        Initialize with a list of already instantiated preprocessors.

        Args:
            preprocessors: List of instantiated preprocessor modules
        """
        super().__init__()
        self.preprocessors = nn.ModuleList(preprocessors)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all preprocessors sequentially."""
        for preprocessor in self.preprocessors:
            x = preprocessor(x)
        return x


# Registry for Hydra instantiation (mirrors SIGNAL_AUGMENTORS_MAP)
PREPROCESSORS_MAP = {
    "DetrendPreprocessor": DetrendPreprocessor,
    "LowpassFilterPreprocessor": LowpassFilterPreprocessor,
    "HighpassFilterPreprocessor": HighpassFilterPreprocessor,
    "BandpassFilterPreprocessor": BandpassFilterPreprocessor,
    "NormalizePreprocessor": NormalizePreprocessor,
    "BaselineCorrectPreprocessor": BaselineCorrectPreprocessor,
    "ResamplePreprocessor": ResamplePreprocessor,
    "SmoothingPreprocessor": SmoothingPreprocessor,
    "UnitConversionPreprocessor": UnitConversionPreprocessor,
    "SensorCalibrationPreprocessor": SensorCalibrationPreprocessor,
    "PatientNormalizationPreprocessor": PatientNormalizationPreprocessor,
    "SessionNormalizationPreprocessor": SessionNormalizationPreprocessor,
    "SignalPreprocessor": SignalPreprocessor,
}