import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from scipy import signal as scipy_signal


class GramianAngularFieldTransform(nn.Module):
    """
    Gramian Angular Field (GAF) preprocessor for time series imaging.

    Transforms 1D time series into 2D images using angular representation,
    which preserves temporal correlations and is particularly effective
    for periodic signals like human gait.

    Supports both:
    - GASF (Gramian Angular Summation Field): cos(φᵢ + φⱼ)
    - GADF (Gramian Angular Difference Field): sin(φᵢ - φⱼ)
    """

    def __init__(
        self,
        method: str = "GASF",
        img_size: int = 224,
        interpolate: bool = True,
        norm_method: str = "minmax",
        norm_range: list = [-1, 1],
        use_segmentation: bool = True,
        segment_size: int = 64,
        segment_step: int = 32,
        use_multiscale: bool = False,
        scales: list = [1, 2, 4],
        use_polar_encoding: bool = True,
        **kwargs,
    ):
        super().__init__()

        # GAF parameters
        self.method = method
        self.img_size = img_size
        self.interpolate = interpolate

        # Normalization parameters
        self.norm_method = norm_method
        self.norm_range = norm_range

        # Segmentation for long sequences
        self.use_segmentation = use_segmentation
        self.segment_size = segment_size
        self.segment_step = segment_step

        # Multi-scale GAF
        self.use_multiscale = use_multiscale
        self.scales = scales

        # Polar encoding parameters
        self.use_polar_encoding = use_polar_encoding

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gramian Angular Field transformation.

        Args:
            x: Input tensor of shape [batch, channels, seq_len]
        Returns:
            GAF images of shape [batch, channels, img_size, img_size]
        """
        B, C, L = x.shape
        device = x.device

        # Process each channel independently
        gaf_images = []

        for b in range(B):
            channel_gafs = []

            for c in range(C):
                signal = x[b, c]

                if self.use_segmentation and L > self.segment_size:
                    # Create multiple GAF images from segments
                    segment_gafs = self._create_segmented_gaf(signal)
                    # Combine segments into a single image
                    combined_gaf = self._combine_segments(segment_gafs)
                    channel_gafs.append(combined_gaf)
                else:
                    # Create single GAF for entire signal
                    gaf = self._create_gaf(signal)
                    channel_gafs.append(gaf)

            # Stack channels
            batch_gaf = torch.stack(channel_gafs, dim=0)
            gaf_images.append(batch_gaf)

        # Stack batches
        gaf_tensor = torch.stack(gaf_images, dim=0)

        # Normalize the GAF values
        gaf_normalized = self._normalize_gaf(gaf_tensor)

        # Interpolate to target size if needed
        if self.interpolate and gaf_normalized.shape[-1] != self.img_size:
            gaf_resized = F.interpolate(
                gaf_normalized,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            return gaf_resized

        return gaf_normalized

    def _create_gaf(self, signal: torch.Tensor) -> torch.Tensor:
        """Create GAF matrix from a 1D signal."""
        # Normalize signal to [-1, 1]
        signal_norm = self._normalize_signal(signal)

        # Ensure values are in valid range for arccos
        signal_norm = torch.clamp(signal_norm, -0.999, 0.999)

        if self.use_polar_encoding:
            # Convert to polar coordinates
            # φ = arccos(x), where x is normalized signal
            phi = torch.arccos(signal_norm)

            # Create angle matrices
            # Use broadcasting to create pairwise angle matrices
            phi_i = phi.unsqueeze(0)  # Shape: [1, L]
            phi_j = phi.unsqueeze(1)  # Shape: [L, 1]

            if self.method == "GASF":
                # Gramian Angular Summation Field
                gaf = torch.cos(phi_i + phi_j)
            elif self.method == "GADF":
                # Gramian Angular Difference Field
                gaf = torch.sin(phi_i - phi_j)
            else:
                raise ValueError(f"Unknown GAF method: {self.method}")
        else:
            # Alternative: Direct computation without polar encoding
            # This can be more stable for noisy signals
            outer_product = torch.outer(signal_norm, signal_norm)

            if self.method == "GASF":
                # GASF approximation
                gaf = outer_product
            elif self.method == "GADF":
                # GADF approximation using temporal difference
                i_mat = torch.arange(len(signal), device=signal.device).unsqueeze(0)
                j_mat = torch.arange(len(signal), device=signal.device).unsqueeze(1)
                time_diff = (i_mat - j_mat).float() / len(signal)
                gaf = outer_product * time_diff

        return gaf

    def _normalize_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """Normalize signal to specified range."""
        if self.norm_method == "minmax":
            # Min-max normalization
            sig_min = signal.min()
            sig_max = signal.max()

            # Avoid division by zero
            if sig_max - sig_min < 1e-6:
                return torch.zeros_like(signal)

            # Normalize to [0, 1]
            signal_01 = (signal - sig_min) / (sig_max - sig_min)

            # Scale to desired range
            range_min, range_max = self.norm_range
            signal_norm = signal_01 * (range_max - range_min) + range_min

        elif self.norm_method == "mean":
            # Mean normalization
            mean = signal.mean()
            std = signal.std() + 1e-8
            signal_norm = (signal - mean) / std

            # Clip to range
            signal_norm = torch.clamp(
                signal_norm, self.norm_range[0], self.norm_range[1]
            )

        else:
            signal_norm = signal

        return signal_norm

    def _create_segmented_gaf(self, signal: torch.Tensor) -> List[torch.Tensor]:
        """Create multiple GAF images from overlapping segments."""
        L = len(signal)
        segments = []

        # Extract overlapping segments
        for start in range(0, L - self.segment_size + 1, self.segment_step):
            end = start + self.segment_size
            segment = signal[start:end]

            # Create GAF for this segment
            gaf = self._create_gaf(segment)
            segments.append(gaf)

        # Handle the last segment if there's remainder
        if L % self.segment_step != 0:
            last_segment = signal[-self.segment_size :]
            gaf = self._create_gaf(last_segment)
            segments.append(gaf)

        return segments

    def _combine_segments(self, segments: List[torch.Tensor]) -> torch.Tensor:
        """Combine multiple GAF segments into a single image."""
        if not segments:
            return torch.zeros(self.img_size, self.img_size)

        n_segments = len(segments)

        if n_segments == 1:
            return segments[0]

        # Strategy 1: Tile segments in a grid
        grid_size = int(np.ceil(np.sqrt(n_segments)))

        # Resize each segment to fit in the grid
        segment_size = self.img_size // grid_size
        resized_segments = []

        for seg in segments:
            if seg.shape[0] != segment_size:
                seg_resized = F.interpolate(
                    seg.unsqueeze(0).unsqueeze(0),
                    size=(segment_size, segment_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            else:
                seg_resized = seg
            resized_segments.append(seg_resized)

        # Create grid
        grid = torch.zeros(self.img_size, self.img_size, device=segments[0].device)

        for idx, seg in enumerate(resized_segments):
            row = idx // grid_size
            col = idx % grid_size

            row_start = row * segment_size
            col_start = col * segment_size

            # Handle edge cases where segment might not fit perfectly
            row_end = min(row_start + segment_size, self.img_size)
            col_end = min(col_start + segment_size, self.img_size)

            actual_height = row_end - row_start
            actual_width = col_end - col_start

            grid[row_start:row_end, col_start:col_end] = seg[
                :actual_height, :actual_width
            ]

        return grid

    def _normalize_gaf(self, gaf: torch.Tensor) -> torch.Tensor:
        """Normalize GAF values for better contrast."""
        # GAF values are typically in [-1, 1] for GASF and GADF
        # We can enhance contrast by adjusting the range

        # Option 1: Keep original range (good for GASF/GADF distinction)
        # return gaf

        # Option 2: Normalize to [0, 1] for better visualization
        gaf_min = gaf.min()
        gaf_max = gaf.max()

        if gaf_max - gaf_min > 1e-6:
            gaf_normalized = (gaf - gaf_min) / (gaf_max - gaf_min)
        else:
            gaf_normalized = gaf

        return gaf_normalized


class QTransform(nn.Module):
    """
    Constant Q Transform (CQT) preprocessor for time-frequency analysis.

    The Q-transform provides better frequency resolution at low frequencies
    and better time resolution at high frequencies, which is ideal for
    analyzing human gait patterns that contain both slow locomotion
    frequencies (0.5-3 Hz) and faster FoG frequencies (3-8 Hz).
    """

    def __init__(
        self,
        sample_rate: int = 100,
        f_min: float = 0.5,
        f_max: float = 10.0,
        bins_per_octave: int = 24,
        filter_scale: float = 1.0,
        hop_length: int = 32,
        img_size: int = 224,
        interpolate: bool = True,
        max_kernel_length: int = 256,
        **kwargs,
    ):
        super().__init__()

        # Q-transform parameters
        self.fs = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.bins_per_octave = bins_per_octave
        self.filter_scale = filter_scale
        self.hop_length = hop_length
        self.img_size = img_size
        self.interpolate = interpolate
        self.max_kernel_length = max_kernel_length

        # Pre-compute Q-transform parameters
        self._setup_qtransform()

    def _setup_qtransform(self):
        """Pre-compute the constant Q transform kernels."""
        # Calculate number of octaves
        n_octaves = np.ceil(np.log2(self.f_max / self.f_min))
        self.n_bins = int(n_octaves * self.bins_per_octave)

        # Generate geometrically spaced frequencies
        freqs = self.f_min * 2.0 ** (np.arange(self.n_bins) / self.bins_per_octave)
        self.register_buffer("frequencies", torch.tensor(freqs, dtype=torch.float32))

        # Q factor (constant across frequencies)
        self.Q = self.filter_scale / (2.0 ** (1.0 / self.bins_per_octave) - 1)

        # Calculate window lengths for each frequency
        # Window length inversely proportional to frequency to maintain constant Q
        self.lengths = (self.Q * self.fs / freqs).astype(int)

        # Get expected input length from config (block_len)
        self.lengths = np.clip(
            self.lengths, 16, self.max_kernel_length
        )  # Ensure kernels fit in input

        # Pre-compute kernels for each frequency bin
        self.kernels = []
        self.kernel_lengths = []

        for k, (freq, length) in enumerate(zip(freqs, self.lengths)):
            # Create complex sinusoid kernel
            t = torch.arange(length, dtype=torch.float32) / self.fs
            # Apply window (Hann window for good frequency selectivity)
            window = torch.hann_window(length)
            # Complex exponential modulated by window
            kernel = window * torch.exp(-2j * np.pi * freq * t)
            self.kernels.append(kernel)
            self.kernel_lengths.append(length)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Constant Q Transform to input signal.

        Args:
            x: Input tensor of shape [batch, channels, seq_len]
        Returns:
            CQT spectrogram of shape [batch, channels, img_size, img_size]
        """
        B, C, L = x.shape
        device = x.device

        # Pad input if needed to ensure we have enough samples
        min_length_needed = max(self.kernel_lengths) + self.hop_length
        if L < min_length_needed:
            pad_length = min_length_needed - L
            x = F.pad(x, (0, pad_length), mode="reflect")
            L = x.shape[-1]

        # Calculate number of time frames
        n_frames = max(1, 1 + (L - max(self.kernel_lengths)) // self.hop_length)

        # Initialize output tensor
        cqt_complex = torch.zeros(
            B * C, self.n_bins, n_frames, dtype=torch.complex64, device=device
        )

        # Flatten batch and channels for processing
        x_flat = x.view(B * C, L)

        # Apply CQT for each frequency bin
        for k, (kernel, kernel_len) in enumerate(
            zip(self.kernels, self.kernel_lengths)
        ):
            kernel = kernel.to(device)

            # Skip if kernel is longer than signal (shouldn't happen with our setup)
            if kernel_len > L:
                continue

            # Convolve with complex kernel using real and imaginary parts separately
            kernel_real = kernel.real.flip(0).unsqueeze(0).unsqueeze(0)
            kernel_imag = kernel.imag.flip(0).unsqueeze(0).unsqueeze(0)

            # Apply convolution
            conv_real = F.conv1d(
                x_flat.unsqueeze(1), kernel_real, stride=self.hop_length, padding=0
            )
            conv_imag = F.conv1d(
                x_flat.unsqueeze(1), kernel_imag, stride=self.hop_length, padding=0
            )

            # Combine into complex result
            result = conv_real + 1j * conv_imag

            # Store in appropriate position
            valid_frames = min(result.shape[-1], n_frames)
            if valid_frames > 0:
                cqt_complex[:, k, :valid_frames] = result[:, 0, :valid_frames]

        # Convert to magnitude and apply log scaling
        cqt_mag = torch.abs(cqt_complex)

        # Apply perceptual scaling (similar to mel scale)
        cqt_db = AF.amplitude_to_DB(
            cqt_mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0, top_db=80.0
        )

        # Reshape back to batch format
        cqt_db = cqt_db.view(B, C, self.n_bins, -1)

        # Normalize per frequency bin
        mu = cqt_db.mean(dim=(0, 1, 3), keepdim=True)
        std = cqt_db.std(dim=(0, 1, 3), keepdim=True).clamp_min(1e-6)
        cqt_normalized = (cqt_db - mu) / std

        # Interpolate to target size if needed
        if self.interpolate:
            cqt_resized = F.interpolate(
                cqt_normalized,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            return cqt_resized.float()

        return cqt_normalized.float()


class HilbertTransform(nn.Module):
    """
    Hilbert Transform preprocessor for analytic signal analysis.

    The Hilbert transform computes the analytic signal, providing:
    1. Instantaneous amplitude (envelope)
    2. Instantaneous phase
    3. Instantaneous frequency

    These features are particularly useful for FoG detection as they can
    capture sudden changes in gait rhythm and amplitude.
    """

    def __init__(
        self,
        sample_rate: int = 100,
        img_size: int = 224,
        interpolate: bool = True,
        compute_envelope: bool = True,
        compute_phase: bool = True,
        compute_frequency: bool = True,
        use_filter_bank: bool = False,
        freq_bands: list = [[0.5, 3.0], [3.0, 8.0], [8.0, 15.0]],
        window_size: int = 128,
        hop_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.fs = sample_rate
        self.img_size = img_size
        self.interpolate = interpolate

        # Feature extraction options
        self.compute_envelope = compute_envelope
        self.compute_phase = compute_phase
        self.compute_frequency = compute_frequency

        # Frequency band analysis (optional)
        self.use_filter_bank = use_filter_bank
        if self.use_filter_bank:
            self.freq_bands = freq_bands
            self._setup_filters()

        # Time-frequency representation parameters
        self.window_size = window_size
        self.hop_size = hop_size

    def _setup_filters(self):
        """Setup bandpass filters for multi-band Hilbert analysis."""
        self.filters = []
        for f_low, f_high in self.freq_bands:
            # Design Butterworth bandpass filter
            sos = scipy_signal.butter(
                4, [f_low, f_high], btype="band", fs=self.fs, output="sos"
            )
            self.filters.append(sos)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Hilbert transform to extract analytic signal features.

        Args:
            x: Input tensor of shape [batch, channels, seq_len]
        Returns:
            Feature tensor of shape [batch, channels * n_features, img_size, img_size]
        """
        B, C, L = x.shape
        device = x.device

        # Process each channel
        features_list = []

        for b in range(B):
            for c in range(C):
                signal = x[b, c].cpu().numpy()

                if self.use_filter_bank:
                    # Multi-band analysis
                    band_features = []
                    for sos in self.filters:
                        # Apply bandpass filter
                        filtered = scipy_signal.sosfiltfilt(sos, signal)
                        # Compute Hilbert transform
                        analytic = scipy_signal.hilbert(filtered)
                        band_features.append(self._extract_features(analytic))

                    # Stack band features
                    channel_features = np.vstack(band_features)
                else:
                    # Single-band analysis on full signal
                    # Apply Hilbert transform
                    analytic = scipy_signal.hilbert(signal)
                    channel_features = self._extract_features(analytic)

                features_list.append(channel_features)

        # Convert to tensor and reshape
        features = np.stack(features_list)
        features_tensor = torch.from_numpy(features).float().to(device)

        # Reshape to [B, C * n_features, time_frames, feature_dims]
        n_features = features_tensor.shape[1]
        features_tensor = features_tensor.view(B, C * n_features, -1)

        # Create time-frequency representation using sliding window
        tf_features = self._create_time_frequency_map(features_tensor)

        # Normalize features
        mu = tf_features.mean(dim=(0, 1), keepdim=True)
        std = tf_features.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        tf_normalized = (tf_features - mu) / std

        # Ensure we have a 4D tensor for interpolation
        if tf_normalized.dim() == 3:
            tf_normalized = tf_normalized.unsqueeze(1)

        # Interpolate to target size
        if self.interpolate:
            tf_resized = F.interpolate(
                tf_normalized,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            return tf_resized

        return tf_normalized

    def _extract_features(self, analytic_signal):
        """Extract features from analytic signal."""
        features = []

        if self.compute_envelope:
            # Instantaneous amplitude (envelope)
            envelope = np.abs(analytic_signal)
            features.append(envelope)

        if self.compute_phase:
            # Instantaneous phase
            phase = np.angle(analytic_signal)
            # Unwrap phase for continuity
            phase_unwrapped = np.unwrap(phase)
            features.append(phase_unwrapped)

        if self.compute_frequency:
            # Instantaneous frequency (derivative of phase)
            phase = np.angle(analytic_signal)
            phase_unwrapped = np.unwrap(phase)
            # Use gradient for smoother derivative
            inst_freq = np.gradient(phase_unwrapped) * self.fs / (2 * np.pi)
            # Clip to reasonable frequency range
            inst_freq = np.clip(inst_freq, 0, self.fs / 2)
            features.append(inst_freq)

        return np.array(features)

    def _create_time_frequency_map(self, features):
        """Create a 2D time-frequency representation from 1D features."""
        B, F, L = features.shape

        # Calculate number of windows
        n_windows = (L - self.window_size) // self.hop_size + 1

        # Initialize output
        tf_map = torch.zeros(B, F, self.window_size, n_windows, device=features.device)

        # Create overlapping windows
        for i in range(n_windows):
            start = i * self.hop_size
            end = start + self.window_size
            if end <= L:
                tf_map[:, :, :, i] = features[:, :, start:end]
            else:
                # Pad last window if necessary
                valid_len = L - start
                tf_map[:, :, :valid_len, i] = features[:, :, start:L]

        # Reshape to combine feature and frequency dimensions
        tf_map = tf_map.view(B, -1, self.window_size, n_windows)

        return tf_map


class Integrator(nn.Module):
    def __init__(
        self,
        window_size: int = None,
        hop_size: int = None,
        normalize: bool = False,
        baseline_correction: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.normalize = normalize
        self.baseline_correction = baseline_correction

    @torch.no_grad()
    def forward(self, x):
        # Basic integration with cumulative sum
        integrated = torch.cumsum(x, dim=-1)

        # Optional baseline correction to remove drift
        if self.baseline_correction:
            # Subtract the linear trend
            seq_len = integrated.shape[-1]
            t = torch.linspace(0, 1, seq_len, device=x.device)
            # For each batch and channel, calculate and remove the trend
            for b in range(integrated.shape[0]):
                for c in range(integrated.shape[1]):
                    signal = integrated[b, c]
                    slope = signal[-1] - signal[0]
                    trend = slope * t
                    integrated[b, c] = signal - trend

        # Optional windowed integration for better stability
        if self.window_size and self.hop_size:
            # Implement windowed integration to prevent error accumulation
            seq_len = x.shape[-1]
            windows = []
            for i in range(0, seq_len - self.window_size + 1, self.hop_size):
                window = torch.cumsum(x[..., i : i + self.window_size], dim=-1)
                windows.append(window)
            integrated = torch.cat(windows, dim=-1)

        # Optional normalization
        if self.normalize:
            # Normalize each channel independently
            mean = integrated.mean(dim=-1, keepdim=True)
            std = integrated.std(dim=-1, keepdim=True) + 1e-8  # Avoid division by zero
            integrated = (integrated - mean) / std

        return integrated


class Differentiator(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x):
        return torch.diff(x, dim=-1, prepend=torch.zeros_like(x[:, :, :1]))


class MelSpectrogramTransform(nn.Module):
    def __init__(
        self,
        sample_rate: int = 100,
        n_fft: int = 256,
        hop_length: int = 32,
        win_length: int = 128,
        n_mels: int = 80,
        f_min: float = 0.5,
        f_max: float = 20.0,
        interpolate: bool = True,
        mel_power: int = 2,
        mel_pwr: float = 0.3,
        mel_eps: float = 3e-6,
        mel_win_mult: float = 1.5,
        center: bool = True,
        norm: str = "slaney",
        mel_scale: str = "slaney",
        multiplier: int = 10,
        amin: float = 1e-9,
        db_multiplier: int = 1,
        top_db: int = 60,
        img_size: int = 224,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()

        # Extract physics-aware parameters with sensible defaults
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.interpolate = interpolate
        # Physics-aware parameters from WalkNetwork
        self.mel_power = mel_power
        self.mel_pwr = mel_pwr
        self.mel_eps = mel_eps
        self.mel_win_mult = mel_win_mult

        # Core STFT->Mel transform
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            window_fn=torch.hann_window,
            center=center,
            power=self.mel_power,
            norm=norm,
            mel_scale=mel_scale,
            f_min=self.f_min,
            f_max=self.f_max,
        ).to(device)

        # Save additional parameters for DB conversion
        self.to_db_kwargs = {
            "multiplier": multiplier,
            "amin": amin,
            "db_multiplier": db_multiplier,
            "top_db": top_db,
        }

        # Target dimensions
        self.img_size = img_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [batch, channels, seq_len]
        returns : [batch, channels, img_size, img_size]
        """
        b, c, seq_len = x.shape
        # (2) Compute STFT→Mel (batched over all channels)
        # Pad input to avoid boundary artifacts (using WalkNetwork style padding)
        pad_size = min(self.win_length // 2, seq_len // 4)  # Ensure padding is not too large
        x_padded = F.pad(x.view(b * c, -1), (pad_size, pad_size), mode="reflect")

        # Apply mel spectrogram
        mel_spec = self.mel(x_padded)

        # (3) Apply physics-aware power transformation (from WalkNetwork)
        if self.mel_pwr:
            # Power transformation (better for acceleration signals)
            mel_spec = mel_spec.pow(self.mel_pwr)
        else:
            # Or use log scaling with epsilon
            mel_spec = torch.log(mel_spec + self.mel_eps) / 3

        # Reshape back to batch dimensions
        mel_spec = mel_spec.view(b, c, *mel_spec.shape[-2:])

        # (4) Frequency-wise whitening over all dimensions
        f_mu = mel_spec.mean(dim=(0, 1, 3), keepdim=True)
        f_std = mel_spec.std(dim=(0, 1, 3), keepdim=True).clamp_min(1e-6)
        mel_spec = (mel_spec - f_mu) / f_std

        # (5) Resize to target dimensions
        if self.interpolate:
            mel_spec = F.interpolate(
                mel_spec,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

        return mel_spec


class PatchTransform(nn.Module):
    """
    Patches the input signal into fixed-size patches with optional overlap.
    Similar to the winning solution's patching approach.
    """

    def __init__(self, patch_size: int, flatten_patches: bool = True, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.flatten_patches = flatten_patches

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, channels, seq_len]
        Returns:
            Patched tensor of shape [batch_size, num_patches, patch_size * channels] if flattened
            or [batch_size, num_patches, channels, patch_size] if not flattened
        """
        batch_size, channels, seq_len = x.shape

        # Ensure sequence length is divisible by patch_size
        if seq_len % self.patch_size != 0:
            pad_len = self.patch_size - (seq_len % self.patch_size)
            x = F.pad(x, (0, pad_len), mode="constant", value=0)
            seq_len = x.shape[2]

        num_patches = seq_len // self.patch_size

        # Reshape to patches
        x = x.view(batch_size, channels, num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3)  # [batch, num_patches, channels, patch_size]

        if self.flatten_patches:
            # Flatten each patch
            x = x.reshape(
                batch_size, num_patches, -1
            )  # [batch, num_patches, channels * patch_size]

        return x


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cwt_gpu import GpuCwt


class WaveletTransform(nn.Module):
    """
    Enhanced Waveletifier with FOG-optimized frequency distribution
    """

    def __init__(
        self,
        n_scales: int = 64,
        wavelet: str = "morl",
        max_freq: float = 10.0,
        img_size: int = 224,
        interpolate: bool = True,
        highlight_transient: bool = False,
        transient_weight: float = 0.15,
        top_db: float = 60.0,
        **kwargs,
    ):
        super().__init__()
        self.fs = 100.0

        # FOG-optimized frequency distribution
        # Still 64 total scales

        # Distribute scales with higher density in FOG range (3-8 Hz)
        fog_range_scales = int(n_scales * 0.625)  # 40 scales for FOG range (3-8 Hz)
        locomotion_scales = int(n_scales * 0.25)  # 16 scales for locomotion (0.5-3 Hz)
        high_freq_scales = (
            n_scales - fog_range_scales - locomotion_scales
        )  # 8 scales for 8-10 Hz

        # Create frequency arrays for each range
        freq_locomotion = torch.logspace(
            math.log10(0.5), math.log10(3.0), steps=locomotion_scales, device="cpu"
        )
        freq_fog = torch.logspace(
            math.log10(3.0), math.log10(8.0), steps=fog_range_scales, device="cpu"
        )
        freq_high = torch.logspace(
            math.log10(8.0), math.log10(max_freq), steps=high_freq_scales, device="cpu"
        )

        # Combine all frequency ranges
        freqs = torch.cat([freq_locomotion, freq_fog, freq_high])

        # Set center frequency based on wavelet type
        if wavelet.startswith("db"):
            fc = 0.67  # Daubechies center frequency
        elif wavelet == "morl":
            fc = 0.8125  # Morlet center frequency
        else:
            fc = 1.0

        # Convert frequencies to scales
        scales = fc * self.fs / freqs
        self.register_buffer("scales", scales)

        # Store frequency bands for analysis
        self.register_buffer("freq_locomotion_mask", freqs <= 3.0)
        self.register_buffer("freq_fog_mask", (freqs > 3.0) & (freqs <= 8.0))

        self.img_size = img_size
        self.interpolate = interpolate
        self.wavelet = wavelet
        self.highlight_transient = highlight_transient
        self.transient_weight = transient_weight
        self.top_db = top_db

        # GPU-native CWT — precomputes wavelet buffers once
        self.gpu_cwt = GpuCwt(scales, wavelet=wavelet)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass with Morlet wavelet and proper power calculation.
        """
        B, C, L = x.shape

        # Pad input to avoid boundary artifacts, but ensure padding is reasonable
        pad = min(L // 4, L // 8)  # Limit padding to avoid issues with very short signals
        x_pad = F.pad(x, (pad, pad), mode="reflect")

        # Apply CWT with GPU-native implementation (no CPU round-trips)
        coefs = self.gpu_cwt(x_pad.reshape(-1, x_pad.size(-1)))

        coefs = coefs.permute(1, 0, 2)  # [B*C, F, T]
        # IMPORTANT: Trim padding from time dimension
        coefs = coefs[..., pad:-pad]

        # Calculate TRUE power (magnitude squared, not sqrt)
        power_spec = coefs.abs().pow(2)

        # Use log1p for stable dynamic range compression
        log_power_spec = torch.log1p(power_spec)

        # Reshape back to [B, C, F, T]
        log_power_spec = log_power_spec.view(B, C, *log_power_spec.shape[-2:])

        # Light instance normalization (preserves spectral shape)
        mu = log_power_spec.mean(dim=(2, 3), keepdim=True)
        std = log_power_spec.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        spec_normalized = (log_power_spec - mu) / std

        # Interpolate to target image size if needed
        if self.interpolate:
            spec_final = F.interpolate(
                spec_normalized,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            return spec_final.float()

        return spec_normalized.float()

    def get_frequency_bands_power(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract power in different frequency bands for FOG analysis
        Useful for computing Freeze Index = Power(3-8Hz) / Power(0.5-3Hz)
        """
        # Apply CWT (simplified version of forward pass)
        B, C, L = x.shape
        pad = L // 2
        x_pad = F.pad(x, (pad, pad), mode="reflect")

        coefs = self.gpu_cwt(x_pad.reshape(-1, x_pad.size(-1)))
        coefs = coefs.permute(1, 0, 2).view(B, C, -1, coefs.shape[-1])

        # Calculate power in each frequency band
        power_spec = coefs.abs().pow(2)

        locomotion_power = power_spec[:, :, self.freq_locomotion_mask, :].mean(dim=2)
        fog_power = power_spec[:, :, self.freq_fog_mask, :].mean(dim=2)

        return {
            "locomotion_power": locomotion_power,  # 0.5-3 Hz
            "fog_power": fog_power,  # 3-8 Hz
            "freeze_index": fog_power / (locomotion_power + 1e-8),  # FOG indicator
        }


class STFTTransform(nn.Module):
    def __init__(
        self,
        win_length: int,
        hop_length: int,
        n_fft: int,
        sample_rate: int,
        device: str,
        img_size: int = 224,
        interpolate: bool = False,
        **kwargs,
    ):
        super().__init__()
        # Expect processor_config to include: img_size, win_length, hop_length, n_fft, sample_rate, device
        self.img_size = img_size
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.interpolate = interpolate
        # Precompute the Hann window on the correct device
        self.register_buffer("window", torch.hann_window(self.win_length))

        # Calculate how many frequency bins to keep
        # freq_bin_width = sample_rate / n_fft (in Hz per bin)
        max_freq = kwargs.get('max_freq', None)
        if max_freq is not None:
            bins_to_keep = int(max_freq * self.n_fft / self.sample_rate) + 1
        else:
            bins_to_keep = self.n_fft // 2 + 1  # keep all bins up to Nyquist
        self.freq_bins_to_keep = min(bins_to_keep, self.n_fft // 2 + 1)

    def forward(self, x):
        """
        x: (batch_size, channels, seq_len) on device='cuda' (or whatever you pass)
        returns: (batch_size, channels, img_size, img_size) real-valued tensor
        """
        batch_size, channels, seq_len = x.shape
        
        # Validate parameters against input length
        if self.win_length >= seq_len:
            # Adjust window length to be smaller than input
            effective_win_length = min(self.win_length, seq_len - 1)
            effective_window = torch.hann_window(effective_win_length, device=x.device)
        else:
            effective_win_length = self.win_length
            effective_window = self.window.to(x.device)
            
        # Ensure n_fft is not smaller than window length
        effective_n_fft = max(self.n_fft, effective_win_length)

        # Ensure n_fft is not larger than the sequence length for STFT compatibility
        if effective_n_fft > seq_len:
            effective_n_fft = seq_len
        
        x_flat = x.view(-1, seq_len)
        stft = torch.stft(
            x_flat,
            n_fft=effective_n_fft,
            hop_length=self.hop_length,
            win_length=effective_win_length,
            window=effective_window,
            center=False,  # Disable center padding to avoid the error
            return_complex=True,
        )

        freq_bins = effective_n_fft // 2 + 1
        time_steps = stft.shape[-1]
        stft = stft.view(batch_size, channels, freq_bins, time_steps)
        magnitude = torch.abs(stft)  # shape: (B, C, freq_bins, time_steps)

        eps = 1e-6
        mag_db = 20.0 * torch.log10(magnitude.clamp(min=eps))

        global_max = mag_db.amax(dim=[-2, -1], keepdim=True)  # shape (B, C, 1, 1)
        lower_bound = global_max - 80.0
        mag_db_clamped = torch.clamp(mag_db, min=lower_bound, max=global_max)

        mag_db_cropped = mag_db_clamped[:, :, : self.freq_bins_to_keep, :]

        B, C, F_crop, T = mag_db_cropped.shape
        mag_db_cropped = mag_db_cropped.view(
            -1, 1, F_crop, T
        ).float()  # shape (B*C, 1, F_crop, T)

        if self.interpolate:
            stft_resized = F.interpolate(
                mag_db_cropped,
                size=(self.img_size, self.img_size),
                mode="bicubic",
                align_corners=False,
            )

            stft_resized = stft_resized.view(
                batch_size, channels, self.img_size, self.img_size
            )
            
            
            return stft_resized
        
        else:
            # If not interpolating, return the cropped STFT
            stft_cropped = mag_db_cropped.view(
                batch_size, channels, F_crop, T
            ).float()
            
            return stft_cropped
            


class RawSignalTransform(nn.Module):
    def __init__(self, img_size: int, interpolate: bool = False, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.interpolate = interpolate

    @torch.no_grad()
    def forward(self, x):
        x = x.unsqueeze(2)  # [B, C, F, T]
        if self.interpolate:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

        return x


class FogTransform(nn.Module):
    def __init__(self, subtransforms: List[nn.Module]):
        super().__init__()
        self.subtransforms = nn.ModuleList(subtransforms)

    @torch.no_grad()
    def forward(self, x):
        for transform in self.subtransforms:
            x = transform(x)
        return x


TRANSFORMS_MAP = {
    "Integrator": Integrator,
    "Differentiator": Differentiator,
    "MelSpectrogramTransform": MelSpectrogramTransform,
    "WaveletTransform": WaveletTransform,
    "STFTTransform": STFTTransform,
    "RawSignalTransform": RawSignalTransform,
    "FogTransform": FogTransform,
    "QTransform": QTransform,
    "HilbertTransform": HilbertTransform,
    "PatchTransform": PatchTransform,
    "GramianAngularFieldTransform": GramianAngularFieldTransform,
    "GAFifier": GramianAngularFieldTransform,
}
