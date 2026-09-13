"""GPU-optimized signal augmentors with fully vectorized operations.

All augmentors operate on batched tensors without per-sample loops.
Random operations use torch RNG and stay on GPU throughout.
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Base Classes and Utilities
# =============================================================================

class BaseAugmentor(nn.Module):
    """Base class for all augmentors with vectorized probability handling."""

    def __init__(self, probability: float):
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation only to probabilistically selected samples.

        Args:
            x: Input tensor [batch_size, channels, seq_len] or [batch_size, channels, H, W]

        Returns:
            Augmented tensor with same shape as input
        """
        if self.probability <= 0:
            return x
        if self.probability >= 1:
            return self.augment(x)

        # Only compute augmentation on selected samples (saves compute when p < 1)
        mask = torch.rand(x.shape[0], device=x.device) < self.probability
        if not mask.any():
            return x

        x_selected = x[mask]
        x_aug = self.augment(x_selected)

        x_out = x.clone()
        x_out[mask] = x_aug
        return x_out

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Implement augmentation logic. Must be fully vectorized."""
        raise NotImplementedError


def generate_sinusoids(batch_size: int, seq_len: int, freq_range: Tuple[float, float],
                       amp_range: Tuple[float, float], sampling_rate: float,
                       device: torch.device) -> torch.Tensor:
    """Generate batch of sinusoidal signals with random frequencies and amplitudes.

    Args:
        batch_size: Number of signals to generate
        seq_len: Length of each signal
        freq_range: (min_freq, max_freq) in Hz
        amp_range: (min_amp, max_amp) amplitude range
        sampling_rate: Sampling rate in Hz
        device: Target device

    Returns:
        [B, seq_len] tensor of sinusoidal signals
    """
    # Sample frequencies and amplitudes
    freqs = torch.rand(batch_size, 1, device=device) * (freq_range[1] - freq_range[0]) + freq_range[0]
    amps = torch.rand(batch_size, 1, device=device) * (amp_range[1] - amp_range[0]) + amp_range[0]
    phases = torch.rand(batch_size, 1, device=device) * 2 * math.pi

    # Generate time vector
    t = torch.linspace(0, seq_len / sampling_rate, seq_len, device=device).unsqueeze(0)  # [1, seq_len]

    # Generate sinusoids: amp * sin(2π * freq * t + phase)
    signals = amps * torch.sin(2 * math.pi * freqs * t + phases)  # [B, seq_len]

    return signals


# =============================================================================
# Signal Augmentors (operate on raw time series)
# =============================================================================

class LRFlipAugmentor(BaseAugmentor):
    """Flip medio-lateral (channel 1) acceleration."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.5))

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, 1, :] *= -1
        return x


class NoiseAugmentor(BaseAugmentor):
    """Add Gaussian and impulse noise."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("noise_probability", 0.5))
        self.noise_level = augmentor_config.get("noise_level", 0.05)
        self.impulse_probability = augmentor_config.get("impulse_probability", 0.2)
        self.impulse_mask_probability = augmentor_config.get("impulse_mask_probability", 0.01)
        self.impulse_level = augmentor_config.get("impulse_level", 0.2)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian noise (per-sample varying level)
        noise_scale = torch.rand(x.shape[0], 1, 1, device=x.device) * self.noise_level
        x = x + torch.randn_like(x) * noise_scale

        # Impulse noise (vectorized per-sample decision)
        apply_impulse = torch.rand(x.shape[0], device=x.device) < self.impulse_probability
        if apply_impulse.any():
            impulse_mask = torch.rand_like(x) < self.impulse_mask_probability
            impulse_values = torch.randn_like(x) * self.impulse_level
            # Only apply to samples where apply_impulse is True
            impulse_mask = impulse_mask & apply_impulse.view(-1, 1, 1)
            x = torch.where(impulse_mask, x + impulse_values, x)

        return x


class MultiplicativeNoiseAugmentor(BaseAugmentor):
    """Apply multiplicative noise with multiple components."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 1.0))
        self.amp_gn = augmentor_config.get("amp_gn", 0.3)
        self.amp_ch_gn = augmentor_config.get("amp_ch_gn", 0.1)
        self.amp_gna = augmentor_config.get("amp_gna", 0.1)
        self.amp_ch_gna = augmentor_config.get("amp_ch_gna", 0.03)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        device = x.device

        # Global noise (broadcast across all dims)
        global_noise = torch.randn(B, 1, 1, device=device) * self.amp_gn

        # Channel-specific noise
        channel_noise = torch.randn(B, C, 1, device=device) * self.amp_ch_gn

        # Time-varying noise (cumulative, normalized)
        time_noise = torch.randn(B, 1, L, device=device) * self.amp_gna
        time_noise = torch.cumsum(time_noise, dim=2) / math.sqrt(L)

        # Channel-and-time specific noise
        channel_time_noise = torch.randn(B, C, L, device=device) * self.amp_ch_gna
        channel_time_noise = torch.cumsum(channel_time_noise, dim=2) / math.sqrt(L)

        # Combine and apply multiplicatively
        total_noise = global_noise + channel_noise + time_noise + channel_time_noise
        return x * torch.exp(total_noise)


class ResizeAugmentor(BaseAugmentor):
    """Resize time series with random temporal scaling (per-sample)."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.9))
        self.stretch_std = augmentor_config.get("stretch_std", 0.5)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape

        # Sample scaling factors from log-normal per sample
        ratios = torch.exp(torch.randn(B, device=x.device) * self.stretch_std)

        # Use grid_sample for batched per-sample interpolation
        # grid_sample expects [B, H_out, W_out, 2] grid, input [B, C, H, W]
        # We treat signal as [B, C, 1, L] image
        x_4d = x.unsqueeze(2)  # [B, C, 1, L]

        # For each sample, map output positions to input positions via ratio
        # Output position i maps to input position i / ratio (stretch) or i * ratio (compress)
        # Random crop offset for stretched signals
        crop_offsets = torch.rand(B, device=x.device)  # [B] uniform in [0, 1]

        # Build per-sample grids
        # For ratio > 1 (stretched signal longer than L): we crop a window of size L/ratio from [0,1]
        # For ratio < 1 (compressed signal shorter than L): mapped range exceeds [0,1], grid_sample zeros out
        # Normalized input coordinates: center of signal = 0, range [-1, 1]
        out_coords = torch.linspace(0, 1, L, device=x.device).unsqueeze(0).expand(B, -1)  # [B, L]

        # Map output coordinates to input coordinates
        # input_coord = (out_coord / ratio) + offset * (1 - 1/ratio)
        inv_ratios = (1.0 / ratios).unsqueeze(1)  # [B, 1]
        offsets = (crop_offsets * (1.0 - inv_ratios.squeeze(1))).unsqueeze(1)  # [B, 1]
        input_coords = out_coords * inv_ratios + offsets  # [B, L] in [0, 1] for stretched

        # Convert to grid_sample coordinates: [-1, 1]
        grid_x = input_coords * 2 - 1  # [B, L]
        grid_y = torch.zeros_like(grid_x)  # [B, L] - always sample from y=0

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)  # [B, 1, L, 2]

        x_out = F.grid_sample(x_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return x_out.squeeze(2)  # [B, C, L]


class CropAugmentor(BaseAugmentor):
    """Crop from start or end with zero padding."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.3))
        self.max_crop_fraction = augmentor_config.get("max_crop_fraction", 0.9)
        self.min_fraction = augmentor_config.get("min_fraction", 0.1)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape

        # Sample crop lengths and positions for each sample
        min_len = int(L * self.min_fraction)
        max_len = int(L * self.max_crop_fraction)
        crop_lengths = torch.randint(min_len, max_len + 1, (B,), device=x.device)

        # Random choice: crop from start or end
        from_end = torch.rand(B, device=x.device) < 0.5

        # Build position mask using broadcasting
        positions = torch.arange(L, device=x.device).unsqueeze(0)  # [1, L]

        # For from_end=True: keep positions >= (L - crop_len)
        # For from_end=False: keep positions < crop_len
        end_mask = positions >= (L - crop_lengths.unsqueeze(1))  # [B, L]
        start_mask = positions < crop_lengths.unsqueeze(1)  # [B, L]

        keep_mask = torch.where(from_end.unsqueeze(1), end_mask, start_mask)  # [B, L]
        keep_mask = keep_mask.unsqueeze(1).expand_as(x)  # [B, C, L]

        return x * keep_mask


class ExciseAugmentor(BaseAugmentor):
    """Remove a segment and insert zeros."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.1))
        self.max_excise_fraction = augmentor_config.get("max_excise_fraction", 0.3)
        self.min_gap_length = augmentor_config.get("min_gap_length", 50)
        self.max_gap_length = augmentor_config.get("max_gap_length", 150)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape

        # Sample parameters per sample
        max_excise = int(L * self.max_excise_fraction)
        excise_lengths = torch.randint(1, max_excise + 1, (B,), device=x.device)
        gap_lengths = torch.randint(self.min_gap_length, self.max_gap_length + 1, (B,), device=x.device)
        max_starts = (L - excise_lengths).clamp(min=0)
        excise_starts = (torch.rand(B, device=x.device) * (max_starts.float() + 1)).long().clamp(max=max_starts)

        # Build index mapping: output position -> input position (or -1 for gap/zero)
        # Layout: [before | gap | after | padding]
        # before: positions [0, excise_start)  -> input [0, excise_start)
        # gap: positions [excise_start, excise_start + gap_len) -> zero
        # after: positions [excise_start + gap_len, ...) -> input [excise_start + excise_len, ...)

        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)  # [B, L]
        excise_s = excise_starts.unsqueeze(1)  # [B, 1]
        excise_l = excise_lengths.unsqueeze(1)  # [B, 1]
        gap_l = gap_lengths.unsqueeze(1)  # [B, 1]

        # Regions
        is_before = positions < excise_s  # [B, L]
        is_gap = (positions >= excise_s) & (positions < excise_s + gap_l)  # [B, L]
        is_after = positions >= excise_s + gap_l  # [B, L]

        # Map output positions to input positions
        # Before: identity mapping
        # After: map to input[excise_start + excise_len + (pos - excise_start - gap_len)]
        after_input_pos = (positions - gap_l + excise_l)  # [B, L]

        # Use before positions as-is, after positions remapped, clamp to valid range
        input_indices = torch.where(is_before, positions, after_input_pos)
        input_indices = input_indices.clamp(0, L - 1)  # [B, L]

        # Mask for valid positions (before and after that map to valid input)
        valid = is_before | (is_after & (after_input_pos < L) & (after_input_pos >= 0))

        # Gather and zero out gap/invalid positions
        input_indices_expanded = input_indices.unsqueeze(1).expand_as(x)  # [B, C, L]
        x_out = torch.gather(x, 2, input_indices_expanded)

        # Zero out gap and overflow positions
        zero_mask = (~valid).unsqueeze(1).expand_as(x)  # [B, C, L]
        x_out[zero_mask] = 0

        return x_out


class TimeMaskingAugmentor(BaseAugmentor):
    """Mask random time segments (set to zero)."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("time_masking_probability", 0.5))
        mask_range = augmentor_config.get("time_masking_length_low_high", [10, 50])
        self.min_mask_len = mask_range[0]
        self.max_mask_len = mask_range[1]
        self.channel_probability = augmentor_config.get("channel_mask_probability", 0.7)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        x = x.clone()

        # Sample mask parameters per sample
        mask_lengths = torch.randint(self.min_mask_len, self.max_mask_len + 1, (B,), device=x.device)
        mask_lengths = mask_lengths.clamp(max=L - 1)
        mask_starts = (torch.rand(B, device=x.device) * (L - mask_lengths).float()).long()

        # Per-sample, per-channel masking decision [B, C]
        channel_mask = torch.rand(B, C, device=x.device) < self.channel_probability

        # Create time-position mask using arange broadcasting
        # positions [1, L], starts [B, 1], ends [B, 1]
        positions = torch.arange(L, device=x.device).unsqueeze(0)  # [1, L]
        starts = mask_starts.unsqueeze(1)  # [B, 1]
        ends = (mask_starts + mask_lengths).unsqueeze(1)  # [B, 1]
        time_mask = (positions >= starts) & (positions < ends)  # [B, L]

        # Combine: [B, C, L] = [B, C, 1] & [B, 1, L]
        full_mask = channel_mask.unsqueeze(2) & time_mask.unsqueeze(1)
        x[full_mask] = 0

        return x


class PhaseShiftAugmentor(BaseAugmentor):
    """Apply temporal phase shift (circular roll)."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("phase_shift_probability", 0.2))
        shift_range = augmentor_config.get("phase_shift_range_low_high", [-10, 11])
        self.min_shift = shift_range[0]
        self.max_shift = shift_range[1]
        self.channel_probability = augmentor_config.get("phase_shift_channel_probability", 0.5)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape

        # Per-sample, per-channel shift amounts [B, C]
        shifts = torch.randint(self.min_shift, self.max_shift, (B, C), device=x.device)

        # Per-channel application mask [B, C]
        channel_mask = torch.rand(B, C, device=x.device) < self.channel_probability
        shifts = shifts * channel_mask.long()  # Zero out shifts for non-selected channels

        # Create gather indices: (arange - shift) % L for each [B, C]
        positions = torch.arange(L, device=x.device).view(1, 1, L).expand(B, C, L)  # [B, C, L]
        # Shift indices: position - shift (circular roll = gather from shifted positions)
        shifted = (positions - shifts.unsqueeze(2)) % L  # [B, C, L]

        return torch.gather(x, 2, shifted)


class TimeWarpingAugmentor(BaseAugmentor):
    """Apply non-linear time warping (per-sample power-law warping)."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.4))
        warp_range = augmentor_config.get("warp_factor_range", [0.9, 1.1])
        self.warp_min = warp_range[0]
        self.warp_max = warp_range[1]

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape

        # Sample warp factors per sample [B]
        warp_factors = torch.rand(B, device=x.device) * (self.warp_max - self.warp_min) + self.warp_min

        # Create power-law warped coordinates for each sample
        # orig_steps [1, L] in (0, 1], warp_factors [B, 1]
        orig_steps = torch.linspace(1e-6, 1, L, device=x.device).unsqueeze(0)  # [1, L]
        warped = torch.pow(orig_steps, warp_factors.unsqueeze(1))  # [B, L]
        # Normalize each sample to [0, 1]
        warped = warped / warped[:, -1:].clamp(min=1e-8)

        # Convert to grid_sample coordinates [-1, 1]
        grid_x = warped * 2 - 1  # [B, L]
        grid_y = torch.zeros_like(grid_x)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)  # [B, 1, L, 2]

        x_4d = x.unsqueeze(2)  # [B, C, 1, L]
        x_out = F.grid_sample(x_4d, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return x_out.squeeze(2)


class TimeStretchAugmentor(BaseAugmentor):
    """Stretch or compress segments temporally (from TemporalAugmentor)."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("time_wraping_probability", 0.3))
        self.stretch_base = augmentor_config.get("strech_factor_base", 0.9)
        self.stretch_slope = augmentor_config.get("strech_factor_slope", 0.2)
        self.channel_probability = augmentor_config.get("time_wrapping_channel_probability", 0.5)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape

        # Per-sample stretch factors (apply same factor to all channels per sample)
        stretch_factors = self.stretch_base + self.stretch_slope * torch.rand(B, device=x.device)

        # Per-channel application mask [B, C]
        channel_mask = torch.rand(B, C, device=x.device) < self.channel_probability

        # Use grid_sample: stretch creates a linear mapping of coordinates
        # stretch_factor > 1 = stretch (read from smaller input range)
        # stretch_factor < 1 = compress (read beyond input range, border-padded)
        orig_coords = torch.linspace(-1, 1, L, device=x.device).unsqueeze(0)  # [1, L]
        # Stretched coords: divide by stretch_factor to read from compressed range
        stretched_coords = orig_coords / stretch_factors.unsqueeze(1)  # [B, L]

        grid_x = stretched_coords
        grid_y = torch.zeros_like(grid_x)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)  # [B, 1, L, 2]

        x_4d = x.unsqueeze(2)  # [B, C, 1, L]
        x_stretched = F.grid_sample(x_4d, grid, mode='bilinear', padding_mode='border', align_corners=True)
        x_stretched = x_stretched.squeeze(2)  # [B, C, L]

        # Apply channel mask: use stretched for selected channels, original for others
        mask_expanded = channel_mask.unsqueeze(2).expand_as(x)  # [B, C, L]
        return torch.where(mask_expanded, x_stretched, x)


class SignalCutMixAugmentor(BaseAugmentor):
    """Mix segments from different samples in batch."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.3))
        self.segment_probability = augmentor_config.get("segment_probability", 0.5)
        self.min_cutmix_fraction = augmentor_config.get("min_cutmix_fraction", 0.3)
        self.max_cutmix_fraction = augmentor_config.get("max_cutmix_fraction", 0.7)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape

        if B <= 1:
            return x

        x = x.clone()

        # Per-sample application decision
        apply_mask = torch.rand(B, device=x.device) < self.segment_probability
        if not apply_mask.any():
            return x

        # Donor indices (circular offset to avoid self-mixing)
        offsets = torch.randint(1, B, (B,), device=x.device)
        donor_indices = (torch.arange(B, device=x.device) + offsets) % B

        # Random segment parameters per sample
        min_len = int(L * self.min_cutmix_fraction)
        max_len = int(L * self.max_cutmix_fraction)
        segment_lengths = torch.randint(min_len, max_len + 1, (B,), device=x.device)
        max_starts = (L - segment_lengths).clamp(min=0)
        start_indices = (torch.rand(B, device=x.device) * (max_starts.float() + 1)).long().clamp(max=max_starts)

        # Create segment mask [B, 1, L] using broadcasting
        positions = torch.arange(L, device=x.device).unsqueeze(0)  # [1, L]
        starts = start_indices.unsqueeze(1)  # [B, 1]
        ends = (start_indices + segment_lengths).unsqueeze(1)  # [B, 1]
        segment_mask = (positions >= starts) & (positions < ends)  # [B, L]

        # Combine with apply_mask: [B, L]
        full_mask = apply_mask.unsqueeze(1) & segment_mask  # [B, L]
        full_mask = full_mask.unsqueeze(1).expand_as(x)  # [B, C, L]

        # Apply cutmix
        x[full_mask] = x[donor_indices][full_mask]

        return x


class TremorAugmentor(BaseAugmentor):
    """Add Parkinson's tremor (4-6Hz oscillations)."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.3))
        self.tremor_freq_range = tuple(augmentor_config.get("tremor_freq_range", [4.0, 6.0]))
        self.tremor_amplitude_range = tuple(augmentor_config.get("tremor_amplitude_range", [0.1, 0.3]))
        self.tremor_duration_range = tuple(augmentor_config.get("tremor_duration_range", [0.3, 0.8]))
        self.sampling_rate = 100.0

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        device = x.device

        # Sample durations per sample
        duration_fractions = (torch.rand(B, device=device) *
                            (self.tremor_duration_range[1] - self.tremor_duration_range[0]) +
                            self.tremor_duration_range[0])
        tremor_lengths = (L * duration_fractions).long().clamp(min=1)

        # Sample start positions
        max_starts = (L - tremor_lengths).clamp(min=0)
        start_indices = (torch.rand(B, device=device) * (max_starts.float() + 1)).long().clamp(max=max_starts)

        # Generate full-length tremor signals for all samples [B, L]
        tremor_signals = generate_sinusoids(
            B, L, self.tremor_freq_range, self.tremor_amplitude_range,
            self.sampling_rate, device
        )

        # Create position mask for where tremor is active [B, L]
        positions = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        starts = start_indices.unsqueeze(1)  # [B, 1]
        ends = (start_indices + tremor_lengths).unsqueeze(1)  # [B, 1]
        active_mask = (positions >= starts) & (positions < ends)  # [B, L]

        # Zero out tremor outside active region
        tremor_signals = tremor_signals * active_mask.float()  # [B, L]

        # Per-channel scaling factors [B, C]
        channel_factors = 0.7 + 0.6 * torch.rand(B, C, device=device)

        # Apply: [B, C, L] = [B, C, 1] * [B, 1, L]
        tremor_contribution = channel_factors.unsqueeze(2) * tremor_signals.unsqueeze(1)

        return x + tremor_contribution


class BradykinesiaAugmentor(BaseAugmentor):
    """Simulate bradykinesia (movement slowness) via time stretching."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.4))
        self.slowdown_range = tuple(augmentor_config.get("slowdown_range", [0.6, 0.9]))
        self.segment_fraction_range = tuple(augmentor_config.get("segment_fraction_range", [0.3, 0.7]))

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        device = x.device

        # Sample segment parameters per sample
        segment_fractions = (torch.rand(B, device=device) *
                           (self.segment_fraction_range[1] - self.segment_fraction_range[0]) +
                           self.segment_fraction_range[0])
        segment_lengths = (L * segment_fractions).long().clamp(min=1)
        max_starts = (L - segment_lengths).clamp(min=0)
        start_indices = (torch.rand(B, device=device) * (max_starts.float() + 1)).long().clamp(max=max_starts)
        end_indices = start_indices + segment_lengths

        # Slowdown factors per sample
        slowdowns = (torch.rand(B, device=device) *
                    (self.slowdown_range[1] - self.slowdown_range[0]) +
                    self.slowdown_range[0])

        # Build a grid that applies identity mapping outside the segment
        # and stretched mapping inside the segment
        # Use grid_sample: for each output position, compute where to read from input
        positions = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)  # [B, L]
        starts_f = start_indices.unsqueeze(1).float()  # [B, 1]
        ends_f = end_indices.unsqueeze(1).float()  # [B, 1]
        seg_lens_f = segment_lengths.unsqueeze(1).float()  # [B, 1]

        # Inside segment: remap positions to read from stretched range
        # Normalized position within segment: (pos - start) / seg_len in [0, 1]
        seg_pos = (positions - starts_f) / seg_lens_f.clamp(min=1)  # [B, L]

        # Apply slowdown: read from a compressed range (slowdown < 1 = slower = stretched)
        # stretched_pos = seg_pos * slowdown maps to compressed input range
        stretched_seg_pos = seg_pos * slowdowns.unsqueeze(1)  # [B, L]

        # Map back to absolute input coordinates
        input_pos_segment = starts_f + stretched_seg_pos * seg_lens_f  # [B, L]

        # Mask for segment region
        is_segment = (positions >= starts_f) & (positions < ends_f)  # [B, L]

        # Final input coordinates: identity outside segment, stretched inside
        input_coords = torch.where(is_segment, input_pos_segment, positions)

        # Normalize to [-1, 1] for grid_sample
        grid_x = input_coords / (L - 1) * 2 - 1  # [B, L]
        grid_y = torch.zeros_like(grid_x)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)  # [B, 1, L, 2]

        x_4d = x.unsqueeze(2)  # [B, C, 1, L]
        x_out = F.grid_sample(x_4d, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return x_out.squeeze(2)


class FoGSynthesisAugmentor(BaseAugmentor):
    """Synthesize artificial FoG episodes."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.2))
        self.fog_duration_range = tuple(augmentor_config.get("fog_duration_range", [1.0, 4.0]))
        self.fog_freq_range = tuple(augmentor_config.get("fog_freq_range", [3.0, 8.0]))
        self.fog_amplitude_range = tuple(augmentor_config.get("fog_amplitude_range", [0.2, 0.5]))
        self.sampling_rate = 100.0

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        device = x.device

        # Sample durations per sample
        fog_durations = (torch.rand(B, device=device) *
                        (self.fog_duration_range[1] - self.fog_duration_range[0]) +
                        self.fog_duration_range[0])
        fog_lengths = (fog_durations * self.sampling_rate).long().clamp(min=1, max=L)

        # Sample start positions
        max_starts = (L - fog_lengths).clamp(min=0)
        start_indices = (torch.rand(B, device=device) * (max_starts.float() + 1)).long().clamp(max=max_starts)

        # Generate multi-component FoG signals (use fixed 4 components, sum them)
        # Each component is a full-length sinusoid [B, L], we'll mask to active region
        num_components = 4  # Use max components (was 2-5 random)
        fog_signal = torch.zeros(B, L, device=device)
        for _ in range(num_components):
            component = generate_sinusoids(
                B, L, self.fog_freq_range, self.fog_amplitude_range,
                self.sampling_rate, device
            )  # [B, L]
            fog_signal = fog_signal + component

        # Create active region mask [B, L]
        positions = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        starts = start_indices.unsqueeze(1)  # [B, 1]
        ends = (start_indices + fog_lengths).unsqueeze(1)  # [B, 1]
        active_mask = (positions >= starts) & (positions < ends)  # [B, L]

        # Apply fade envelope within active region
        # Compute normalized position within FoG region [0, 1]
        fog_pos = (positions.float() - starts.float()) / fog_lengths.unsqueeze(1).float().clamp(min=1)
        fade_frac = 0.1  # 10% fade in/out
        fade_in_env = (fog_pos / fade_frac).clamp(0, 1)
        fade_out_env = ((1.0 - fog_pos) / fade_frac).clamp(0, 1)
        envelope = fade_in_env * fade_out_env  # [B, L]

        fog_signal = fog_signal * active_mask.float() * envelope  # [B, L]

        # Channel scaling: vertical=1.5, ML=1.0, AP=0.7
        channel_scales = torch.tensor([1.5, 1.0, 0.7], device=device)[:C]  # [C]
        # [B, C, L] = [1, C, 1] * [B, 1, L]
        fog_contribution = channel_scales.view(1, -1, 1) * fog_signal.unsqueeze(1)

        # Replace (not add) in FoG region, add elsewhere stays as original
        # Original behavior was replacement (=), so we replace in active region
        active_mask_3d = active_mask.unsqueeze(1).expand_as(x)  # [B, C, L]
        x_out = x.clone()
        x_out[active_mask_3d] = fog_contribution[active_mask_3d]

        return x_out


class GaitAsymmetryAugmentor(BaseAugmentor):
    """Simulate gait asymmetry via modulation."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.4))
        self.asymmetry_freq_range = tuple(augmentor_config.get("asymmetry_freq_range", [0.5, 2.0]))
        self.asymmetry_strength_range = tuple(augmentor_config.get("asymmetry_strength_range", [0.1, 0.3]))
        self.sampling_rate = 100.0

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        device = x.device

        # Generate modulation signals for batch
        modulations = generate_sinusoids(
            B, L, self.asymmetry_freq_range, self.asymmetry_strength_range,
            self.sampling_rate, device
        )  # [B, L]

        x = x.clone()

        # Apply to medio-lateral channel (1) primarily
        if C > 1:
            x[:, 1, :] = x[:, 1, :] * (1.0 + modulations)

        # Slightly affect vertical channel
        if C > 0:
            x[:, 0, :] = x[:, 0, :] * (1.0 + 0.3 * modulations)

        return x


class DeviceOrientationAugmentor(BaseAugmentor):
    """Simulate device orientation changes via 2D rotation."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.3))
        rot_range = augmentor_config.get("rotation_angle_range", [-30, 30])
        self.min_angle = rot_range[0]
        self.max_angle = rot_range[1]

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        device = x.device

        if C < 2:
            return x

        # Sample rotation angles per sample (in radians)
        angles_deg = torch.rand(B, device=device) * (self.max_angle - self.min_angle) + self.min_angle
        angles_rad = angles_deg * math.pi / 180.0

        # Create rotation matrices [B, 2, 2]
        cos_a = torch.cos(angles_rad)  # [B]
        sin_a = torch.sin(angles_rad)  # [B]

        # Build rotation matrix for each sample
        # R = [[cos, -sin], [sin, cos]]
        rotation_matrices = torch.stack([
            torch.stack([cos_a, -sin_a], dim=1),
            torch.stack([sin_a, cos_a], dim=1)
        ], dim=1)  # [B, 2, 2]

        # Apply rotation to first two channels
        # xy_channels: [B, 2, L]
        # rotation_matrices: [B, 2, 2]
        # Result: [B, 2, L]
        xy_channels = x[:, :2, :]  # [B, 2, L]
        xy_rotated = torch.bmm(rotation_matrices, xy_channels)  # [B, 2, L]

        x = x.clone()
        x[:, :2, :] = xy_rotated

        return x


# =============================================================================
# Spectral Augmentors (operate on 2D spectrograms/wavelets)
# =============================================================================

class SpectralNoiseAugmentor(BaseAugmentor):
    """Add calibrated noise to spectral representations."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.4))
        noise_range = augmentor_config.get("noise_level_range", [0.01, 0.05])
        self.noise_min = noise_range[0]
        self.noise_max = noise_range[1]

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        # Sample noise levels per sample
        sigma = torch.rand(x.shape[0], 1, 1, 1, device=x.device) * (self.noise_max - self.noise_min) + self.noise_min

        # Calculate signal power and scale noise
        signal_power = torch.mean(x ** 2, dim=(1, 2, 3), keepdim=True)
        noise_power = signal_power * sigma
        noise = torch.randn_like(x) * torch.sqrt(noise_power)

        return x + noise


class FrequencyMaskingAugmentor(BaseAugmentor):
    """Mask random frequency bands."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.4))
        self.max_masks = augmentor_config.get("max_masks", 2)
        self.min_mask_fraction = augmentor_config.get("min_mask_fraction", 0.05)
        self.max_mask_fraction = augmentor_config.get("max_mask_fraction", 0.2)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.clone()

        mask_min_height = int(H * self.min_mask_fraction)
        mask_max_height = int(H * self.max_mask_fraction)

        for b in range(B):
            num_masks = torch.randint(1, self.max_masks + 1, (1,), device=x.device).item()

            for _ in range(num_masks):
                mask_height = torch.randint(mask_min_height, mask_max_height + 1, (1,), device=x.device).item()
                mask_start = torch.randint(0, H - mask_height + 1, (1,), device=x.device).item()
                x[b, :, mask_start:mask_start + mask_height, :] = 0

        return x


class TimeMaskingAugmentorSpectral(BaseAugmentor):
    """Mask random time segments in spectrograms."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.4))
        self.max_masks = augmentor_config.get("max_masks", 2)
        self.min_mask_fraction = augmentor_config.get("min_mask_fraction", 0.05)
        self.max_mask_fraction = augmentor_config.get("max_mask_fraction", 0.2)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.clone()

        mask_min_width = int(W * self.min_mask_fraction)
        mask_max_width = int(W * self.max_mask_fraction)

        for b in range(B):
            num_masks = torch.randint(1, self.max_masks + 1, (1,), device=x.device).item()

            for _ in range(num_masks):
                mask_width = torch.randint(mask_min_width, mask_max_width + 1, (1,), device=x.device).item()
                mask_start = torch.randint(0, W - mask_width + 1, (1,), device=x.device).item()
                x[b, :, :, mask_start:mask_start + mask_width] = 0

        return x


class SpectralMixupAugmentor(BaseAugmentor):
    """Mixup augmentation for spectrograms."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.3))
        self.alpha = augmentor_config.get("alpha", 0.2)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        if B <= 1:
            return x

        # Generate random permutation
        indices = torch.randperm(B, device=x.device)

        # Approximate beta distribution with torch (using two gamma samples)
        # Beta(α, α) ≈ Gamma(α, 1) / (Gamma(α, 1) + Gamma(α, 1))
        # For simplicity, use uniform distribution centered around 0.5 for mixup
        # This is a common approximation when alpha is small
        lam = torch.rand(B, 1, 1, 1, device=x.device) * 0.4 + 0.3  # Range [0.3, 0.7]

        # Apply mixup
        mixed = lam * x + (1 - lam) * x[indices]

        return mixed


class SpectralCutMixAugmentor(BaseAugmentor):
    """CutMix augmentation for spectrograms."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.3))
        self.sample_probability = augmentor_config.get("sample_probability", 0.5)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        if B <= 1:
            return x

        x = x.clone()

        for b in range(B):
            if torch.rand(1, device=x.device).item() < self.sample_probability:
                other_idx = (b + torch.randint(1, B, (1,), device=x.device).item()) % B

                # Random rectangle
                cut_h = torch.randint(int(H * 0.1), int(H * 0.4) + 1, (1,), device=x.device).item()
                cut_w = torch.randint(int(W * 0.1), int(W * 0.4) + 1, (1,), device=x.device).item()
                cut_h_start = torch.randint(0, H - cut_h + 1, (1,), device=x.device).item()
                cut_w_start = torch.randint(0, W - cut_w + 1, (1,), device=x.device).item()

                # Apply cutmix
                x[b, :, cut_h_start:cut_h_start + cut_h, cut_w_start:cut_w_start + cut_w] = \
                    x[other_idx, :, cut_h_start:cut_h_start + cut_h, cut_w_start:cut_w_start + cut_w]

        return x


class ContrastJitterAugmentor(BaseAugmentor):
    """Adjust contrast of spectrograms."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.3))
        contrast_range = augmentor_config.get("contrast_range", [0.8, 1.2])
        self.contrast_min = contrast_range[0]
        self.contrast_max = contrast_range[1]

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Sample contrast factors per sample
        contrast_factors = (torch.rand(B, 1, 1, 1, device=x.device) *
                           (self.contrast_max - self.contrast_min) + self.contrast_min)

        # Calculate means per sample per channel
        means = torch.mean(x, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # Adjust contrast
        x = means + contrast_factors * (x - means)

        return x


class EdgeEmphasisAugmentor(BaseAugmentor):
    """Emphasize edges in spectrograms using convolution."""

    def __init__(self, augmentor_config: Dict):
        super().__init__(augmentor_config.get("probability", 0.3))
        strength_range = augmentor_config.get("strength_range", [0.1, 0.3])
        self.strength_min = strength_range[0]
        self.strength_max = strength_range[1]

        # Register edge kernels
        self.register_buffer(
            "edge_kernel_h",
            torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            "edge_kernel_v",
            torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Sample edge weights per sample
        edge_weights = (torch.rand(B, device=x.device) *
                       (self.strength_max - self.strength_min) + self.strength_min)

        x_out = x.clone()

        # Process each sample (edge detection requires per-sample processing)
        for b in range(B):
            for c in range(C):
                x_channel = x[b, c].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

                # Detect edges
                edges_h = F.conv2d(x_channel, self.edge_kernel_h, padding=1)
                edges_v = F.conv2d(x_channel, self.edge_kernel_v, padding=1)
                edges = torch.sqrt(edges_h ** 2 + edges_v ** 2)

                # Enhance with edges
                x_out[b, c] = x[b, c] + edge_weights[b] * edges.squeeze()

        return x_out


# =============================================================================
# Composite Augmenters
# =============================================================================

class SignalAugmenter(nn.Module):
    """Main signal augmentation pipeline (no grad, applies augmentors sequentially)."""

    def __init__(self, augmentors: List[nn.Module]):
        super().__init__()
        self.augmentors = nn.ModuleList(augmentors)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for augmentor in self.augmentors:
            x = augmentor(x)
        return x


class SpectralAugmenter(nn.Module):
    """Main spectral augmentation pipeline."""

    def __init__(self, augmentors: List[nn.Module]):
        super().__init__()
        self.augmentors = nn.ModuleList(augmentors)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for augmentor in self.augmentors:
            x = augmentor(x)
        return x


# =============================================================================
# Registry Maps
# =============================================================================

SIGNAL_AUGMENTORS_MAP = {
    "LRFlipAugmentor": LRFlipAugmentor,
    "NoiseAugmentor": NoiseAugmentor,
    "MultiplicativeNoiseAugmentor": MultiplicativeNoiseAugmentor,
    "ResizeAugmentor": ResizeAugmentor,
    "CropAugmentor": CropAugmentor,
    "ExciseAugmentor": ExciseAugmentor,
    "TimeMaskingAugmentor": TimeMaskingAugmentor,
    "PhaseShiftAugmentor": PhaseShiftAugmentor,
    "TimeWarpingAugmentor": TimeWarpingAugmentor,
    "TimeStretchAugmentor": TimeStretchAugmentor,
    "SignalCutMixAugmentor": SignalCutMixAugmentor,
    "TremorAugmentor": TremorAugmentor,
    "BradykinesiaAugmentor": BradykinesiaAugmentor,
    "FoGSynthesisAugmentor": FoGSynthesisAugmentor,
    "GaitAsymmetryAugmentor": GaitAsymmetryAugmentor,
    "DeviceOrientationAugmentor": DeviceOrientationAugmentor,
    "SignalAugmenter": SignalAugmenter,
}

SPECTRAL_AUGMENTORS_MAP = {
    "SpectralNoiseAugmentor": SpectralNoiseAugmentor,
    "FrequencyMaskingAugmentor": FrequencyMaskingAugmentor,
    "TimeMaskingAugmentor": TimeMaskingAugmentorSpectral,
    "SpectralMixupAugmentor": SpectralMixupAugmentor,
    "SpectralCutMixAugmentor": SpectralCutMixAugmentor,
    "ContrastJitterAugmentor": ContrastJitterAugmentor,
    "EdgeEmphasisAugmentor": EdgeEmphasisAugmentor,
    "SpectralAugmenter": SpectralAugmenter,
}
