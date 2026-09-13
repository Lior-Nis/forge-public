"""GPU-native Continuous Wavelet Transform.

Precomputes the integrated wavelet at init (CPU/numpy is fine),
stores everything as registered buffers, then the forward pass
uses only torch.fft — zero CPU round-trips.

Scales are grouped by their required FFT padding size so each
group does a single batched FFT multiply instead of N individual calls.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from pywt import ContinuousWavelet


def _next_fast_len(n: int) -> int:
    """Round up to nearest power of 2 for FFT efficiency."""
    return 1 << math.ceil(math.log2(max(n, 1)))


class GpuCwt(nn.Module):
    """Fully GPU-resident CWT with precomputed wavelet buffers.

    For a fixed set of scales and wavelet (typical: 64 Morlet scales),
    the integrated wavelet, per-scale sampling indices, trim offsets,
    and sqrt(scale) factors are deterministic constants.  We compute
    them once in ``__init__`` and register them as buffers so the
    forward pass is pure ``torch.fft``.
    """

    def __init__(
        self,
        scales: torch.Tensor,
        wavelet: str = "morl",
        precision: int = 10,
        max_scales_per_batch: int = 16,
    ):
        super().__init__()
        self.wavelet_name = wavelet
        self.precision = precision
        self.max_scales_per_batch = max_scales_per_batch

        scales_np = scales.detach().cpu().numpy().astype(np.float64)
        n_scales = len(scales_np)

        # --- 1. Compute integrated wavelet (same as ptwt) ----------------
        wav = ContinuousWavelet(wavelet)
        psi, x = wav.wavefun(precision)
        step = x[1] - x[0]
        int_psi = np.cumsum(psi) * step  # integrated wavelet
        int_psi_len = len(int_psi)
        x_range = x[-1] - x[0]

        # --- 2. For each scale: compute sampled, flipped int_psi ---------
        psi_samples = []  # list of 1-D numpy arrays (variable length)
        psi_lengths = np.zeros(n_scales, dtype=np.int64)
        sqrt_scales = np.sqrt(scales_np)
        trim_floor = np.zeros(n_scales, dtype=np.int64)
        trim_ceil = np.zeros(n_scales, dtype=np.int64)

        for i, scale in enumerate(scales_np):
            # Index into int_psi at scaled positions
            # Match ptwt: torch.arange(scale * x_range + 1) / (scale * step)
            j = np.arange(0, scale * x_range + 1, 1.0) / (scale * step)
            j = np.floor(j).astype(np.int64)

            # Trim indices that exceed int_psi length
            j = j[j < int_psi_len]

            # Sample and flip
            psi_scale = int_psi[j][::-1].copy()
            psi_samples.append(psi_scale)
            psi_lengths[i] = len(psi_scale)

        # --- 3. Pad all psi samples to max length, stack ------------------
        max_psi_len = int(psi_lengths.max())
        int_psi_scales = np.zeros((n_scales, max_psi_len), dtype=np.float64)
        for i, ps in enumerate(psi_samples):
            int_psi_scales[i, : len(ps)] = ps

        # --- 4. Precompute trim offsets (depend on data_len) --------------
        # These depend on data_len, so we store psi_lengths and compute
        # trim at forward time.  But sqrt_scales and psi are fixed.
        self.register_buffer(
            "int_psi_scales",
            torch.from_numpy(int_psi_scales).float(),
        )
        self.register_buffer(
            "psi_lengths", torch.from_numpy(psi_lengths).long()
        )
        self.register_buffer(
            "sqrt_scales", torch.from_numpy(sqrt_scales).float()
        )
        self.register_buffer("scales", scales.float())

        # Cache for FFT groups (keyed by data_len)
        self._group_cache: dict[int, list] = {}

    def _build_groups(self, data_len: int) -> list:
        """Group scales by their required FFT size for batched computation.

        Precomputes gather indices so the forward pass is fully vectorized.
        """
        n_scales = len(self.psi_lengths)
        psi_lens = self.psi_lengths.cpu().numpy()

        # Compute required FFT size per scale
        fft_sizes = np.array(
            [_next_fast_len(data_len + int(pl) - 1) for pl in psi_lens]
        )

        # Trim offsets: after diff on valid conv region, symmetric trim to T
        trim_floor = np.floor((psi_lens - 2) / 2.0).astype(np.int64)

        # Group by FFT size
        unique_fft_sizes = np.unique(fft_sizes)
        groups = []
        arange_T = torch.arange(data_len).long()  # reused

        for fs in unique_fft_sizes:
            mask = fft_sizes == fs
            indices = np.where(mask)[0]
            n_group = len(indices)

            # Build gather indices: [n_group, T]
            # For each scale j, after diff on full fft_size result,
            # the valid output starts at trim_floor[j].
            # gather_idx[j, t] = trim_floor[j] + t
            starts = torch.from_numpy(trim_floor[indices]).long()  # [n_group]
            gather_idx = starts.unsqueeze(1) + arange_T.unsqueeze(0)  # [n_group, T]

            groups.append(
                {
                    "fft_size": int(fs),
                    "indices": torch.from_numpy(indices).long(),
                    "gather_idx": gather_idx,  # [n_group, T]
                    "max_valid_len": int(
                        data_len + psi_lens[indices].max() - 2
                    ),
                }
            )
        return groups

    def _get_groups(self, data_len: int) -> list:
        """Get or build FFT groups for given data length."""
        if data_len not in self._group_cache:
            self._group_cache[data_len] = self._build_groups(data_len)
        return self._group_cache[data_len]

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Compute CWT using batched FFT — pure torch, zero CPU ops.

        Args:
            data: [batch, time] tensor

        Returns:
            [n_scales, batch, time] tensor of CWT coefficients (real).
        """
        B, T = data.shape
        device = data.device
        n_scales = self.scales.shape[0]
        groups = self._get_groups(T)

        out = torch.empty(n_scales, B, T, device=device, dtype=data.dtype)

        msb = self.max_scales_per_batch

        for g in groups:
            fft_size = g["fft_size"]
            idx = g["indices"].to(device)
            n_group = idx.shape[0]
            gather_idx = g["gather_idx"].to(device)  # [n_group, T]
            max_valid = g["max_valid_len"]

            # FFT of data (shared across all scales in this group)
            fft_data = torch.fft.fft(data, n=fft_size, dim=-1)  # [B, fft_size]

            # Process scales in sub-batches to control memory
            for sb_start in range(0, n_group, msb):
                sb_end = min(sb_start + msb, n_group)
                sb_idx = idx[sb_start:sb_end]
                sb_n = sb_end - sb_start

                # FFT of wavelets for this sub-batch
                wavs = self.int_psi_scales[sb_idx]  # [sb_n, max_psi_len]
                fft_wav = torch.fft.fft(wavs, n=fft_size, dim=-1)

                # Batched multiply → [sb_n, B, fft_size]
                conv = torch.fft.ifft(
                    fft_wav[:, None, :] * fft_data[None, :, :], dim=-1
                ).real

                # Diff on valid region
                coef = torch.diff(conv[..., : max_valid + 1], dim=-1)

                # Scale by -sqrt(scale)
                sqrt_sc = self.sqrt_scales[sb_idx].view(sb_n, 1, 1)
                coef = -sqrt_sc * coef

                # Vectorized trim via gather
                sb_gi = gather_idx[sb_start:sb_end].unsqueeze(1).expand(
                    sb_n, B, T
                )
                trimmed = torch.gather(coef, 2, sb_gi)  # [sb_n, B, T]

                out[sb_idx] = trimmed

        return out
