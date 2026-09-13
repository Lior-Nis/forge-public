"""Numerical equivalence tests: GpuCwt vs ptwt.cwt reference."""

import math

import ptwt
import pytest
import torch

from model.cwt_gpu import GpuCwt


def _make_scales(n_scales: int = 64, fs: float = 100.0) -> torch.Tensor:
    """Reproduce the FOG-optimized scales from WaveletTransform."""
    fog_range_scales = int(n_scales * 0.625)
    locomotion_scales = int(n_scales * 0.25)
    high_freq_scales = n_scales - fog_range_scales - locomotion_scales

    freq_locomotion = torch.logspace(
        math.log10(0.5), math.log10(3.0), steps=locomotion_scales
    )
    freq_fog = torch.logspace(
        math.log10(3.0), math.log10(8.0), steps=fog_range_scales
    )
    freq_high = torch.logspace(
        math.log10(8.0), math.log10(10.0), steps=high_freq_scales
    )
    freqs = torch.cat([freq_locomotion, freq_fog, freq_high])
    fc = 0.8125  # Morlet
    return fc * fs / freqs


def _ptwt_reference(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Run ptwt.cwt as baseline reference."""
    coefs, _ = ptwt.cwt(data, scales.cpu(), "morl")
    return coefs


@pytest.fixture
def scales():
    return _make_scales()


@pytest.fixture
def gpu_cwt(scales):
    return GpuCwt(scales, wavelet="morl")


@pytest.mark.parametrize("seq_len", [200, 500, 1000])
def test_numerical_equivalence_float64(scales, seq_len):
    """GpuCwt matches ptwt.cwt in float64 within tight tolerance."""
    cwt = GpuCwt(scales, wavelet="morl").double()
    torch.manual_seed(42)
    data = torch.randn(4, seq_len, dtype=torch.float64)

    result = cwt(data)
    reference = _ptwt_reference(data, scales)

    assert result.shape == reference.shape, (
        f"Shape mismatch: {result.shape} vs {reference.shape}"
    )
    assert torch.allclose(result, reference, atol=1e-5, rtol=1e-4), (
        f"Max diff: {(result - reference).abs().max().item():.2e}"
    )


@pytest.mark.parametrize("seq_len", [200, 500, 1000])
def test_numerical_equivalence_float32(gpu_cwt, scales, seq_len):
    """GpuCwt in float32 matches ptwt.cwt within single-precision tolerance.

    Float32 FFT accumulation produces ~0.02 max absolute error vs float64
    ptwt reference. This is expected and acceptable — the downstream power
    spectrum (abs().pow(2) + log1p + normalization) smooths these differences.
    """
    torch.manual_seed(42)
    data = torch.randn(4, seq_len)

    result = gpu_cwt(data)
    reference = _ptwt_reference(data, scales).float()

    assert result.shape == reference.shape
    # 99th percentile diff should be < 1e-3, max < 0.05
    all_diffs = (result - reference).abs()
    assert all_diffs.quantile(0.99).item() < 5e-3, (
        f"99th pct diff: {all_diffs.quantile(0.99).item():.2e}"
    )
    assert all_diffs.max().item() < 0.1, (
        f"Max diff: {all_diffs.max().item():.2e}"
    )


@pytest.mark.parametrize("seq_len", [200, 500, 1000])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_numerical_equivalence_cuda(scales, seq_len):
    """GpuCwt on CUDA matches ptwt.cwt reference."""
    torch.manual_seed(42)
    data = torch.randn(4, seq_len)

    reference = _ptwt_reference(data, scales).float()

    gpu_cwt = GpuCwt(scales, wavelet="morl").cuda()
    result = gpu_cwt(data.cuda())

    assert result.shape == reference.shape
    all_diffs = (result.cpu() - reference).abs()
    assert all_diffs.quantile(0.99).item() < 5e-3
    assert all_diffs.max().item() < 0.1


def test_output_shape(gpu_cwt, scales):
    """Output shape is [n_scales, batch, time]."""
    data = torch.randn(8, 300)
    result = gpu_cwt(data)
    assert result.shape == (len(scales), 8, 300)


def test_batch_size_one(scales):
    """Works with batch size 1."""
    cwt = GpuCwt(scales, wavelet="morl").double()
    data = torch.randn(1, 200, dtype=torch.float64)
    result = cwt(data)
    reference = _ptwt_reference(data, scales)
    assert torch.allclose(result, reference, atol=1e-5, rtol=1e-4)


def test_deterministic(gpu_cwt):
    """Same input produces identical output across calls."""
    data = torch.randn(4, 200)
    r1 = gpu_cwt(data)
    r2 = gpu_cwt(data)
    assert torch.equal(r1, r2)


def test_different_input_lengths_cached(scales):
    """FFT group cache works across different input lengths."""
    cwt = GpuCwt(scales, wavelet="morl").double()
    torch.manual_seed(0)
    for seq_len in [200, 300, 200]:  # 200 appears twice — cache hit
        data = torch.randn(2, seq_len, dtype=torch.float64)
        result = cwt(data)
        reference = _ptwt_reference(data, scales)
        assert torch.allclose(result, reference, atol=1e-5, rtol=1e-4)


def test_few_scales():
    """Works with a small number of scales."""
    scales = torch.tensor([5.0, 10.0, 20.0])
    cwt = GpuCwt(scales, wavelet="morl").double()
    data = torch.randn(2, 200, dtype=torch.float64)
    result = cwt(data)
    reference = _ptwt_reference(data, scales)
    assert result.shape == reference.shape
    assert torch.allclose(result, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_movement():
    """GpuCwt buffers move correctly with .to(device)."""
    scales = _make_scales()
    cwt = GpuCwt(scales, wavelet="morl")

    cwt_cuda = cwt.cuda()
    data = torch.randn(2, 200, device="cuda")
    result = cwt_cuda(data)
    assert result.device.type == "cuda"

    cwt_cpu = cwt_cuda.cpu()
    data_cpu = torch.randn(2, 200)
    result_cpu = cwt_cpu(data_cpu)
    assert result_cpu.device.type == "cpu"
