"""Tests for code paths that had no coverage: group-wise / per-tensor quantisation,
block-diagonal FWHT (ADR-003), the quantised RotatedLinear forward, bias, and fake_quantize."""

import pytest
import torch

from quarot_hadamard.analysis.outlier_metrics import compute_outlier_profile
from quarot_hadamard.kernels.rotated_linear import RotatedLinear
from quarot_hadamard.quantization.uniform_int4 import (
    Granularity,
    dequantize_symmetric_int4,
    fake_quantize_int4,
    quantize_symmetric_int4,
)
from quarot_hadamard.rotations.fwht import rotated_vector
from quarot_hadamard.rotations.hadamard import generate_random_signs


@pytest.mark.parametrize("granularity", [Granularity.PER_TENSOR, Granularity.PER_TOKEN, Granularity.PER_CHANNEL, Granularity.GROUP_WISE])
def test_quantize_dequantize_roundtrip_bounds(granularity):
    """Every granularity: ints in [-8, 7]; |x - x_hat| <= scale/2 element-wise (RTN bound)."""
    torch.manual_seed(0)
    x = torch.randn(8, 256) * 3.0
    q, scale = quantize_symmetric_int4(x, granularity=granularity, group_size=128)
    assert q.min() >= -8 and q.max() <= 7 and q.dtype == torch.int8
    x_hat = dequantize_symmetric_int4(q, scale, group_size=128)
    assert x_hat.shape == x.shape
    if granularity == Granularity.GROUP_WISE:
        scale_b = scale.expand(*scale.shape[:-1], 128).reshape(x.shape)
    else:
        scale_b = scale.expand_as(x) if scale.ndim else scale
    assert ((x - x_hat).abs() <= scale_b / 2 + 1e-6).all()


def test_group_wise_scale_shape_and_locality():
    """Group-wise scale has one entry per 128-group and a spike only affects its own group."""
    x = torch.randn(2, 512)
    x[0, 10] = 1000.0
    q, scale = quantize_symmetric_int4(x, granularity=Granularity.GROUP_WISE, group_size=128)
    assert scale.shape == (2, 4, 1)
    assert scale[0, 0, 0] > 100 and scale[0, 1, 0] < 2.0


def test_fake_quantize_matches_quant_dequant():
    x = torch.randn(4, 256)
    a = fake_quantize_int4(x, granularity=Granularity.PER_TOKEN)
    q, s = quantize_symmetric_int4(x, granularity=Granularity.PER_TOKEN)
    assert torch.equal(a, dequantize_symmetric_int4(q, s))


def test_block_diagonal_rotation_isometry_and_blockwise_mixing():
    """ADR-003: d = 7 x 2048 with block 2048 preserves norm and does not mix across blocks."""
    d, b = 14336, 2048
    signs = generate_random_signs(d, seed=3)
    x = torch.zeros(1, d)
    x[0, 5] = 1.0  # single spike in block 0
    y = rotated_vector(x, signs, block_size=b)
    assert torch.allclose(x.norm(), y.norm(), atol=1e-5)
    assert torch.all(y[0, b:] == 0), "spike leaked outside its block"
    assert torch.allclose(y[0, :b].abs(), torch.full((b,), 1 / b**0.5), atol=1e-6)


def test_block_diagonal_requires_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        rotated_vector(torch.randn(1, 100), generate_random_signs(100, seed=0), block_size=64)


def test_rotated_linear_block_path_exact_unquantized():
    torch.manual_seed(1)
    layer = RotatedLinear(14336, 16, seed=1, block_size=2048)
    x = torch.randn(3, 14336)
    assert torch.allclose(x @ layer.weight.T, layer(x, quantize=False), atol=1e-4)


def test_rotated_linear_quantized_forward_runs_and_caches_int8_weights():
    torch.manual_seed(2)
    layer = RotatedLinear(256, 32, bias=True, seed=2)
    x = torch.randn(5, 256)
    y = layer(x, quantize=True)
    assert y.shape == (5, 32)
    assert layer.W_rot_q is not None and layer.W_rot_q.dtype == torch.int8
    assert layer.W_rot_scale.shape == (32, 1)
    rel = (y - (x @ layer.weight.T + layer.bias)).norm() / (x @ layer.weight.T + layer.bias).norm()
    assert rel < 0.5  # fake W4A4 on Gaussian data: ~0.22 expected; guards against gross breakage


def test_rotated_linear_quantized_beats_naive_on_outliers():
    torch.manual_seed(3)
    d = 1024
    layer = RotatedLinear(d, 32, seed=3)
    x = torch.randn(16, d)
    x[:, 7] = 60.0
    y_ref = x @ layer.weight.T
    y_rot = layer(x, quantize=True)
    xq, xs = quantize_symmetric_int4(x, granularity=Granularity.PER_TOKEN)
    wq, ws = quantize_symmetric_int4(layer.weight.detach(), granularity=Granularity.PER_CHANNEL)
    y_naive = dequantize_symmetric_int4(xq, xs) @ dequantize_symmetric_int4(wq, ws).T
    assert (y_rot - y_ref).norm() < (y_naive - y_ref).norm()


@pytest.mark.skipif(not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()), reason="MPS not available")
def test_mps_path_signs_and_profile():
    signs = generate_random_signs(64, seed=42, device="mps")
    assert signs.device.type == "mps"
    assert torch.equal(signs.cpu(), generate_random_signs(64, seed=42))  # same draw on every device
    prof = compute_outlier_profile(torch.randn(4, 64, device="mps"))
    assert prof.max_abs > 0
