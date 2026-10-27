"""Unit tests for symmetric INT4 quantization and RotatedLinear layer."""

import torch

from quarot_hadamard.analysis.error_profiler import compute_reconstruction_metrics
from quarot_hadamard.analysis.outlier_metrics import compute_outlier_profile
from quarot_hadamard.kernels.rotated_linear import RotatedLinear
from quarot_hadamard.quantization.uniform_int4 import (
    Granularity,
    dequantize_symmetric_int4,
    quantize_symmetric_int4,
)
from quarot_hadamard.rotations.fwht import rotated_vector
from quarot_hadamard.rotations.hadamard import generate_random_signs


def test_quantization_bounds():
    """Verify that quantized INT4 tensor strictly adheres to [-8, 7] bounds."""
    x = torch.randn(16, 256) * 5.0
    q, scale = quantize_symmetric_int4(x, granularity=Granularity.PER_TOKEN)

    assert q.min().item() >= -8
    assert q.max().item() <= 7
    assert q.dtype == torch.int8


def test_outlier_suppression():
    """Verify that randomized Hadamard rotation reduces peak-to-average ratio on outlier tensors."""
    d = 1024
    torch.manual_seed(42)
    x = torch.randn(1, d)
    # Inject 80.0 magnitude outlier into 2 channels
    x[0, 15] = 80.0
    x[0, 500] = -80.0

    profile_before = compute_outlier_profile(x)

    signs = generate_random_signs(d, seed=42)
    x_rot = rotated_vector(x, signs)
    profile_after = compute_outlier_profile(x_rot)

    # Peak amplitude should drop by at least 4x
    assert profile_after.max_abs < profile_before.max_abs / 4.0
    # Kurtosis should drop significantly
    assert profile_after.kurtosis < profile_before.kurtosis / 10.0


def test_rotated_linear_unquantized_exact_match():
    """Verify that unquantized RotatedLinear layer is mathematically identical to standard Linear."""
    torch.manual_seed(42)
    in_features = 128
    out_features = 64

    layer = RotatedLinear(in_features, out_features, bias=False, seed=42)
    x = torch.randn(8, in_features)

    # Reference forward pass: x @ W.T
    y_ref = x @ layer.weight.T

    # Rotated forward pass without quantization
    y_rot = layer(x, quantize=False)

    assert torch.allclose(y_ref, y_rot, atol=1e-5)


def test_rotated_int4_lower_reconstruction_error():
    """Verify that Rotated INT4 achieves lower reconstruction error than naive INT4 on heavy-tailed inputs."""
    torch.manual_seed(42)
    d = 2048
    x = torch.randn(16, d)
    # Inject heavy-tailed outliers into 4 channels
    for ch in [10, 100, 500, 1500]:
        x[:, ch] = torch.sign(torch.randn(16)) * 70.0

    w = torch.randn(64, d) * 0.02
    y_exact = x @ w.T

    # 1. Naive INT4
    x_q_naive, x_scale = quantize_symmetric_int4(x, granularity=Granularity.PER_TOKEN)
    w_q_naive, w_scale = quantize_symmetric_int4(w, granularity=Granularity.PER_CHANNEL)
    y_naive = dequantize_symmetric_int4(x_q_naive, x_scale) @ dequantize_symmetric_int4(w_q_naive, w_scale).T

    # 2. Rotated INT4
    signs = generate_random_signs(d, seed=42)
    x_rot = rotated_vector(x, signs)
    w_rot = rotated_vector(w, signs)

    x_q_rot, x_rot_scale = quantize_symmetric_int4(x_rot, granularity=Granularity.PER_TOKEN)
    w_q_rot, w_rot_scale = quantize_symmetric_int4(w_rot, granularity=Granularity.PER_CHANNEL)
    y_rot = dequantize_symmetric_int4(x_q_rot, x_rot_scale) @ dequantize_symmetric_int4(w_q_rot, w_rot_scale).T

    metrics_naive = compute_reconstruction_metrics(y_exact, y_naive)
    metrics_rot = compute_reconstruction_metrics(y_exact, y_rot)

    # Relative Frobenius error must be strictly lower with rotation
    assert metrics_rot.frobenius_error < metrics_naive.frobenius_error
    # SQNR must be higher with rotation
    assert metrics_rot.sqnr_db > metrics_naive.sqnr_db
