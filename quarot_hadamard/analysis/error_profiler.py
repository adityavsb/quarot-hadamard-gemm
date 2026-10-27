"""Error profiling and signal-to-noise metrics for quantized representations."""

from dataclasses import dataclass

import torch


@dataclass
class ReconstructionMetrics:
    frobenius_error: float  # Relative Frobenius error: ||exact - quant||_F / ||exact||_F
    max_error: float  # Max absolute point error: max(|exact - quant|)
    mean_squared_error: float  # MSE
    sqnr_db: float  # Signal-to-Quantization-Noise Ratio: 10 * log10(||exact||^2 / ||noise||^2)


def compute_reconstruction_metrics(exact: torch.Tensor, quantized: torch.Tensor) -> ReconstructionMetrics:
    """Evaluate fidelity of quantized tensor against floating-point reference.

    Args:
        exact: Ground-truth unquantized tensor.
        quantized: Quantized-dequantized tensor.

    Returns:
        ReconstructionMetrics dataclass.
    """
    diff = (exact - quantized).detach().cpu().to(torch.float64)
    exact_f64 = exact.detach().cpu().to(torch.float64)

    noise_norm = diff.norm().item()
    signal_norm = exact_f64.norm().item()

    rel_frobenius = noise_norm / max(signal_norm, 1e-12)
    max_err = diff.abs().max().item()
    mse = (diff**2).mean().item()

    # SQNR in decibels
    if noise_norm > 1e-12:
        sqnr = 20.0 * (torch.tensor(signal_norm) / torch.tensor(noise_norm)).log10().item()
    else:
        sqnr = float("inf")

    return ReconstructionMetrics(
        frobenius_error=rel_frobenius,
        max_error=max_err,
        mean_squared_error=mse,
        sqnr_db=sqnr,
    )


def compute_background_error(
    X: torch.Tensor,
    W: torch.Tensor,
    Y_hat: torch.Tensor,
    outlier_mask: torch.Tensor,
) -> float:
    """Relative Frobenius error of the *non-outlier* channels' contribution to Y = X @ W.T.

    The plain relative Frobenius error of Y is dominated by whichever channels carry the
    most energy. With a few large outlier channels, >90% of ||Y||_F^2 can come from them,
    so a quantiser that zeroes every background channel still scores a modest error.
    This metric isolates the part of Y produced by background channels and asks how much
    of it survives.

    Args:
        X: Exact activations, shape (n, d).
        W: Exact weights, shape (m, d).
        Y_hat: Reconstructed output from any quantised pipeline, shape (n, m).
        outlier_mask: Boolean tensor of shape (d,), True for outlier channels.

    Returns:
        ||Y_bg - Y_hat_bg||_F / ||Y_bg||_F, where Y_bg = X[:, ~mask] @ W[:, ~mask].T and
        Y_hat_bg = Y_hat - X[:, mask] @ W[:, mask].T (outlier contribution removed exactly).
    """
    # float64 on CPU: MPS has no float64, and the subtraction below must not add rounding noise.
    X64, W64, Yh64 = (t.detach().cpu().to(torch.float64) for t in (X, W, Y_hat))
    m = outlier_mask.cpu()
    Y_bg = X64[:, ~m] @ W64[:, ~m].T
    Y_out = X64[:, m] @ W64[:, m].T
    Y_hat_bg = Yh64 - Y_out
    return ((Y_bg - Y_hat_bg).norm() / max(Y_bg.norm().item(), 1e-12)).item()
