"""Error profiling and signal-to-noise metrics for quantized representations."""

from dataclasses import dataclass
import torch


@dataclass
class ReconstructionMetrics:
    frobenius_error: float     # Relative Frobenius error: ||exact - quant||_F / ||exact||_F
    max_error: float           # Max absolute point error: max(|exact - quant|)
    mean_squared_error: float  # MSE
    sqnr_db: float             # Signal-to-Quantization-Noise Ratio: 10 * log10(||exact||^2 / ||noise||^2)


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
    mse = (diff ** 2).mean().item()
    
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
