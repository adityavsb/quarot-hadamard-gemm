"""Statistical profiling of activation outliers: Kurtosis, Peak-to-Average Ratio (PAR), and dynamic range."""

from dataclasses import dataclass
import torch


@dataclass
class OutlierProfile:
    max_abs: float
    mean_abs: float
    std: float
    par: float           # Peak-to-Average Ratio: max(|x|) / mean(|x|)
    kurtosis: float      # Excess kurtosis (0 for standard Gaussian)
    p99_ratio: float     # Ratio of max(|x|) to 99th percentile


def compute_outlier_profile(x: torch.Tensor) -> OutlierProfile:
    """Compute statistical distribution metrics quantifying channel outlier severity.
    
    Args:
        x: Activation tensor of arbitrary shape.
        
    Returns:
        OutlierProfile dataclass containing statistical metrics.
    """
    x_flat = x.flatten().to(torch.float64)
    abs_x = x_flat.abs()
    
    max_val = abs_x.max().item()
    mean_abs = abs_x.mean().item()
    par = max_val / max(mean_abs, 1e-12)
    
    mean = x_flat.mean()
    var = x_flat.var(unbiased=False)
    std = var.sqrt().item()
    
    # Excess Kurtosis: E[(x - mu)^4] / sigma^4 - 3
    if var > 1e-12:
        m4 = ((x_flat - mean) ** 4).mean()
        kurtosis = (m4 / (var ** 2)).item() - 3.0
    else:
        kurtosis = 0.0
        
    # 99th percentile ratio
    p99 = torch.quantile(abs_x, 0.99).item()
    p99_ratio = max_val / max(p99, 1e-12)
    
    return OutlierProfile(
        max_abs=max_val,
        mean_abs=mean_abs,
        std=std,
        par=par,
        kurtosis=kurtosis,
        p99_ratio=p99_ratio,
    )
