"""Statistical metrics and error profiling for activation outliers and quantization fidelity."""

from .outlier_metrics import compute_outlier_profile, OutlierProfile
from .error_profiler import compute_reconstruction_metrics, ReconstructionMetrics

__all__ = [
    "compute_outlier_profile",
    "OutlierProfile",
    "compute_reconstruction_metrics",
    "ReconstructionMetrics",
]
