"""Statistical metrics and error profiling for activation outliers and quantization fidelity."""

from .error_profiler import ReconstructionMetrics, compute_background_error, compute_reconstruction_metrics
from .outlier_metrics import OutlierProfile, compute_outlier_profile

__all__ = [
    "compute_outlier_profile",
    "OutlierProfile",
    "compute_background_error",
    "compute_reconstruction_metrics",
    "ReconstructionMetrics",
]
