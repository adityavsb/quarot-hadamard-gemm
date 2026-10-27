"""Comparative benchmark runner: FP32 baseline vs. naive fake-INT4 vs. rotated fake-INT4 (QuaRot-style).

What this measures
------------------
* Activation outlier statistics before/after a randomised Hadamard rotation.
* Reconstruction fidelity of Y = X @ W.T under symmetric per-token / per-channel
  round-to-nearest INT4 *fake* quantisation (quantise -> dequantise -> float32 matmul).
* Wall-clock cost of the FWHT rotation relative to the float32 GEMM it is added to.

What this does NOT measure
--------------------------
* Integer GEMM throughput or memory bandwidth: operands are dequantised to float32 and
  stored as int8, so no 4-bit kernel and no bandwidth saving is exercised.
* Any real LLM: activations are synthetic (Gaussian background + injected outlier channels).
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.error_profiler import compute_background_error, compute_reconstruction_metrics
from src.analysis.outlier_metrics import compute_outlier_profile
from src.quantization.uniform_int4 import (
    Granularity,
    dequantize_symmetric_int4,
    quantize_symmetric_int4,
)
from src.rotations.fwht import rotated_vector
from src.rotations.hadamard import generate_random_signs


def generate_synthetic_activations(
    batch_size: int,
    dim: int,
    outlier_scale: float = 80.0,
    outlier_ratio: float = 0.002,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic activation tensor with persistent heavy-tailed channel outliers.

    Returns (X, outlier_mask). RNG consumption order is identical to the 2026-10-25 version
    so that the committed deterministic numbers reproduce.
    """
    torch.manual_seed(seed)
    X = torch.randn(batch_size, dim, device=device)
    num_outliers = max(1, int(dim * outlier_ratio))
    perm = torch.randperm(dim, generator=torch.Generator().manual_seed(seed))
    outlier_indices = perm[:num_outliers]
    mask = torch.zeros(dim, dtype=torch.bool, device=device)
    mask[outlier_indices] = True
    for ch in outlier_indices:
        direction = torch.sign(torch.randn(1, device=device)).item()
        if direction == 0:
            direction = 1.0
        magnitudes = outlier_scale * (0.8 + 0.4 * torch.rand(batch_size, device=device))
        X[:, ch] = direction * magnitudes
    return X, mask


def timeit(fn, warmup: int = 10, reps: int = 50) -> dict:
    """Median / p10 / p90 wall-clock in ms after warm-up."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        "median_ms": round(statistics.median(ts), 3),
        "p10_ms": round(ts[int(0.1 * (len(ts) - 1))], 3),
        "p90_ms": round(ts[int(0.9 * (len(ts) - 1))], 3),
        "reps": reps,
    }


def system_info(device: str) -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "device_used": device,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cuda_available": torch.cuda.is_available(),
    }
    if sys.platform == "darwin":
        for key, cmd in {
            "chip": ["sysctl", "-n", "machdep.cpu.brand_string"],
            "model": ["sysctl", "-n", "hw.model"],
            "memory_bytes": ["sysctl", "-n", "hw.memsize"],
            "perf_cores": ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            "eff_cores": ["sysctl", "-n", "hw.perflevel1.physicalcpu"],
        }.items():
            try:
                info[key] = subprocess.check_output(cmd, text=True).strip()
            except Exception:  # noqa: BLE001 - best-effort telemetry
                info[key] = None
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def run_benchmark(
    dim: int,
    batch_size: int,
    outlier_scale: float,
    outlier_ratio: float,
    seed: int,
    device: str,
    reps: int,
    out_path: Path | None,
) -> dict:
    torch.manual_seed(seed)
    X, mask = generate_synthetic_activations(batch_size, dim, outlier_scale, outlier_ratio, seed, device)
    W = torch.randn(dim, dim, device=device) * (2.0 / dim) ** 0.5

    # Exact float32 reference
    Y_exact = torch.matmul(X, W.T)

    # Naive fake-INT4 (weights quantised once, offline, same as the rotated arm)
    W_q_naive, W_s_naive = quantize_symmetric_int4(W, granularity=Granularity.PER_CHANNEL)
    W_deq_naive = dequantize_symmetric_int4(W_q_naive, W_s_naive)

    def naive_forward():
        X_q, X_s = quantize_symmetric_int4(X, granularity=Granularity.PER_TOKEN)
        return torch.matmul(dequantize_symmetric_int4(X_q, X_s), W_deq_naive.T)

    Y_naive = naive_forward()

    # Rotated fake-INT4 (QuaRot-style): offline weight rotation + quantisation
    signs = generate_random_signs(dim, seed=seed, device=device)
    W_rot = rotated_vector(W, signs)
    W_q_rot, W_s_rot = quantize_symmetric_int4(W_rot, granularity=Granularity.PER_CHANNEL)
    W_deq_rot = dequantize_symmetric_int4(W_q_rot, W_s_rot)

    def rotated_forward():
        X_rot_ = rotated_vector(X, signs)
        X_q, X_s = quantize_symmetric_int4(X_rot_, granularity=Granularity.PER_TOKEN)
        return torch.matmul(dequantize_symmetric_int4(X_q, X_s), W_deq_rot.T)

    X_rot = rotated_vector(X, signs)
    Y_rot = rotated_forward()

    # Dense rotation reference for the FWHT wall-clock claim
    H_dense = rotated_vector(torch.eye(dim, device=device), signs)

    prof_orig = compute_outlier_profile(X)
    prof_rot = compute_outlier_profile(X_rot)
    err_naive = compute_reconstruction_metrics(Y_exact, Y_naive)
    err_rot = compute_reconstruction_metrics(Y_exact, Y_rot)
    bg_naive = compute_background_error(X, W, Y_naive, mask)
    bg_rot = compute_background_error(X, W, Y_rot, mask)
    outlier_energy_share = ((X[:, mask] ** 2).sum() / (X**2).sum()).item()
    X_q_naive_only, _ = quantize_symmetric_int4(X, granularity=Granularity.PER_TOKEN)
    bg_zeroed_frac = (X_q_naive_only[:, ~mask] == 0).float().mean().item()

    if device != "cpu":
        sync = (lambda: torch.mps.synchronize()) if device == "mps" else torch.cuda.synchronize
    else:
        sync = lambda: None  # noqa: E731

    def timed(fn):
        def inner():
            fn()
            sync()

        return timeit(inner, reps=reps)

    lat = {
        "fp32_gemm": timed(lambda: torch.matmul(X, W.T)),
        "fwht_rotation_only": timed(lambda: rotated_vector(X, signs)),
        "dense_rotation_only": timed(lambda: torch.matmul(X, H_dense)),
        "naive_fake_int4_forward": timed(naive_forward),
        "rotated_fake_int4_forward": timed(rotated_forward),
    }
    overhead_pct = 100.0 * lat["fwht_rotation_only"]["median_ms"] / lat["fp32_gemm"]["median_ms"]
    fwht_vs_dense = lat["dense_rotation_only"]["median_ms"] / lat["fwht_rotation_only"]["median_ms"]

    result = {
        "benchmark": "quarot-hadamard-gemm",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "system": system_info(device),
        "parameters": {
            "dimension_d": dim,
            "batch_size": batch_size,
            "outlier_scale": outlier_scale,
            "outlier_channel_ratio": outlier_ratio,
            "outlier_channels": int(mask.sum().item()),
            "seed": seed,
            "quantization": "symmetric RTN INT4, per-token activations / per-channel weights, FAKE quantisation (float32 matmul)",
        },
        "activation_distribution": {
            "unrotated": prof_orig.__dict__,
            "rotated": prof_rot.__dict__,
            "outlier_channel_energy_share_of_X": round(outlier_energy_share, 4),
        },
        "reconstruction_fidelity": {
            "naive_int4": {**err_naive.__dict__, "background_channel_rel_error": bg_naive},
            "rotated_int4": {**err_rot.__dict__, "background_channel_rel_error": bg_rot},
            "naive_fraction_of_background_entries_quantised_to_zero": round(bg_zeroed_frac, 4),
        },
        "latency_ms": lat,
        "derived": {
            "fwht_overhead_pct_of_fp32_gemm": round(overhead_pct, 1),
            "fwht_speedup_vs_dense_rotation_wallclock": round(fwht_vs_dense, 2),
        },
    }

    # Human-readable report
    print("=" * 80)
    print("quarot-hadamard-gemm — synthetic PTQ comparison (fake quantisation, float32 matmul)")
    print(f"device={device}  d={dim}  B={batch_size}  outlier_scale={outlier_scale}  outlier_channels={int(mask.sum())} ({outlier_ratio*100:.2f}%)  seed={seed}")
    print("=" * 80)
    print("\n[1] ACTIVATION DISTRIBUTION")
    print(f"{'metric':<28}{'unrotated':>14}{'rotated':>14}")
    for k in ("max_abs", "par", "kurtosis", "p99_ratio"):
        print(f"{k:<28}{getattr(prof_orig, k):>14.2f}{getattr(prof_rot, k):>14.2f}")
    print(f"share of ||X||_F^2 in outlier channels: {outlier_energy_share:.3f}")
    print("\n[2] RECONSTRUCTION FIDELITY (Y = X W^T)")
    print(f"{'metric':<34}{'naive INT4':>14}{'rotated INT4':>14}")
    for k in ("frobenius_error", "sqnr_db", "max_error", "mean_squared_error"):
        print(f"{k:<34}{getattr(err_naive, k):>14.4f}{getattr(err_rot, k):>14.4f}")
    print(f"{'background-channel rel. error':<34}{bg_naive:>14.4f}{bg_rot:>14.4f}")
    print(f"naive INT4 zeroes {100*bg_zeroed_frac:.1f}% of background-channel entries")
    print(f"\n[3] LATENCY (median of {reps} after warm-up, ms)")
    for k, v in lat.items():
        print(f"{k:<30}{v['median_ms']:>10.2f}  (p10 {v['p10_ms']:.2f} / p90 {v['p90_ms']:.2f})")
    print(f"FWHT overhead vs FP32 GEMM: {overhead_pct:.1f}%   |   FWHT vs dense rotation wall-clock: {fwht_vs_dense:.2f}x")
    print("=" * 80)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"wrote {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuaRot-style FWHT synthetic PTQ benchmark")
    parser.add_argument("--dim", type=int, default=4096, help="Hidden dimension (power of 2)")
    parser.add_argument("--batch", type=int, default=128, help="Number of tokens")
    parser.add_argument("--outlier-scale", type=float, default=80.0)
    parser.add_argument("--outlier-ratio", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--reps", type=int, default=50, help="Timed repetitions after warm-up")
    parser.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0 = leave default)")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON telemetry to this path")
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    run_benchmark(
        dim=args.dim,
        batch_size=args.batch,
        outlier_scale=args.outlier_scale,
        outlier_ratio=args.outlier_ratio,
        seed=args.seed,
        device=args.device,
        reps=args.reps,
        out_path=args.out,
    )
