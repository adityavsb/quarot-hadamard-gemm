"""Sweep: reconstruction error vs. injected outlier scale and seed (synthetic activations).

Shows (a) how the plain relative-Frobenius gap depends on the injected outlier scale and
(b) the seed-to-seed spread of the headline numbers. Writes CSV to --out.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_ptq_comparison import generate_synthetic_activations
from quarot_hadamard.analysis.error_profiler import compute_background_error, compute_reconstruction_metrics
from quarot_hadamard.quantization.uniform_int4 import (
    Granularity,
    dequantize_symmetric_int4,
    quantize_symmetric_int4,
)
from quarot_hadamard.rotations.fwht import rotated_vector
from quarot_hadamard.rotations.hadamard import generate_random_signs


def fake_w4a4(X: torch.Tensor, W: torch.Tensor, signs: torch.Tensor | None) -> torch.Tensor:
    if signs is not None:
        X = rotated_vector(X, signs)
        W = rotated_vector(W, signs)
    Xq, Xs = quantize_symmetric_int4(X, granularity=Granularity.PER_TOKEN)
    Wq, Ws = quantize_symmetric_int4(W, granularity=Granularity.PER_CHANNEL)
    return dequantize_symmetric_int4(Xq, Xs) @ dequantize_symmetric_int4(Wq, Ws).T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--outlier-ratio", type=float, default=0.002)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", type=Path, default=Path("results/synthetic_sweep.csv"))
    args = ap.parse_args()

    rows = []
    for scale in args.scales:
        for seed in args.seeds:
            if scale > 0:
                X, mask = generate_synthetic_activations(args.batch, args.dim, scale, args.outlier_ratio, seed)
            else:
                torch.manual_seed(seed)
                X = torch.randn(args.batch, args.dim)
                mask = torch.zeros(args.dim, dtype=torch.bool)
            W = torch.randn(args.dim, args.dim, generator=torch.Generator().manual_seed(10_000 + seed)) * (2.0 / args.dim) ** 0.5
            Y = X @ W.T
            signs = generate_random_signs(args.dim, seed=seed)
            Yn, Yr = fake_w4a4(X, W, None), fake_w4a4(X, W, signs)
            mn, mr = compute_reconstruction_metrics(Y, Yn), compute_reconstruction_metrics(Y, Yr)
            rows.append(
                {
                    "scale": scale,
                    "seed": seed,
                    "naive_frob": round(mn.frobenius_error, 4),
                    "rot_frob": round(mr.frobenius_error, 4),
                    "naive_sqnr_db": round(mn.sqnr_db, 2),
                    "rot_sqnr_db": round(mr.sqnr_db, 2),
                    "naive_bg_err": round(compute_background_error(X, W, Yn, mask), 4) if mask.any() else None,
                    "rot_bg_err": round(compute_background_error(X, W, Yr, mask), 4) if mask.any() else None,
                    "rot_max_abs": round(rotated_vector(X, signs).abs().max().item(), 2),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'scale':>7} {'naive_frob':>18} {'rot_frob':>18} {'naive/rot':>10} {'naive_bg_err':>13} {'rot_bg_err':>11}")
    for scale in args.scales:
        sub = [r for r in rows if r["scale"] == scale]
        nf = [r["naive_frob"] for r in sub]
        rf = [r["rot_frob"] for r in sub]
        nb = [r["naive_bg_err"] for r in sub if r["naive_bg_err"] is not None]
        rb = [r["rot_bg_err"] for r in sub if r["rot_bg_err"] is not None]
        def sd(v: list[float]) -> float:
            return statistics.stdev(v) if len(v) > 1 else 0.0

        print(
            f"{scale:>7.1f} {statistics.mean(nf):>9.4f} ±{sd(nf):<7.4f} {statistics.mean(rf):>9.4f} ±{sd(rf):<7.4f} "
            f"{statistics.mean(nf)/statistics.mean(rf):>10.2f} "
            f"{(statistics.mean(nb) if nb else float('nan')):>13.3f} {(statistics.mean(rb) if rb else float('nan')):>11.3f}"
        )
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
