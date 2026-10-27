"""Capture one linear layer's input activations from a small open model and run the same
naive vs rotated fake-W4A4 comparison on them. Requires: pip install transformers>=4.40.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quarot_hadamard.analysis.error_profiler import compute_reconstruction_metrics
from quarot_hadamard.analysis.outlier_metrics import compute_outlier_profile
from quarot_hadamard.quantization.uniform_int4 import (
    Granularity,
    dequantize_symmetric_int4,
    quantize_symmetric_int4,
)
from quarot_hadamard.rotations.fwht import rotated_vector
from quarot_hadamard.rotations.hadamard import generate_random_signs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B"
    )  # down_proj input dim 4864 = 38*128 (not a power of two) -> block 128
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument(
        "--text",
        default="The patient presented with acute chest pain radiating to the left arm. " * 16,
    )
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    lin = model.model.layers[args.layer].mlp.down_proj  # (out, in); input is the MLP intermediate
    captured = {}

    def hook_fn(m, i, o):
        captured.setdefault("x", i[0].detach()[0])
        return None

    h = lin.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**tok(args.text, return_tensors="pt"))
    h.remove()
    X = captured["x"]
    W = lin.weight.detach()
    d = X.shape[-1]
    block = None if d & (d - 1) == 0 else args.block
    Y = X @ W.T

    def arm(signs):
        Xa, Wa = (
            (X, W)
            if signs is None
            else (
                rotated_vector(X, signs, block),
                rotated_vector(W, signs, block),
            )
        )
        xq, xs = quantize_symmetric_int4(Xa, granularity=Granularity.PER_TOKEN)
        wq, ws = quantize_symmetric_int4(Wa, granularity=Granularity.PER_CHANNEL)
        return dequantize_symmetric_int4(xq, xs) @ dequantize_symmetric_int4(wq, ws).T

    pn, pr = compute_outlier_profile(X), compute_outlier_profile(
        rotated_vector(X, generate_random_signs(d, seed=0), block)
    )
    mn, mr = compute_reconstruction_metrics(Y, arm(None)), compute_reconstruction_metrics(
        Y, arm(generate_random_signs(d, seed=0))
    )
    print(f"{args.model} layer {args.layer} down_proj input: tokens={X.shape[0]} d={d} block={block}")
    print(
        f"max|x| {pn.max_abs:.2f} -> {pr.max_abs:.2f} | PAR {pn.par:.1f} -> {pr.par:.1f} | kurtosis {pn.kurtosis:.1f} -> {pr.kurtosis:.1f}"
    )
    print(
        f"rel-Frob naive {mn.frobenius_error:.4f} rotated {mr.frobenius_error:.4f} | SQNR {mn.sqnr_db:.2f} -> {mr.sqnr_db:.2f} dB"
    )


if __name__ == "__main__":
    main()
