"""RotatedLinear Layer: Linear projection with online Fast Walsh-Hadamard activation rotation."""

from typing import Optional
import torch
import torch.nn as nn

from ..rotations.fwht import rotated_vector
from ..rotations.hadamard import generate_random_signs
from ..quantization.uniform_int4 import (
    quantize_symmetric_int4,
    dequantize_symmetric_int4,
    Granularity,
)


class RotatedLinear(nn.Module):
    """Linear layer applying randomized Hadamard rotations to eliminate activation outliers prior to INT4 quantization.
    
    Mathematical equivalence:
        Y = X W^T = (X H_rand^T) (W H_rand^T)^T = X_rot W_rot^T
        
    Offline:
        W_rot = FWHT(W * signs) is pre-computed and quantized to INT4.
        
    Online forward:
        X_rot = FWHT(X * signs) is computed on-the-fly in O(d log d).
        X_rot is quantized to INT4 and GEMM is executed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        seed: Optional[int] = 42,
        block_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # Standard linear weights
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # Rademacher sign vector s in {-1, +1}^d
        signs = generate_random_signs(in_features, seed=seed, dtype=dtype, device=device)
        self.register_buffer("signs", signs)

        # Quantized pre-computed weights (cached offline)
        self.register_buffer("W_rot_q", None)
        self.register_buffer("W_rot_scale", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5.0**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def prepare_offline_weights(self):
        """Rotate weights offline and quantize to INT4: W_rot = FWHT(W * signs)."""
        with torch.no_grad():
            W_rot = rotated_vector(self.weight, self.signs, block_size=self.block_size)
            q, scale = quantize_symmetric_int4(W_rot, granularity=Granularity.PER_CHANNEL)
            self.W_rot_q = q
            self.W_rot_scale = scale

    def forward(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input activations of shape (..., in_features).
            quantize: If True, simulate W4A4 quantized inference.
                      If False, compute unquantized rotated forward pass.
        """
        # Step 1: Rotate activations online in O(d log d)
        x_rot = rotated_vector(x, self.signs, block_size=self.block_size)

        if not quantize:
            # Exact unquantized rotated forward pass (algebraically equal to x @ W^T)
            W_rot = rotated_vector(self.weight, self.signs, block_size=self.block_size)
            y = torch.matmul(x_rot, W_rot.T)
        else:
            # Quantized forward pass (W4A4)
            if self.W_rot_q is None:
                self.prepare_offline_weights()

            # Quantize rotated activations (per-token INT4)
            x_rot_q, x_rot_scale = quantize_symmetric_int4(x_rot, granularity=Granularity.PER_TOKEN)
            x_rot_dequant = dequantize_symmetric_int4(x_rot_q, x_rot_scale)
            w_rot_dequant = dequantize_symmetric_int4(self.W_rot_q, self.W_rot_scale)

            y = torch.matmul(x_rot_dequant, w_rot_dequant.T)

        if self.bias is not None:
            y = y + self.bias
        return y
