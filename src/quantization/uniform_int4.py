"""Symmetric Uniform INT4 Quantizer for weights, activations, and KV-cache."""

from enum import Enum
from typing import Tuple
import torch


class Granularity(Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"   # Per-channel for weights (dim 0) or per-token for activations (dim -2)
    PER_TOKEN = "per_token"       # Last-dimension normalization (dim -1)
    GROUP_WISE = "group_wise"     # Block-wise groups (e.g. group_size=128)


def quantize_symmetric_int4(
    x: torch.Tensor,
    granularity: Granularity = Granularity.PER_TOKEN,
    group_size: int = 128,
    qmin: int = -8,
    qmax: int = 7,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize floating-point tensor x to signed 4-bit integers [-8, 7].
    
    Formula:
        scale = max(|x|) / qmax
        q = clip(round(x / scale), qmin, qmax)
        
    Args:
        x: Input floating-point tensor.
        granularity: Granularity mode (PER_TENSOR, PER_CHANNEL, PER_TOKEN, GROUP_WISE).
        group_size: Group size for GROUP_WISE mode.
        qmin: Minimum quantized integer (default: -8).
        qmax: Maximum quantized integer (default: 7).
        eps: Minimum scale to avoid division by zero.
        
    Returns:
        Tuple of (q_int4, scale).
    """
    orig_shape = x.shape
    
    if granularity == Granularity.PER_TENSOR:
        max_val = x.abs().max()
        scale = torch.clamp(max_val / float(qmax), min=eps)
    elif granularity == Granularity.PER_TOKEN:
        # Scale computed across the last dimension (hidden dimension d)
        max_val = x.abs().max(dim=-1, keepdim=True)[0]
        scale = torch.clamp(max_val / float(qmax), min=eps)
    elif granularity == Granularity.PER_CHANNEL:
        # Scale computed across output channels (dim 0 for weights)
        dims = tuple(range(1, x.ndim))
        max_val = x.abs().amax(dim=dims, keepdim=True)
        scale = torch.clamp(max_val / float(qmax), min=eps)
    elif granularity == Granularity.GROUP_WISE:
        d = orig_shape[-1]
        if d % group_size != 0:
            raise ValueError(f"Last dimension {d} must be divisible by group_size={group_size}.")
        reshaped = x.view(*orig_shape[:-1], d // group_size, group_size)
        max_val = reshaped.abs().max(dim=-1, keepdim=True)[0]
        scale = torch.clamp(max_val / float(qmax), min=eps)
        q = torch.clamp(torch.round(reshaped / scale), qmin, qmax)
        return q.view(orig_shape).to(torch.int8), scale
    else:
        raise ValueError(f"Unsupported granularity: {granularity}")
        
    q = torch.clamp(torch.round(x / scale), qmin, qmax).to(torch.int8)
    return q, scale


def dequantize_symmetric_int4(q: torch.Tensor, scale: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Dequantize signed 4-bit integer tensor back to floating point: x_hat = q * scale."""
    if scale.ndim == q.ndim + 1:
        # Group-wise dequantization
        orig_shape = q.shape
        d = orig_shape[-1]
        reshaped = q.view(*orig_shape[:-1], d // group_size, group_size)
        x_hat = reshaped.to(scale.dtype) * scale
        return x_hat.view(orig_shape)
    return q.to(scale.dtype) * scale


def fake_quantize_int4(
    x: torch.Tensor,
    granularity: Granularity = Granularity.PER_TOKEN,
    group_size: int = 128,
) -> torch.Tensor:
    """Simulate symmetric INT4 quantization error in floating-point precision."""
    q, scale = quantize_symmetric_int4(x, granularity=granularity, group_size=group_size)
    return dequantize_symmetric_int4(q, scale, group_size=group_size)
