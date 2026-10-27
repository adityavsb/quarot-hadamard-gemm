"""Vectorized Fast Walsh-Hadamard Transform (FWHT) in O(d log d) arithmetic operations."""

import torch


def fast_walsh_hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Compute the orthonormal Fast Walsh-Hadamard Transform over the last dimension.

    Evaluates y = x @ H_d^T in O(d log2(d)) operations using a vectorized butterfly graph.

    Args:
        x: Input tensor of shape (..., d), where d must be a power of 2.
        normalize: If True, scale by 1 / sqrt(d) to ensure isometry (norm preservation).

    Returns:
        Transformed tensor of identical shape (..., d).
    """
    orig_shape = x.shape
    d = orig_shape[-1]

    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"Fast Walsh-Hadamard Transform requires power-of-2 dimension, got d={d}.")

    # Operate on flattened batch dimensions for contiguous memory access
    out = x.reshape(-1, d).clone()
    h = 1

    while h < d:
        # Reshape to pair adjacent butterfly elements
        out = out.view(-1, d // (2 * h), 2, h)
        a = out[..., 0, :]
        b = out[..., 1, :]
        # Vectorized butterfly: [a + b, a - b]
        out = torch.stack([a + b, a - b], dim=-2)
        h *= 2

    out = out.view(orig_shape)
    if normalize:
        out = out / (float(d) ** 0.5)

    return out


def rotated_vector(x: torch.Tensor, signs: torch.Tensor, block_size: int | None = None) -> torch.Tensor:
    """Apply randomized Fast Walsh-Hadamard Transform: x_rot = FWHT(x * signs).

    Supports block-diagonal decomposition for non-power-of-two dimensions (ADR-003).

    Args:
        x: Input tensor of shape (..., d).
        signs: Random Rademacher sign vector of shape (d,) with entries in {-1, +1}.
        block_size: Optional sub-block size for non-power-of-two hidden dimensions.
                    If None, d must be a power of 2.

    Returns:
        Rotated tensor x_rot of identical shape.
    """
    d = x.shape[-1]
    x_signed = x * signs

    if block_size is None:
        return fast_walsh_hadamard_transform(x_signed, normalize=True)

    if d % block_size != 0:
        raise ValueError(f"Dimension d={d} must be divisible by block_size={block_size}.")

    # Block-diagonal FWHT across sub-blocks
    orig_shape = x.shape
    num_blocks = d // block_size
    reshaped = x_signed.view(*orig_shape[:-1], num_blocks, block_size)
    transformed = fast_walsh_hadamard_transform(reshaped, normalize=True)
    return transformed.view(orig_shape)
