"""Sylvester-Hadamard Matrix Construction and Randomized Orthogonal Basis Utilities."""

from typing import Optional
import torch


def get_hadamard_matrix(d: int, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """Construct an orthonormal Walsh-Hadamard matrix of order d via Sylvester recursion.
    
    H_1 = [1]
    H_{2k} = 1/sqrt(2) * [[H_k,  H_k],
                         [H_k, -H_k]]
    
    Args:
        d: Matrix dimension. Must be a power of 2.
        dtype: PyTorch tensor dtype (default: float32).
        device: PyTorch tensor device.
        
    Returns:
        Orthonormal tensor H of shape (d, d) satisfying H.T @ H = I.
    """
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"Dimension d={d} must be a positive power of 2.")
        
    H = torch.tensor([[1.0]], dtype=dtype, device=device)
    curr_d = 1
    sqrt2 = 2.0 ** 0.5
    
    while curr_d < d:
        top = torch.cat([H, H], dim=1)
        bottom = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bottom], dim=0) / sqrt2
        curr_d *= 2
        
    return H


def generate_random_signs(d: int, seed: Optional[int] = None, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """Generate a randomized Rademacher sign vector s in {-1, +1}^d.
    
    Used to form randomized orthogonal Hadamard matrices H_rand = H @ diag(s).
    
    Args:
        d: Vector length.
        seed: Optional RNG seed for deterministic reproduction.
        dtype: PyTorch tensor dtype.
        device: PyTorch tensor device.
        
    Returns:
        Tensor of shape (d,) with entries uniformly chosen from {-1.0, +1.0}.
    """
    gen = None
    if seed is not None:
        dev_obj = torch.device(device) if device is not None else None
        gen_dev = dev_obj.type if dev_obj and dev_obj.type != "mps" else "cpu"
        gen = torch.Generator(device=gen_dev)
        gen.manual_seed(seed)
        
    bits = torch.randint(0, 2, (d,), generator=gen, device=device)
    signs = (bits * 2 - 1).to(dtype=dtype)
    return signs
