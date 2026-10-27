"""Unit tests for Hadamard orthogonality and inner product preservation."""

import pytest
import torch

from quarot_hadamard.rotations.hadamard import generate_random_signs, get_hadamard_matrix


@pytest.mark.parametrize("d", [2, 4, 8, 16, 32, 64])
def test_hadamard_orthonormality(d: int):
    """Verify that H.T @ H == I and H @ H.T == I."""
    H = get_hadamard_matrix(d, dtype=torch.float64)
    I = torch.eye(d, dtype=torch.float64)

    HTH = H.T @ H
    HHT = H @ H.T

    assert torch.allclose(HTH, I, atol=1e-12), f"H.T @ H != I for d={d}"
    assert torch.allclose(HHT, I, atol=1e-12), f"H @ H.T != I for d={d}"


@pytest.mark.parametrize("d", [4, 16, 64])
def test_randomized_hadamard_orthogonality(d: int):
    """Verify that H_rand = H @ diag(s) remains strictly orthonormal."""
    H = get_hadamard_matrix(d, dtype=torch.float64)
    signs = generate_random_signs(d, seed=42, dtype=torch.float64)

    H_rand = H @ torch.diag(signs)
    I = torch.eye(d, dtype=torch.float64)

    assert torch.allclose(H_rand.T @ H_rand, I, atol=1e-12)


def test_inner_product_invariance():
    """Verify that inner product <u, v> == <H_rand u, H_rand v>."""
    d = 128
    H = get_hadamard_matrix(d, dtype=torch.float64)
    signs = generate_random_signs(d, seed=99, dtype=torch.float64)
    H_rand = H @ torch.diag(signs)

    u = torch.randn(d, dtype=torch.float64)
    v = torch.randn(d, dtype=torch.float64)

    dot_original = torch.dot(u, v)

    u_rot = H_rand @ u
    v_rot = H_rand @ v
    dot_rot = torch.dot(u_rot, v_rot)

    assert torch.allclose(dot_original, dot_rot, atol=1e-12)
