"""Unit tests for Fast Walsh-Hadamard Transform (FWHT)."""

import pytest
import torch

from quarot_hadamard.rotations.hadamard import get_hadamard_matrix, generate_random_signs
from quarot_hadamard.rotations.fwht import fast_walsh_hadamard_transform, rotated_vector


@pytest.mark.parametrize("d", [2, 4, 8, 16, 64, 256, 1024])
def test_fwht_matches_dense_sylvester(d: int):
    """Verify that O(d log d) FWHT produces mathematically identical output to dense H @ x."""
    H = get_hadamard_matrix(d, dtype=torch.float64)
    x = torch.randn(8, d, dtype=torch.float64)
    
    # Dense matrix multiplication: y = x @ H^T
    y_dense = x @ H.T
    
    # Fast butterfly FWHT
    y_fast = fast_walsh_hadamard_transform(x)
    
    max_diff = (y_dense - y_fast).abs().max().item()
    assert max_diff < 1e-10, f"FWHT failed to match dense Sylvester matrix at d={d}, max_diff={max_diff}"


@pytest.mark.parametrize("d", [4, 16, 128])
def test_fwht_isometry_norm_preservation(d: int):
    """Verify Parseval theorem: ||FWHT(x)||_2 == ||x||_2."""
    x = torch.randn(10, d, dtype=torch.float64)
    y = fast_walsh_hadamard_transform(x)
    
    norm_x = x.norm(dim=-1)
    norm_y = y.norm(dim=-1)
    
    rel_diff = ((norm_x - norm_y).abs() / norm_x).max().item()
    assert rel_diff < 1e-12, f"FWHT violated isometric norm preservation, rel_diff={rel_diff}"


@pytest.mark.parametrize("d", [4, 16, 64])
def test_fwht_involution_property(d: int):
    """Verify that FWHT is its own inverse: FWHT(FWHT(x)) == x."""
    x = torch.randn(5, d, dtype=torch.float64)
    x_roundtrip = fast_walsh_hadamard_transform(fast_walsh_hadamard_transform(x))
    
    max_diff = (x - x_roundtrip).abs().max().item()
    assert max_diff < 1e-12, f"FWHT involution failed, max_diff={max_diff}"


def test_fwht_non_power_of_two_raises():
    """Verify that non-power-of-two dimensions raise ValueError."""
    x = torch.randn(4, 100)
    with pytest.raises(ValueError, match="power-of-2"):
        fast_walsh_hadamard_transform(x)


def test_rotated_vector_with_signs():
    """Verify randomized FWHT with Rademacher sign vector."""
    d = 64
    signs = generate_random_signs(d, seed=123)
    x = torch.randn(4, d)
    
    x_rot = rotated_vector(x, signs)
    # Energy must be preserved
    assert torch.allclose(x.norm(dim=-1), x_rot.norm(dim=-1), atol=1e-5)
