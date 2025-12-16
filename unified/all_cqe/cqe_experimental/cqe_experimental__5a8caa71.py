"""
Unit tests for E8 Lattice
"""

import pytest
import numpy as np
from cqe.core.lattice import E8Lattice


def test_lattice_creation(e8_lattice):
    """Test E8 lattice initialization"""
    assert e8_lattice.dimension == 8
    assert e8_lattice.num_roots == 240
    assert e8_lattice.total_slots == 248


def test_basis_matrix(e8_lattice):
    """Test E8 basis matrix shape and properties"""
    B = e8_lattice.B

    assert B.shape == (8, 8)
    assert np.linalg.det(B) != 0  # Non-singular


def test_simple_root(e8_lattice):
    """Test simple root retrieval"""
    root = e8_lattice.get_simple_root(0)

    assert len(root) == 8
    assert root[0] == 1.0
    assert root[1] == -1.0


def test_invalid_root_index(e8_lattice):
    """Test invalid root index raises error"""
    with pytest.raises(ValueError):
        e8_lattice.get_simple_root(10)


def test_babai_projection(e8_lattice):
    """Test Babai nearest-plane algorithm"""
    vector = np.random.randn(8)

    lattice_point, error = e8_lattice.project_to_lattice(vector)

    assert len(lattice_point) == 8
    assert error >= 0  # Error is non-negative


def test_weyl_reflection(e8_lattice):
    """Test Weyl reflection"""
    vector = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)

    reflected = e8_lattice.weyl_reflect(vector, root_index=0)

    assert len(reflected) == 8
    assert not np.array_equal(reflected, vector)  # Should change


def test_lattice_info(e8_lattice):
    """Test lattice info retrieval"""
    info = e8_lattice.info()

    assert 'dimension' in info
    assert 'num_roots' in info
    assert info['dimension'] == 8
