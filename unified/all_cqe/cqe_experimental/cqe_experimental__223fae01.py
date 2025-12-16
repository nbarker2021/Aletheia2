"""
Unit tests for CQE Overlay
"""

import pytest
import numpy as np
from cqe.core.overlay import CQEOverlay


def test_overlay_creation():
    """Test basic overlay creation"""
    overlay = CQEOverlay(
        present=np.zeros(248, dtype=bool),
        w=np.zeros(248),
        phi=np.zeros(248),
        pose={'domain': 'test'}
    )

    assert len(overlay.present) == 248
    assert len(overlay.w) == 248
    assert len(overlay.phi) == 248


def test_overlay_validation():
    """Test overlay size validation"""
    with pytest.raises(AssertionError):
        CQEOverlay(
            present=np.zeros(100, dtype=bool),  # Wrong size
            w=np.zeros(248),
            phi=np.zeros(248),
            pose={}
        )


def test_active_slots(sample_overlay):
    """Test active slot detection"""
    active = sample_overlay.active_slots
    assert len(active) == 3  # 1 root + 2 Cartan
    assert 0 in active
    assert 240 in active
    assert 241 in active


def test_cartan_active(sample_overlay):
    """Test Cartan lane counting"""
    assert sample_overlay.cartan_active == 2


def test_overlay_copy(sample_overlay):
    """Test overlay deep copy"""
    copy = sample_overlay.copy()

    assert np.array_equal(copy.present, sample_overlay.present)
    assert np.array_equal(copy.w, sample_overlay.w)
    assert np.array_equal(copy.phi, sample_overlay.phi)

    # Modify copy shouldn't affect original
    copy.w[0] = 999.0
    assert sample_overlay.w[0] != 999.0


def test_overlay_hash():
    """Test content-addressed hashing"""
    overlay1 = CQEOverlay(
        present=np.array([True] + [False]*247),
        w=np.array([1.0] + [0.0]*247),
        phi=np.zeros(248),
        pose={}
    )

    overlay2 = CQEOverlay(
        present=np.array([True] + [False]*247),
        w=np.array([1.0] + [0.0]*247),
        phi=np.zeros(248),
        pose={}
    )

    hash1 = overlay1.compute_hash()
    hash2 = overlay2.compute_hash()

    assert hash1 == hash2  # Same content = same hash


def test_overlay_serialization(sample_overlay):
    """Test overlay to/from dict"""
    data = sample_overlay.to_dict()

    assert 'present' in data
    assert 'w' in data
    assert 'phi' in data
    assert 'pose' in data

    # Deserialize
    restored = CQEOverlay.from_dict(data)

    assert np.array_equal(restored.present, sample_overlay.present)
    assert np.array_equal(restored.w, sample_overlay.w)
    assert np.array_equal(restored.phi, sample_overlay.phi)
