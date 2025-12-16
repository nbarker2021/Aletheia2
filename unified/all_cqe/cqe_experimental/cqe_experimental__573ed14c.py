"""
Unit tests for ALENA Operators
"""

import pytest
import numpy as np
from cqe.operators.rotation import RotationOperator
from cqe.operators.reflection import ReflectionOperator
from cqe.operators.midpoint import MidpointOperator
from cqe.operators.parity import ParityMirrorOperator, ECCParityOperator
from cqe.operators.insertion import SingleInsertOperator


def test_rotation_operator(sample_overlay):
    """Test rotation operator"""
    op = RotationOperator(theta=np.pi/4)

    result = op.apply(sample_overlay)

    assert len(result.active_slots) == len(sample_overlay.active_slots)
    assert result.provenance[-1].startswith("R_theta")


def test_rotation_inverse(sample_overlay):
    """Test rotation is reversible"""
    op = RotationOperator(theta=np.pi/4)

    transformed = op.apply(sample_overlay)
    restored = op.inverse(transformed)

    # Phases should be approximately restored
    assert np.allclose(restored.phi, sample_overlay.phi, atol=1e-6)


def test_rotation_quantization():
    """Test theta quantization to π/12"""
    op = RotationOperator(theta=0.3)

    # Should quantize to nearest π/12 multiple
    expected_quanta = np.round(0.3 / (np.pi/12))
    expected_theta = expected_quanta * (np.pi/12)

    assert np.isclose(op.theta, expected_theta)


def test_reflection_operator(sample_overlay):
    """Test Weyl reflection operator"""
    op = ReflectionOperator(simple_root_idx=0)

    result = op.apply(sample_overlay)

    assert len(result.active_slots) == len(sample_overlay.active_slots)
    assert result.provenance[-1].startswith("WeylReflect")


def test_reflection_involution(sample_overlay):
    """Test reflection is its own inverse"""
    op = ReflectionOperator(simple_root_idx=0)

    transformed = op.apply(sample_overlay)
    restored = op.apply(transformed)

    # Double reflection should restore
    assert np.allclose(restored.phi, sample_overlay.phi, atol=1e-6)


def test_midpoint_operator(sample_overlay):
    """Test midpoint palindrome operator"""
    op = MidpointOperator()

    result = op.apply(sample_overlay)

    assert result.provenance[-1] == "Midpoint(palindrome)"


def test_parity_mirror_operator(sample_overlay):
    """Test parity mirror operator"""
    op = ParityMirrorOperator()

    result = op.apply(sample_overlay)

    # Should have more active Cartan lanes
    assert result.cartan_active >= sample_overlay.cartan_active
    assert result.provenance[-1] == "ParityMirror"


def test_ecc_parity_operator(sample_overlay):
    """Test ECC parity correction"""
    op = ECCParityOperator()

    result = op.apply(sample_overlay)

    # Check parity is even
    cartan_bits = result.present[240:248].astype(int)
    parity = np.sum(cartan_bits) % 2

    assert parity == 0  # Even parity after ECC


def test_single_insert_operator(sample_overlay):
    """Test single insertion operator"""
    op = SingleInsertOperator(weight=2.0)

    result = op.apply(sample_overlay)

    # Should have one more active slot
    assert len(result.active_slots) >= len(sample_overlay.active_slots)


def test_operator_cost(sample_overlay):
    """Test operator cost estimation"""
    ops = [
        RotationOperator(),
        ReflectionOperator(),
        MidpointOperator(),
        ECCParityOperator()
    ]

    for op in ops:
        cost = op.cost(sample_overlay)
        assert cost > 0
        assert isinstance(cost, float)


def test_operator_validation(sample_overlay):
    """Test operator validation"""
    op = RotationOperator()

    # Canonical overlay should validate
    sample_overlay.hash_id = "test_hash"
    assert op.validate(sample_overlay)

    # Non-canonical should fail
    sample_overlay.hash_id = None
    assert not op.validate(sample_overlay)


def test_operator_composition(sample_overlay):
    """Test applying multiple operators in sequence"""
    from canonicalizer import Canonicalizer
    from cqe.core.lattice import E8Lattice

    canonicalizer = Canonicalizer(E8Lattice())
    sample_overlay = canonicalizer.canonicalize(sample_overlay)

    op1 = RotationOperator()
    op2 = MidpointOperator()

    result = op2.apply(op1.apply(sample_overlay))

    assert len(result.provenance) >= 2
