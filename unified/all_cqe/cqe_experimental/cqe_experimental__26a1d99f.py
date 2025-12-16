"""
Unit tests for MORSR Protocol
"""

import pytest
import numpy as np
from cqe.morsr.protocol import MORSRProtocol
from cqe.morsr.acceptance import AcceptanceChecker
from cqe.morsr.handshake import HandshakeRecord, HandshakeLogger


def test_acceptance_checker():
    """Test monotone acceptance logic"""
    checker = AcceptanceChecker(tolerance=1e-6)

    # Strict decrease should accept
    accepted, reason = checker.check(10.0, 8.0)
    assert accepted
    assert reason == "strict_decrease"

    # Plateau should accept
    accepted, reason = checker.check(10.0, 10.0000001)
    assert accepted
    assert reason == "plateau"

    # Increase should reject
    accepted, reason = checker.check(10.0, 12.0)
    assert not accepted
    assert reason == "increase_rejected"


def test_convergence_detection():
    """Test convergence detection"""
    checker = AcceptanceChecker(tolerance=1e-6)

    assert checker.is_converged(1e-7)
    assert not checker.is_converged(1e-5)


def test_handshake_record():
    """Test handshake record creation"""
    record = HandshakeRecord(
        operator_name="RotationOperator",
        phi_before=10.0,
        phi_after=8.0,
        delta_phi=-2.0,
        accepted=True,
        reason="strict_decrease",
        overlay_hash="abc123"
    )

    assert record.operator_name == "RotationOperator"
    assert record.accepted
    assert record.delta_phi < 0
    assert record.timestamp is not None


def test_handshake_serialization():
    """Test handshake to dict"""
    record = HandshakeRecord(
        operator_name="Test",
        phi_before=10.0,
        phi_after=8.0,
        delta_phi=-2.0,
        accepted=True,
        reason="test",
        overlay_hash="xyz"
    )

    data = record.to_dict()

    assert 'operator_name' in data
    assert 'phi_before' in data
    assert 'timestamp' in data


def test_handshake_logger():
    """Test handshake logging"""
    logger = HandshakeLogger()

    record1 = HandshakeRecord("Op1", 10.0, 9.0, -1.0, True, "decrease", "hash1")
    record2 = HandshakeRecord("Op2", 9.0, 11.0, 2.0, False, "increase", "hash2")

    logger.log(record1)
    logger.log(record2)

    assert len(logger.get_log()) == 2
    assert len(logger.get_accepted()) == 1
    assert len(logger.get_rejected()) == 1
    assert logger.acceptance_rate() == 0.5


def test_morsr_pulse_sweep(sample_overlay, morsr, canonicalizer):
    """Test MORSR pulse sweep"""
    # Canonicalize first
    sample_overlay = canonicalizer.canonicalize(sample_overlay)

    # Run pulse sweep
    optimized = morsr.pulse_sweep(sample_overlay, max_iterations=3)

    assert optimized.hash_id is not None
    assert len(optimized.provenance) >= len(sample_overlay.provenance)


def test_morsr_convergence(sample_overlay, morsr, canonicalizer):
    """Test MORSR convergence detection"""
    sample_overlay = canonicalizer.canonicalize(sample_overlay)

    # Run with low max_iterations
    result = morsr.pulse_sweep(sample_overlay, max_iterations=2)

    # Should complete without error
    assert result is not None


def test_morsr_handshake_logging(sample_overlay, morsr, canonicalizer):
    """Test MORSR logs handshakes"""
    sample_overlay = canonicalizer.canonicalize(sample_overlay)

    morsr.clear_log()
    morsr.pulse_sweep(sample_overlay, max_iterations=2)

    log = morsr.get_handshake_log()

    # Should have logged handshakes
    assert len(log) > 0

    # All records should have required fields
    for record in log:
        assert hasattr(record, 'operator_name')
        assert hasattr(record, 'phi_before')
        assert hasattr(record, 'accepted')
