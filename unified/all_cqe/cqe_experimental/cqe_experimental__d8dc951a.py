"""
Validation utilities for CQE objects
"""

import numpy as np
from typing import Tuple, Optional
from cqe.core.overlay import CQEOverlay


def validate_overlay(overlay: CQEOverlay) -> Tuple[bool, Optional[str]]:
    """
    Validate CQE overlay structure and constraints.

    Args:
        overlay: Overlay to validate

    Returns:
        (is_valid, error_message) tuple
    """
    # Check slot counts
    if len(overlay.present) != 248:
        return False, f"Invalid present array size: {len(overlay.present)}"

    if len(overlay.w) != 248:
        return False, f"Invalid weight array size: {len(overlay.w)}"

    if len(overlay.phi) != 248:
        return False, f"Invalid phase array size: {len(overlay.phi)}"

    # Check Cartan lane count
    cartan_active = overlay.cartan_active
    if cartan_active > 8:
        return False, f"Too many active Cartan lanes: {cartan_active}"

    # Check weight constraints
    active_weights = overlay.w[overlay.active_slots]
    if np.any(active_weights < 0):
        return False, "Negative weights detected"

    if np.any(np.isnan(active_weights)) or np.any(np.isinf(active_weights)):
        return False, "Invalid weight values (NaN or Inf)"

    # Check phase constraints (-π to π)
    active_phases = overlay.phi[overlay.active_slots]
    if np.any(active_phases < -np.pi) or np.any(active_phases > np.pi):
        return False, "Phase values out of range [-π, π]"

    if np.any(np.isnan(active_phases)) or np.any(np.isinf(active_phases)):
        return False, "Invalid phase values (NaN or Inf)"

    return True, None


def validate_features(features: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Validate feature vector for embedding.

    Args:
        features: 8-dimensional feature vector

    Returns:
        (is_valid, error_message) tuple
    """
    if not isinstance(features, np.ndarray):
        return False, "Features must be numpy array"

    if features.shape != (8,):
        return False, f"Features must be 8-dimensional, got {features.shape}"

    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        return False, "Features contain NaN or Inf values"

    return True, None


def validate_phi_components(components: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate Φ component dictionary.

    Args:
        components: Dictionary with Φ components

    Returns:
        (is_valid, error_message) tuple
    """
    required_keys = {'geom', 'parity', 'sparsity', 'kissing'}

    if not all(key in components for key in required_keys):
        missing = required_keys - set(components.keys())
        return False, f"Missing Φ components: {missing}"

    for key, value in components.items():
        if not isinstance(value, (int, float)):
            return False, f"Invalid type for {key}: {type(value)}"

        if np.isnan(value) or np.isinf(value):
            return False, f"Invalid value for {key}: {value}"

    return True, None
