"""
Mathematical utility functions
"""

import numpy as np
from typing import Tuple


def normalize_vector(vector: np.ndarray, method: str = "l2") -> np.ndarray:
    """
    Normalize vector using specified method.

    Args:
        vector: Input vector
        method: Normalization method ("l2", "l1", "max")

    Returns:
        Normalized vector
    """
    if method == "l2":
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-10 else vector

    elif method == "l1":
        norm = np.sum(np.abs(vector))
        return vector / norm if norm > 1e-10 else vector

    elif method == "max":
        max_val = np.max(np.abs(vector))
        return vector / max_val if max_val > 1e-10 else vector

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity [-1, 1]
    """
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    return dot_product / (norm1 * norm2)


def angular_distance(phi1: float, phi2: float) -> float:
    """
    Compute angular distance between two phases.

    Handles wraparound at ±π.

    Args:
        phi1: First phase [-π, π]
        phi2: Second phase [-π, π]

    Returns:
        Angular distance [0, π]
    """
    diff = abs(phi1 - phi2)

    # Handle wraparound
    if diff > np.pi:
        diff = 2 * np.pi - diff

    return diff


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide, returning default on division by zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero

    Returns:
        Result of division or default
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def quantize_angle(angle: float, quantum: float = np.pi/12) -> float:
    """
    Quantize angle to nearest quantum multiple.

    Args:
        angle: Input angle
        quantum: Quantum step size

    Returns:
        Quantized angle
    """
    return np.round(angle / quantum) * quantum


def compute_entropy(probabilities: np.ndarray) -> float:
    """
    Compute Shannon entropy of probability distribution.

    Args:
        probabilities: Probability distribution (should sum to 1)

    Returns:
        Entropy in bits
    """
    # Filter out zeros to avoid log(0)
    p = probabilities[probabilities > 1e-10]

    if len(p) == 0:
        return 0.0

    return -np.sum(p * np.log2(p))
