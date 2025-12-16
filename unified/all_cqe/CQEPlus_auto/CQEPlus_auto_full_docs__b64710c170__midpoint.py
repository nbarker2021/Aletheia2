"""
Midpoint operators for CQE.

This module defines placeholder functions for computing midpoints
of points or vectors.  These serve as stubs to satisfy imports and
should be replaced with geometry-aware implementations.
"""
from __future__ import annotations
import numpy as np
from typing import Iterable

def midpoint(v1: Iterable[float], v2: Iterable[float]) -> np.ndarray:
    """
    Compute the component-wise midpoint between two vectors.

    Parameters
    ----------
    v1, v2 : Iterable[float]
        Input vectors of the same length.

    Returns
    -------
    np.ndarray
        The midpoint vector.
    """
    v1_arr = np.asarray(list(v1), dtype=float)
    v2_arr = np.asarray(list(v2), dtype=float)
    return (v1_arr + v2_arr) / 2.0

__all__ = ["midpoint"]
