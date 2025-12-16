"""
Parity operators for CQE.

This module provides simple functions for computing parity-related
quantities.  The implementations here are placeholders used to satisfy
imports and may be replaced by more sophisticated logic.
"""
from __future__ import annotations
import numpy as np
from typing import Iterable

def vector_parity(v: Iterable[int]) -> int:
    """
    Compute the parity (0 for even, 1 for odd) of the sum of a vector of integers.

    Parameters
    ----------
    v : Iterable[int]
        The input sequence of integers.

    Returns
    -------
    int
        0 if the sum is even, 1 if the sum is odd.
    """
    total = int(np.sum(list(v)))
    return total & 1

__all__ = ["vector_parity"]
