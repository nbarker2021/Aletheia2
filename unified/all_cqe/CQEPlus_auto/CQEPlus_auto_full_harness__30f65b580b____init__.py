"""
Harness utilities for CQE.

The harness subpackage re-exports the reference harness from the testing
suite.  It provides a stable import path for user code.
"""

from .core import run_harness, HarnessResult, compute_pose_metrics, legalize_update, write_receipt  # noqa: F401

__all__ = [
    "run_harness",
    "HarnessResult",
    "compute_pose_metrics",
    "legalize_update",
    "write_receipt",
]
