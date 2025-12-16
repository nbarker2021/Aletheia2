"""
Core harness functionality for the CQE framework.

This module forwards calls to the reference harness implementation in
`testing.harness.core`.  It exposes the public API expected by
client code via the `cqe.harness.core` namespace.
"""

from testing.harness.core import (
    run_harness,
    HarnessResult,
    compute_pose_metrics,
    legalize_update,
    write_receipt,
)  # type: ignore

__all__ = [
    "run_harness",
    "HarnessResult",
    "compute_pose_metrics",
    "legalize_update",
    "write_receipt",
]
