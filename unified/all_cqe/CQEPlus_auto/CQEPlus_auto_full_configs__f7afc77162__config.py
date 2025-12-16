"""
Configuration utilities for the CQE framework.

This module simply re-exports all objects from the existing
`utils.config` module so that legacy import paths under `cqe.utils`
continue to resolve.  See `utils.config` for implementation details.
"""
from utils.config import *  # type: ignore  # noqa: F401,F403

__all__ = [name for name in globals().keys() if not name.startswith("_")]
