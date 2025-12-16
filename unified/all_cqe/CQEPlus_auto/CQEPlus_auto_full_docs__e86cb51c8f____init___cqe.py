"""
CQE – Cartan Quadratic Equivalence
Public interface: run_harness()
"""
from importlib import metadata as _meta
try:
    __version__ = _meta.version("cqe")
except _meta.PackageNotFoundError:
    __version__ = "dev"

from .harness.core import run_harness  # noqa: E402

__all__ = ["run_harness", "__version__"]
