# CQE top-level package
"""
The `cqe` package provides a stable alias to core CQE modules.

This lightweight layer exposes a subset of the existing functionality in
the repository under a unified namespace.  It does not implement new
logic itself; rather, it forwards imports to the underlying modules.
"""

# Re-export useful subpackages on import.
from . import cli as cli  # noqa: F401
from . import harness as harness  # noqa: F401
from . import operators as operators  # noqa: F401
from . import utils as utils  # noqa: F401

__all__ = ["cli", "harness", "operators", "utils"]
