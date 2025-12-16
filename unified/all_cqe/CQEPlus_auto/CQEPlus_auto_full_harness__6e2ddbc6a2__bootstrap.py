"""
Bootstrap entry points for the CQE framework.

This module simply re-exports the master bootstrap script from
`interfaces.cli.bootstrap` so that it can be invoked via the canonical
`cqe.cli.bootstrap` namespace.  See that module for implementation
details.
"""

# Forward all public names from interfaces.cli.bootstrap into this namespace.
from interfaces.cli.bootstrap import *  # type: ignore  # noqa: F401,F403

__all__ = [name for name in globals().keys() if not name.startswith("_")]
