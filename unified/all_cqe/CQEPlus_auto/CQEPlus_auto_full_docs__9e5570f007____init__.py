"""
Operators for CQE algebra.

This package defines stub implementations of rotation, midpoint and parity
operations used within the CQE system.  These operators should be
extended or replaced with concrete implementations as the framework
matures.  For now they exist purely to satisfy imports from legacy code.
"""

from .rotation import *  # noqa: F401,F403
from .midpoint import *  # noqa: F401,F403
from .parity import *  # noqa: F401,F403

__all__ = []
