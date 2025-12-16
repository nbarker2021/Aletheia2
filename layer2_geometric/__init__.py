"""
Layer 2: Core Geometric Engine

Provides fundamental lattice structures and geometric operations:
- E8 lattice (8D, 240 roots)
- Leech lattice (24D, 196,560 minimal vectors)
- 24 Niemeier lattices
- Weyl group navigation
- ALENA tensor operations
"""

from .e8 import E8Lattice
from .leech import LeechLattice
from .niemeier import NiemeierLattice, NiemeierFamily
from .weyl import WeylChamberNavigator, ChamberInfo
from .quaternion import Quaternion

__all__ = ['E8Lattice', 'LeechLattice', 'NiemeierLattice', 'NiemeierFamily', 'WeylChamberNavigator', 'ChamberInfo', 'Quaternion']
