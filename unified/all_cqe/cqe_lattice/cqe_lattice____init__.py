"""
Lattice embeddings for CQE vNext.

This subpackage contains implementations of the high‑dimensional lattices
used by the CQE framework, including the exceptional 8‑dimensional even
unimodular lattice E8 and the 24‑dimensional Leech lattice Λ24.  It also
provides facilities for snapping arbitrary vectors to the nearest lattice
point and canonicalising poses via Weyl group actions.

The current modules contain placeholder classes with simplified API
signatures; future work will implement full quantisation and
reflection algorithms as described in the master playbook.
"""

from .leech24 import Leech24  # noqa: F401
from .e8 import E8Lattice  # noqa: F401
