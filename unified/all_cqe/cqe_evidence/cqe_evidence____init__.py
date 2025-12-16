"""
Evidence‑surface decoders for CQE vNext.

Evidence surfaces capture residual patterns left after snapping a vector
to a lattice and can reveal higher‑order structure, such as hidden
parity or shell counts.  This subpackage implements decoders for
interpreting decimal remainders, turning them into actionable
information for the CQE pipeline.

Currently this package contains a placeholder ESD implementation
demonstrating the API; it should be extended with the radix‑robust
decoding described in the master playbook.
"""

from .esd import EvidenceSurfaceDecoder  # noqa: F401
