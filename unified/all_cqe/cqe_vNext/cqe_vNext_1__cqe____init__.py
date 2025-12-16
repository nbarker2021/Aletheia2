"""
Central package for the CQE vNext system.

This package organises modules for the next-generation CQE framework.  The
implementation emphasises a geometry‑first architecture with deterministic
processing, receipts and reproducible state transitions.  Individual
subpackages provide functionality for core mathematical operations (e.g. the
Chinese Remainder Theorem ring scheduler and Φ metric), lattice
embeddings such as the E8 and Leech (Λ24) lattices, evidence‑surface
decoders, and pipeline steps for staging data through the CQE workflow.

During development, these modules serve as placeholders and will be
incrementally populated with the full algorithms described in the master
playbook.  See :mod:`cqe.docs.MASTER_PLAYBOOK` for an overview.
"""

# Expose top‑level subpackages
from . import core  # noqa: F401  # re-export
from . import lattice  # noqa: F401  # re-export
from . import evidence  # noqa: F401  # re-export
from . import pipeline  # noqa: F401  # re-export
