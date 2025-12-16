"""
Morphonic Operation Platform - Core Module
==========================================

This module provides the core functionality of the Morphonic Operation Platform.

Components:
- SpeedLight: Mandatory receipt generation and audit trail
- GeoPipeline: Geometric transformation and tokenization
- LatticeEngine: E8, Niemeier, and Leech lattice operations
- MorphonicRuntime: Unified runtime orchestrator

Usage:
    from core import get_runtime, process, embed, validate
    
    # Quick embedding
    result = embed("Hello, world!")
    
    # Full processing
    runtime = get_runtime()
    result = runtime.process(data, operation="transform")
"""

from core.speedlight_wrapper import (
    SpeedLightReceipts,
    SpeedLightContext,
    get_speedlight,
    requires_receipt,
    log_transform,
    log_embedding,
    log_governance_check,
    log_lattice_operation
)

from core.geo_pipeline import (
    GeoPipeline,
    GeoTokenizer,
    GeoTransformer,
    E8Projector,
    process_text
)

from core.lattice_engine import (
    LatticeEngine,
    E8Lattice,
    NiemeierLattice,
    LeechLattice,
    get_lattice_engine
)

from core.morphonic_runtime import (
    MorphonicRuntime,
    ProcessingResult,
    Layer,
    get_runtime,
    process,
    embed,
    validate
)

__all__ = [
    # SpeedLight
    "SpeedLightReceipts",
    "SpeedLightContext",
    "get_speedlight",
    "requires_receipt",
    "log_transform",
    "log_embedding",
    "log_governance_check",
    "log_lattice_operation",
    
    # GeoPipeline
    "GeoPipeline",
    "GeoTokenizer",
    "GeoTransformer",
    "E8Projector",
    "process_text",
    
    # LatticeEngine
    "LatticeEngine",
    "E8Lattice",
    "NiemeierLattice",
    "LeechLattice",
    "get_lattice_engine",
    
    # Runtime
    "MorphonicRuntime",
    "ProcessingResult",
    "Layer",
    "get_runtime",
    "process",
    "embed",
    "validate"
]

__version__ = "4.0.0"
