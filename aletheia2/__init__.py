"""
Aletheia2 - Morphonic Operation Platform
========================================

A complete AI reasoning system using geometric embeddings, lattice structures,
and constraint-first reasoning to eliminate ambiguity before computation.

Core Components:
- UnifiedRuntime: Main entry point with SpeedLight caching and receipts
- GeoTransformer: E8-constrained geometric transformer
- SpeedLightV2: Idempotent caching with Merkle ledger
- LambdaE8Builder: Lambda calculus for geometric operations

Example:
    from aletheia2 import UnifiedRuntime
    runtime = UnifiedRuntime()
    state = runtime.process([1, 2, 3, 4, 5, 6, 7, 8])
    print(state.receipt)  # Every operation generates a receipt
"""

__version__ = "2.0.0"
__author__ = "Manus AI"

# Core runtime
from unified_runtime import UnifiedRuntime, UnifiedRuntimeState, OperationReceipt

# Geometric transformer
from geo_transformer import GeoTransformer, TransformerConfig, GeometricAttention

# SpeedLight caching
from morphonic_cqe_unified.sidecar.speedlight_sidecar_plus import SpeedLightV2, SpeedLightPlus

# Lambda E8 calculus
from morphonic_cqe_unified.experimental.lambda_e8_calculus import (
    LambdaE8Builder, LambdaTerm, LambdaType, GeometricLambdaCapture
)

# Core layers
from layer1_morphonic import UniversalMorphon, MGLCEngine
from layer2_geometric import E8Lattice, LeechLattice
from layer3_operational import ConservationEnforcer, MORSRExplorer
from layer4_governance import GravitationalLayer, SevenWitness, DigitalRoot
from layer5_interface import CQESDK

__all__ = [
    # Version
    "__version__",
    # Runtime
    "UnifiedRuntime",
    "UnifiedRuntimeState",
    "OperationReceipt",
    # Transformer
    "GeoTransformer",
    "TransformerConfig",
    "GeometricAttention",
    # SpeedLight
    "SpeedLightV2",
    "SpeedLightPlus",
    # Lambda
    "LambdaE8Builder",
    "LambdaTerm",
    "LambdaType",
    "GeometricLambdaCapture",
    # Layers
    "UniversalMorphon",
    "MGLCEngine",
    "E8Lattice",
    "LeechLattice",
    "ConservationEnforcer",
    "MORSRExplorer",
    "GravitationalLayer",
    "SevenWitness",
    "DigitalRoot",
    "CQESDK",
]
