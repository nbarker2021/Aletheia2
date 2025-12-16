"""
Morphonic Operation Platform - Spine
The complete structural backbone that all modules plug into.
"""

from spine.kernel import CQEKernel, CQEAtom, GaugeMode
from spine.speedlight import SpeedLight, get_speedlight, Receipt, receipted
from spine.io_manager import IOManager, GeoTokenizer, GeoTransformer
from spine.governance import GovernanceEngine, GovernancePolicy, DeltaPhiPolicy, ParityPolicy
from spine.reasoning import ReasoningEngine, CQESlice, MORSRSlice, SACNUMSlice, SPECTRALSlice
from spine.storage import StorageManager, MDHGNode
from spine.interface import InterfaceManager, InterfaceResponse
from spine.operators import ALENAOperators, get_operators
from spine.lattice import E8Lattice, LeechLattice, NiemeierLattice, LatticeEngine, get_lattice_engine
from spine.slices import get_all_slices, register_all_slices
from spine.uvibs import UVIBSEngine, UVIBSMetrics, get_uvibs
from spine.niqas import NIQASEngine, NIQASResult, get_niqas
from spine.agrm_mdhg import AGRMEngine, MDHGEngine, MDHGNode, get_agrm, get_mdhg
from spine.ledger import Ledger, get_ledger, Provenance, TransactionRecord
from spine.domain_adapter import DomainAdapter, get_domain_adapter, DomainEmbedding
from spine.glyph_lambda import (
    GlyphRegistry, GlyphCalculus, GlyphState,
    get_glyph_registry, get_glyph_calculus,
    OverlayRegistry, HyperpermOracle
)
from spine.conservation import (
    ConservationEngine, ConservationLaw, InformationState,
    get_conservation_engine, shannon_entropy, information_content
)

__all__ = [
    # Core
    'CQEKernel',
    'CQEAtom',
    'GaugeMode',
    # SpeedLight
    'SpeedLight',
    'get_speedlight',
    'Receipt',
    'receipted',
    # I/O
    'IOManager',
    'GeoTokenizer',
    'GeoTransformer',
    # Governance
    'GovernanceEngine',
    'GovernancePolicy',
    'DeltaPhiPolicy',
    'ParityPolicy',
    # Reasoning
    'ReasoningEngine',
    'CQESlice',
    'MORSRSlice',
    'SACNUMSlice',
    'SPECTRALSlice',
    # Storage
    'StorageManager',
    'MDHGNode',
    # Interface
    'InterfaceManager',
    'InterfaceResponse',
    # Operators
    'ALENAOperators',
    'get_operators',
    # Lattice
    'E8Lattice',
    'LeechLattice',
    'NiemeierLattice',
    'LatticeEngine',
    'get_lattice_engine',
    # Extended Slices
    'get_all_slices',
    'register_all_slices',
    # UVIBS
    'UVIBSEngine',
    'UVIBSMetrics',
    'get_uvibs',
    # NIQAS
    'NIQASEngine',
    'NIQASResult',
    'get_niqas',
    # AGRM/MDHG
    'AGRMEngine',
    'MDHGEngine',
    'get_agrm',
    'get_mdhg',
    # Ledger
    'Ledger',
    'get_ledger',
    'Provenance',
    'TransactionRecord',
    # Domain Adapter
    'DomainAdapter',
    'get_domain_adapter',
    'DomainEmbedding',
    # Glyph Lambda
    'GlyphRegistry',
    'GlyphCalculus',
    'GlyphState',
    'get_glyph_registry',
    'get_glyph_calculus',
    'OverlayRegistry',
    'HyperpermOracle',
    # Conservation
    'ConservationEngine',
    'ConservationLaw',
    'InformationState',
    'get_conservation_engine',
    'shannon_entropy',
    'information_content',
]
