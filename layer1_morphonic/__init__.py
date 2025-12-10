"""
Layer 1: Morphonic Foundation

Provides the categorical and computational foundation:
- Universal Morphon (M₀)
- Observation functors
- Morphonic Lambda Calculus (MGLC)
- Morphonic operations (⊕, ⊗, ∇)
"""

from .morphon import UniversalMorphon, Observation, ObservationType
from .mglc import MGLCEngine, LambdaTerm, LambdaLevel
from .seed_generator import MorphonicSeedGenerator, MorphonSeed
from .master_message import MasterMessage, MessageLayer, CQEPattern, get_master_message

__all__ = [
    'UniversalMorphon',
    'Observation',
    'ObservationType',
    'MGLCEngine',
    'LambdaTerm',
    'LambdaLevel',
    'MorphonicSeedGenerator',
    'MorphonSeed',
    'MasterMessage',
    'MessageLayer',
    'CQEPattern',
    'get_master_message'
]
