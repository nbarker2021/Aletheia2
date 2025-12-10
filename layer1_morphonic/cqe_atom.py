"""CQE Atom - Universal data structure for CQE framework"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np

@dataclass
class CQEAtom:
    """Universal CQE Atom containing all framework properties"""
    
    # Core identifiers
    atom_id: str
    data_hash: str
    creation_timestamp: float
    
    # CQE properties
    e8_coordinates: np.ndarray
    quad_encoding: Tuple[int, int, int, int]
    parity_channels: np.ndarray
    
    # Sacred geometry properties
    digital_root: int
    sacred_frequency: float
    binary_guidance: str
    rotational_pattern: str
    
    # Mandelbrot properties
    fractal_coordinate: complex
    fractal_behavior: str
    compression_ratio: float
    iteration_depth: int
    
    # Storage properties
    bit_representation: bytes
    storage_size: int
    combination_mask: int
    
    # Metadata
    access_count: int = 0
    combination_history: List[str] = field(default_factory=list)
    validation_status: str = "PENDING"
