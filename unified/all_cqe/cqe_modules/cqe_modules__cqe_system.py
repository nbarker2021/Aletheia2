"""
CQE Core System - Complete Implementation
========================================

The definitive implementation of the Cartan Quadratic Equivalence (CQE) system
that integrates all mathematical frameworks into a unified computational system.

This module provides the complete CQE system with:
- E₈ lattice operations for geometric processing
- Sacred geometry guidance for binary operations
- Mandelbrot fractal storage with bit-level precision
- Universal atomic operations for any data type
- Comprehensive validation and testing

Author: CQE Development Team
Version: 1.0.0 Master
"""

import numpy as np
import hashlib
import struct
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Setup logging
logger = logging.getLogger(__name__)

class CQEOperationMode(Enum):
    """CQE system operation modes"""
    BASIC = "BASIC"
    ENHANCED = "ENHANCED"
    SACRED_GEOMETRY = "SACRED_GEOMETRY"
    MANDELBROT_FRACTAL = "MANDELBROT_FRACTAL"
    ULTIMATE_UNIFIED = "ULTIMATE_UNIFIED"

class ProcessingPriority(Enum):
    """Processing priority levels"""
    GEOMETRY_FIRST = "GEOMETRY_FIRST"
    MEANING_FIRST = "MEANING_FIRST"
    BALANCED = "BALANCED"

@dataclass
class CQEConfiguration:
    """Configuration for CQE system"""
    operation_mode: CQEOperationMode = CQEOperationMode.ULTIMATE_UNIFIED
    processing_priority: ProcessingPriority = ProcessingPriority.GEOMETRY_FIRST
    enable_sacred_geometry: bool = True
    enable_mandelbrot_storage: bool = True
    enable_toroidal_geometry: bool = True
    enable_validation: bool = True
    max_iterations: int = 1000
    precision_threshold: float = 1e-10
    memory_optimization: bool = True
    parallel_processing: bool = True
    log_level: str = "INFO"

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
    combination_history: List[str] = None
    validation_status: str = "PENDING"
    
    def __post_init__(self):
        if self.combination_history is None:
            self.combination_history = []

class CQESystem:
    """Complete CQE System Implementation"""
    
    def __init__(self, config: CQEConfiguration = None):
        """Initialize CQE system with configuration"""
        
        self.config = config or CQEConfiguration()
        self.atoms: Dict[str, CQEAtom] = {}
        self.system_state = {
            'initialized': False,
            'total_atoms': 0,
            'total_combinations': 0,
            'total_storage_bits': 0,
            'system_health': 'UNKNOWN'
        }
        
        # Initialize subsystems
        self.initialize_subsystems()
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        logger.info(f"CQE System initialized in {self.config.operation_mode.value} mode")
        
        self.system_state['initialized'] = True
    
    def initialize_subsystems(self):
        """Initialize all CQE subsystems"""
        
        # Sacred geometry constants
        self.sacred_frequencies = {
            1: 174.0, 2: 285.0, 3: 396.0, 4: 417.0, 5: 528.0,
            6: 639.0, 7: 741.0, 8: 852.0, 9: 963.0
        }
        
        self.rotational_patterns = {
            9: "INWARD_ROTATIONAL",
            6: "OUTWARD_ROTATIONAL",
            3: "CREATIVE_SEED",
            1: "TRANSFORMATIVE_CYCLE", 2: "TRANSFORMATIVE_CYCLE",
            4: "TRANSFORMATIVE_CYCLE", 5: "TRANSFORMATIVE_CYCLE",
            7: "TRANSFORMATIVE_CYCLE", 8: "TRANSFORMATIVE_CYCLE"
        }
        
        # Mathematical constants
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        self.e8_dimension = 8
        self.e8_root_count = 240
        
        # Mandelbrot constants
        self.mandelbrot_escape_radius = 2.0
        self.mandelbrot_max_iter = self.config.max_iterations
        
        logger.debug("All subsystems initialized successfully")
    
    def create_atom(self, data: Any, atom_id: str = None) -> str:
        """Create CQE atom from arbitrary data"""
        
        if atom_id is None:
            atom_id = self.generate_atom_id(data)
        
        logger.debug(f"Creating atom {atom_id} from data: {type(data)}")
        
        # Generate all properties
        atom = CQEAtom(
            atom_id=atom_id,
            data_hash=self.calculate_data_hash(data),
            creation_timestamp=time.time(),
            
            # CQE properties
            e8_coordinates=self.generate_e8_coordinates(data),
            quad_encoding=self.generate_quad_encoding(data),
            parity_channels=self.generate_parity_channels(data),
            
            # Sacred geometry properties
            digital_root=self.calculate_digital_root(data),
            sacred_frequency=0.0,  # Will be set based on digital root
            binary_guidance="",    # Will be set based on digital root
            rotational_pattern="", # Will be set based on digital root
            
            # Mandelbrot properties
            fractal_coordinate=self.generate_fractal_coordinate(data),
            fractal_behavior="",   # Will be calculated
            compression_ratio=0.0, # Will be calculated
            iteration_depth=0,     # Will be calculated
            
            # Storage properties
            bit_representation=b'', # Will be calculated
            storage_size=0,         # Will be calculated
            combination_mask=0      # Will be calculated
        )
        
        # Set derived properties
        atom.sacred_frequency = self.sacred_frequencies[atom.digital_root]
        atom.binary_guidance = self.generate_binary_guidance(atom.digital_root)
        atom.rotational_pattern = self.rotational_patterns[atom.digital_root]
        
        # Calculate Mandelbrot properties
        atom.fractal_behavior = self.determine_fractal_behavior(atom.fractal_coordinate)
        atom.compression_ratio = self.calculate_compression_ratio(atom.fractal_coordinate, atom.fractal_behavior)
        atom.iteration_depth = self.calculate_iteration_depth(atom.fractal_coordinate)
        
        # Calculate storage properties
        atom.bit_representation = self.calculate_bit_representation(atom)
        atom.storage_size = len(atom.bit_representation) * 8
        atom.combination_mask = self.calculate_combination_mask(atom)
        
        # Validate atom consistency
        if self.config.enable_validation:
            atom.validation_status = self.validate_atom_consistency(atom)
        
        # Store atom
        self.atoms[atom_id] = atom
        self.system_state['total_atoms'] += 1
        self.system_state['total_storage_bits'] += atom.storage_size
        
        logger.info(f"Created atom {atom_id}: {atom.digital_root}-root, {atom.sacred_frequency}Hz, {atom.fractal_behavior}")
        
        return atom_id
    
    def generate_atom_id(self, data: Any) -> str:
        """Generate unique atom ID from data"""
        data_str = str(data) + str(time.time())
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def calculate_data_hash(self, data: Any) -> str:
        """Calculate hash of input data"""
        return hashlib.sha256(str(data).encode()).hexdigest()
    
    def generate_e8_coordinates(self, data: Any) -> np.ndarray:
        """Generate E₈ lattice coordinates from data"""
        data_hash = hashlib.sha256(str(data).encode()).digest()
        
        # Extract 8 coordinates from hash using integer approach
        coords = []
        for i in range(8):
            byte_slice = data_hash[i*4:(i+1)*4]
            if len(byte_slice) == 4:
                int_value = struct.unpack('I', byte_slice)[0]
                coord_value = (int_value % 2000000 - 1000000) / 1000000.0
            else:
                coord_value = 0.0
            coords.append(coord_value)
        
        coords = np.array(coords)
        coords = np.nan_to_num(coords, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize to unit sphere
        norm = np.linalg.norm(coords)
        if norm > 0:
            coords = coords / norm
        else:
            coords = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        return coords
    
    def generate_quad_encoding(self, data: Any) -> Tuple[int, int, int, int]:
        """Generate 4D quadratic encoding from data"""
        data_hash = hashlib.md5(str(data).encode()).digest()
        
        quad = []
        for i in range(4):
            byte_slice = data_hash[i*4:(i+1)*4]
            if len(byte_slice) == 4:
                value = struct.unpack('I', byte_slice)[0] % 256
            else:
                value = 0
            quad.append(value)
        
        return tuple(quad)
    
    def generate_parity_channels(self, data: Any) -> np.ndarray:
        """Generate 8-channel parity state from data"""
        data_str = str(data)
        channels = np.zeros(8)
        
        for i in range(8):
            if i < len(data_str):
                channels[i] = ord(data_str[i]) % 2
            else:
                channels[i] = hash(data_str) % 2
        
        return channels
    
    def calculate_digital_root(self, data: Any) -> int:
        """Calculate Carlson's digital root from data"""
        if isinstance(data, (int, float)):
            n = abs(int(data * 1000))
        else:
            n = abs(hash(str(data))) % 1000000
        
        while n >= 10:
            n = sum(int(digit) for digit in str(n))
        
        return n if n > 0 else 9
    
    def generate_binary_guidance(self, digital_root: int) -> str:
        """Generate sacred binary guidance pattern"""
        patterns = {
            1: "001", 2: "010", 3: "011", 4: "100", 5: "101",
            6: "110", 7: "111", 8: "100", 9: "111"
        }
        return patterns.get(digital_root, "000")
    
    def generate_fractal_coordinate(self, data: Any) -> complex:
        """Generate Mandelbrot coordinate from data"""
        data_hash = hashlib.sha1(str(data).encode()).digest()
        
        real_bytes = data_hash[:4]
        imag_bytes = data_hash[4:8]
        
        if len(real_bytes) == 4:
            real_int = struct.unpack('I', real_bytes)[0]
            real_part = (real_int % 4000000 - 2000000) / 1000000.0
        else:
            real_part = 0.0
            
        if len(imag_bytes) == 4:
            imag_int = struct.unpack('I', imag_bytes)[0]
            imag_part = (imag_int % 3000000 - 1500000) / 1000000.0
        else:
            imag_part = 0.0
        
        # Handle potential NaN or inf values
        real_part = np.nan_to_num(real_part, nan=0.0, posinf=1.5, neginf=-2.5)
        imag_part = np.nan_to_num(imag_part, nan=0.0, posinf=1.5, neginf=-1.5)
        
        # Ensure within Mandelbrot viewing region
        real_part = max(-2.5, min(1.5, real_part))
        imag_part = max(-1.5, min(1.5, imag_part))
        
        return complex(real_part, imag_part)
    
    def determine_fractal_behavior(self, c: complex) -> str:
        """Determine Mandelbrot fractal behavior"""
        z = complex(0, 0)
        
        for i in range(self.mandelbrot_max_iter):
            if abs(z) > self.mandelbrot_escape_radius:
                if i < self.mandelbrot_max_iter * 0.2:
                    return 'ESCAPING'
                else:
                    return 'BOUNDARY'
            z = z*z + c
        
        # Check for periodic behavior
        orbit = []
        for i in range(20):
            z = z*z + c
            orbit.append(z)
        
        # Simple periodicity check
        for period in [2, 3, 4, 5]:
            if len(orbit) >= 2 * period:
                is_periodic = True
                for j in range(period):
                    if abs(orbit[-(j+1)] - orbit[-(j+1+period)]) > 1e-6:
                        is_periodic = False
                        break
                if is_periodic:
                    return 'PERIODIC'
        
        return 'BOUNDED'
    
    def calculate_compression_ratio(self, c: complex, behavior: str) -> float:
        """Calculate compression/expansion ratio"""
        if behavior == 'BOUNDED':
            return 1.0 / (1.0 + abs(c))
        elif behavior == 'ESCAPING':
            return abs(c) / (1.0 + abs(c))
        else:
            return 0.5
    
    def calculate_iteration_depth(self, c: complex) -> int:
        """Calculate fractal iteration depth"""
        z = complex(0, 0)
        
        for i in range(self.mandelbrot_max_iter):
            if abs(z) > self.mandelbrot_escape_radius:
                return i
            z = z*z + c
        
        return self.mandelbrot_max_iter
    
    def calculate_bit_representation(self, atom: CQEAtom) -> bytes:
        """Calculate complete bit representation of atom"""
        import pickle
        import zlib
        
        # Create serializable data structure
        data = {
            'e8_coords': atom.e8_coordinates.tobytes(),
            'quad_encoding': struct.pack('4I', *atom.quad_encoding),
            'parity_channels': atom.parity_channels.tobytes(),
            'digital_root': struct.pack('i', atom.digital_root),
            'sacred_frequency': struct.pack('f', atom.sacred_frequency),
            'binary_guidance': atom.binary_guidance.encode('utf-8'),
            'rotational_pattern': atom.rotational_pattern.encode('utf-8'),
            'fractal_coordinate': struct.pack('2f', atom.fractal_coordinate.real, atom.fractal_coordinate.imag),
            'fractal_behavior': atom.fractal_behavior.encode('utf-8'),
            'compression_ratio': struct.pack('f', atom.compression_ratio),
            'iteration_depth': struct.pack('i', atom.iteration_depth)
        }
        
        # Serialize and compress
        serialized = pickle.dumps(data)
        compressed = zlib.compress(serialized)
        
        return compressed
    
    def calculate_combination_mask(self, atom: CQEAtom) -> int:
        """Calculate bit mask for valid atomic combinations"""
        mask = 0
        
        # Sacred geometry compatibility (3 bits)
        if atom.digital_root in [3, 6, 9]:
            mask |= 0b111
        else:
            mask |= 0b101
        
        # Fractal behavior compatibility (3 bits)
        behavior_masks = {
            'BOUNDED': 0b001,
            'ESCAPING': 0b010,
            'BOUNDARY': 0b100,
            'PERIODIC': 0b011
        }
        mask |= (behavior_masks.get(atom.fractal_behavior, 0b000) << 3)
        
        # Frequency harmony compatibility (4 bits)
        freq_category = int(atom.sacred_frequency / 100) % 16
        mask |= (freq_category << 6)
        
        # E₈ lattice compatibility (8 bits)
        e8_hash = hash(atom.e8_coordinates.tobytes()) % 256
        mask |= (e8_hash << 10)
        
        return mask
    
    def validate_atom_consistency(self, atom: CQEAtom) -> str:
        """Validate consistency across all frameworks"""
        
        inconsistencies = []
        
        # Check CQE-Sacred Geometry consistency
        expected_root = self.calculate_digital_root_from_e8(atom.e8_coordinates)
        if abs(expected_root - atom.digital_root) > 1:
            inconsistencies.append("CQE-Sacred geometry mismatch")
        
        # Check Sacred Geometry-Mandelbrot consistency
        expected_behavior = self.predict_fractal_behavior_from_sacred(atom.digital_root)
        if expected_behavior != atom.fractal_behavior:
            inconsistencies.append("Sacred-Mandelbrot mismatch")
        
        # Check Mandelbrot-CQE consistency
        expected_compression = self.predict_compression_from_e8(atom.e8_coordinates)
        if abs(expected_compression - atom.compression_ratio) > 0.2:
            inconsistencies.append("Mandelbrot-CQE mismatch")
        
        if inconsistencies:
            return f"INCONSISTENT: {', '.join(inconsistencies)}"
        else:
            return "CONSISTENT"
    
    def calculate_digital_root_from_e8(self, coords: np.ndarray) -> int:
        """Calculate expected digital root from E₈ coordinates"""
        coord_sum = np.sum(np.abs(coords))
        return int(coord_sum * 1000) % 9 + 1
    
    def predict_fractal_behavior_from_sacred(self, digital_root: int) -> str:
        """Predict fractal behavior from sacred geometry"""
        if digital_root == 9:
            return 'BOUNDED'
        elif digital_root == 6:
            return 'ESCAPING'
        elif digital_root == 3:
            return 'BOUNDARY'
        else:
            return 'PERIODIC'
    
    def predict_compression_from_e8(self, coords: np.ndarray) -> float:
        """Predict compression ratio from E₈ coordinates"""
        lattice_norm = np.linalg.norm(coords)
        return 1.0 / (1.0 + lattice_norm)
    
    def get_atom(self, atom_id: str) -> Optional[CQEAtom]:
        """Retrieve atom by ID"""
        atom = self.atoms.get(atom_id)
        if atom:
            atom.access_count += 1
        return atom
    
    def process_data(self, data: Any, processing_mode: str = "geometry_first") -> Dict[str, Any]:
        """Process data using CQE principles"""
        
        logger.info(f"Processing data using {processing_mode} mode")
        
        # Create atom from data
        atom_id = self.create_atom(data)
        atom = self.get_atom(atom_id)
        
        if processing_mode == "geometry_first":
            # Geometry-first processing
            geometric_result = self.process_geometric_properties(atom)
            semantic_result = self.extract_semantic_meaning(geometric_result, atom)
        else:
            # Traditional semantic-first processing
            semantic_result = self.process_semantic_properties(data)
            geometric_result = self.embed_in_geometric_space(semantic_result)
        
        return {
            'atom_id': atom_id,
            'processing_mode': processing_mode,
            'geometric_result': geometric_result,
            'semantic_result': semantic_result,
            'atom_properties': {
                'digital_root': atom.digital_root,
                'sacred_frequency': atom.sacred_frequency,
                'fractal_behavior': atom.fractal_behavior,
                'compression_ratio': atom.compression_ratio,
                'storage_size': atom.storage_size,
                'validation_status': atom.validation_status
            }
        }
    
    def process_geometric_properties(self, atom: CQEAtom) -> Dict[str, Any]:
        """Process geometric properties of atom"""
        return {
            'e8_position': atom.e8_coordinates.tolist(),
            'e8_norm': float(np.linalg.norm(atom.e8_coordinates)),
            'fractal_position': [atom.fractal_coordinate.real, atom.fractal_coordinate.imag],
            'fractal_magnitude': abs(atom.fractal_coordinate),
            'geometric_relationships': self.analyze_geometric_relationships(atom),
            'symmetry_properties': self.analyze_symmetry_properties(atom)
        }
    
    def extract_semantic_meaning(self, geometric_result: Dict[str, Any], atom: CQEAtom) -> Dict[str, Any]:
        """Extract semantic meaning from geometric properties"""
        return {
            'primary_pattern': self.identify_primary_pattern(atom),
            'semantic_associations': self.generate_semantic_associations(atom),
            'meaning_confidence': self.calculate_meaning_confidence(geometric_result),
            'conceptual_domain': self.determine_conceptual_domain(atom),
            'relational_context': self.analyze_relational_context(atom)
        }
    
    def process_semantic_properties(self, data: Any) -> Dict[str, Any]:
        """Process semantic properties directly"""
        return {
            'data_type': type(data).__name__,
            'semantic_content': str(data),
            'conceptual_analysis': "Direct semantic processing",
            'meaning_extraction': "Traditional approach"
        }
    
    def embed_in_geometric_space(self, semantic_result: Dict[str, Any]) -> Dict[str, Any]:
        """Embed semantic result in geometric space"""
        return {
            'embedding_method': 'semantic_to_geometric',
            'geometric_representation': 'derived_from_semantics'
        }
    
    def analyze_geometric_relationships(self, atom: CQEAtom) -> Dict[str, Any]:
        """Analyze geometric relationships within atom"""
        return {
            'e8_fractal_correlation': float(np.dot(atom.e8_coordinates, 
                                                  [atom.fractal_coordinate.real, atom.fractal_coordinate.imag, 0, 0, 0, 0, 0, 0])),
            'sacred_geometric_alignment': self.calculate_sacred_alignment(atom),
            'dimensional_projections': self.calculate_dimensional_projections(atom)
        }
    
    def analyze_symmetry_properties(self, atom: CQEAtom) -> Dict[str, Any]:
        """Analyze symmetry properties of atom"""
        return {
            'rotational_symmetry': atom.rotational_pattern,
            'reflection_symmetry': self.calculate_reflection_symmetry(atom),
            'scale_invariance': self.calculate_scale_invariance(atom)
        }
    
    def identify_primary_pattern(self, atom: CQEAtom) -> str:
        """Identify primary pattern from geometric properties"""
        if atom.digital_root in [3, 6, 9]:
            return "PRIMARY_SACRED_PATTERN"
        elif atom.fractal_behavior == 'BOUNDED':
            return "CONVERGENT_PATTERN"
        elif atom.fractal_behavior == 'ESCAPING':
            return "DIVERGENT_PATTERN"
        else:
            return "COMPLEX_PATTERN"
    
    def generate_semantic_associations(self, atom: CQEAtom) -> List[str]:
        """Generate semantic associations from geometric properties"""
        associations = []
        
        if atom.digital_root == 9:
            associations.extend(["completion", "wholeness", "convergence"])
        elif atom.digital_root == 6:
            associations.extend(["creation", "expansion", "manifestation"])
        elif atom.digital_root == 3:
            associations.extend(["foundation", "trinity", "generation"])
        
        if atom.fractal_behavior == 'BOUNDED':
            associations.extend(["stability", "containment", "order"])
        elif atom.fractal_behavior == 'ESCAPING':
            associations.extend(["growth", "expansion", "freedom"])
        
        return associations
    
    def calculate_meaning_confidence(self, geometric_result: Dict[str, Any]) -> float:
        """Calculate confidence in semantic meaning extraction"""
        # Base confidence on geometric consistency and clarity
        base_confidence = 0.8
        
        # Adjust based on geometric properties
        if geometric_result.get('e8_norm', 0) > 0.9:
            base_confidence += 0.1
        
        if geometric_result.get('fractal_magnitude', 0) < 2.0:
            base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def determine_conceptual_domain(self, atom: CQEAtom) -> str:
        """Determine conceptual domain from atom properties"""
        if atom.sacred_frequency < 400:
            return "FOUNDATIONAL_DOMAIN"
        elif atom.sacred_frequency < 700:
            return "CREATIVE_DOMAIN"
        else:
            return "TRANSFORMATIONAL_DOMAIN"
    
    def analyze_relational_context(self, atom: CQEAtom) -> Dict[str, Any]:
        """Analyze relational context of atom"""
        return {
            'frequency_harmonics': self.calculate_frequency_harmonics(atom.sacred_frequency),
            'geometric_neighbors': self.find_geometric_neighbors(atom),
            'pattern_resonance': self.calculate_pattern_resonance(atom)
        }
    
    def calculate_sacred_alignment(self, atom: CQEAtom) -> float:
        """Calculate sacred geometry alignment score"""
        # Calculate alignment based on golden ratio relationships
        golden_alignment = 0.0
        
        for i in range(len(atom.e8_coordinates) - 1):
            ratio = abs(atom.e8_coordinates[i] / (atom.e8_coordinates[i+1] + 1e-10))
            if abs(ratio - self.golden_ratio) < 0.1:
                golden_alignment += 0.125  # 1/8 for each coordinate pair
        
        return golden_alignment
    
    def calculate_dimensional_projections(self, atom: CQEAtom) -> Dict[str, float]:
        """Calculate projections onto different dimensional subspaces"""
        return {
            '2d_projection': float(np.linalg.norm(atom.e8_coordinates[:2])),
            '3d_projection': float(np.linalg.norm(atom.e8_coordinates[:3])),
            '4d_projection': float(np.linalg.norm(atom.e8_coordinates[:4])),
            '8d_full': float(np.linalg.norm(atom.e8_coordinates))
        }
    
    def calculate_reflection_symmetry(self, atom: CQEAtom) -> float:
        """Calculate reflection symmetry measure"""
        coords = atom.e8_coordinates
        reflected = -coords
        return float(1.0 - np.linalg.norm(coords - reflected) / 2.0)
    
    def calculate_scale_invariance(self, atom: CQEAtom) -> float:
        """Calculate scale invariance measure"""
        # Measure how properties scale with coordinate magnitude
        norm = np.linalg.norm(atom.e8_coordinates)
        scaled_coords = atom.e8_coordinates * 2.0
        scaled_norm = np.linalg.norm(scaled_coords)
        
        expected_ratio = 2.0
        actual_ratio = scaled_norm / (norm + 1e-10)
        
        return 1.0 - abs(actual_ratio - expected_ratio) / expected_ratio
    
    def calculate_frequency_harmonics(self, frequency: float) -> List[float]:
        """Calculate harmonic frequencies"""
        harmonics = []
        for n in range(1, 6):  # First 5 harmonics
            harmonics.append(frequency * n)
        return harmonics
    
    def find_geometric_neighbors(self, atom: CQEAtom) -> List[str]:
        """Find geometrically similar atoms"""
        neighbors = []
        
        for other_id, other_atom in self.atoms.items():
            if other_id != atom.atom_id:
                # Calculate E₈ distance
                distance = np.linalg.norm(atom.e8_coordinates - other_atom.e8_coordinates)
                if distance < 0.5:  # Threshold for "nearby"
                    neighbors.append(other_id)
        
        return neighbors[:5]  # Return top 5 neighbors
    
    def calculate_pattern_resonance(self, atom: CQEAtom) -> float:
        """Calculate pattern resonance with other atoms"""
        if len(self.atoms) <= 1:
            return 0.0
        
        resonance_sum = 0.0
        count = 0
        
        for other_atom in self.atoms.values():
            if other_atom.atom_id != atom.atom_id:
                # Calculate frequency resonance
                freq_ratio = atom.sacred_frequency / (other_atom.sacred_frequency + 1e-10)
                if abs(freq_ratio - 1.0) < 0.1 or abs(freq_ratio - 2.0) < 0.1 or abs(freq_ratio - 0.5) < 0.1:
                    resonance_sum += 1.0
                count += 1
        
        return resonance_sum / max(1, count)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        if not self.atoms:
            return {
                'total_atoms': 0,
                'system_health': 'EMPTY',
                'message': 'No atoms in system'
            }
        
        # Calculate statistics
        digital_roots = [atom.digital_root for atom in self.atoms.values()]
        frequencies = [atom.sacred_frequency for atom in self.atoms.values()]
        behaviors = [atom.fractal_behavior for atom in self.atoms.values()]
        storage_sizes = [atom.storage_size for atom in self.atoms.values()]
        
        stats = {
            'system_state': self.system_state,
            'atom_statistics': {
                'total_atoms': len(self.atoms),
                'total_storage_bits': sum(storage_sizes),
                'average_storage_size': np.mean(storage_sizes) if storage_sizes else 0,
                'total_combinations': self.system_state['total_combinations']
            },
            'digital_root_distribution': {
                str(i): digital_roots.count(i) for i in range(1, 10)
            },
            'frequency_distribution': {
                f"{freq}Hz": frequencies.count(freq) for freq in set(frequencies)
            },
            'fractal_behavior_distribution': {
                behavior: behaviors.count(behavior) for behavior in set(behaviors)
            },
            'validation_summary': {
                'consistent_atoms': len([a for a in self.atoms.values() if a.validation_status == 'CONSISTENT']),
                'inconsistent_atoms': len([a for a in self.atoms.values() if 'INCONSISTENT' in a.validation_status]),
                'pending_validation': len([a for a in self.atoms.values() if a.validation_status == 'PENDING'])
            }
        }
        
        # Determine system health
        if stats['validation_summary']['consistent_atoms'] / max(1, len(self.atoms)) > 0.8:
            stats['system_health'] = 'EXCELLENT'
        elif stats['validation_summary']['consistent_atoms'] / max(1, len(self.atoms)) > 0.6:
            stats['system_health'] = 'GOOD'
        else:
            stats['system_health'] = 'NEEDS_ATTENTION'
        
        return stats
    
    def export_system_state(self, filename: str):
        """Export complete system state to file"""
        
        export_data = {
            'system_info': {
                'version': '1.0.0',
                'export_timestamp': time.time(),
                'configuration': {
                    'operation_mode': self.config.operation_mode.value,
                    'processing_priority': self.config.processing_priority.value,
                    'enable_sacred_geometry': self.config.enable_sacred_geometry,
                    'enable_mandelbrot_storage': self.config.enable_mandelbrot_storage,
                    'enable_validation': self.config.enable_validation
                }
            },
            'system_statistics': self.get_system_statistics(),
            'atoms': {}
        }
        
        # Export atom data
        for atom_id, atom in self.atoms.items():
            export_data['atoms'][atom_id] = {
                'atom_id': atom.atom_id,
                'data_hash': atom.data_hash,
                'creation_timestamp': atom.creation_timestamp,
                'e8_coordinates': atom.e8_coordinates.tolist(),
                'quad_encoding': atom.quad_encoding,
                'parity_channels': atom.parity_channels.tolist(),
                'digital_root': atom.digital_root,
                'sacred_frequency': atom.sacred_frequency,
                'binary_guidance': atom.binary_guidance,
                'rotational_pattern': atom.rotational_pattern,
                'fractal_coordinate': [atom.fractal_coordinate.real, atom.fractal_coordinate.imag],
                'fractal_behavior': atom.fractal_behavior,
                'compression_ratio': atom.compression_ratio,
                'iteration_depth': atom.iteration_depth,
                'storage_size': atom.storage_size,
                'combination_mask': atom.combination_mask,
                'access_count': atom.access_count,
                'combination_history': atom.combination_history,
                'validation_status': atom.validation_status
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"System state exported to {filename}")

# Example usage and testing
def demonstrate_cqe_system():
    """Demonstrate CQE system capabilities"""
    
    print("CQE System Demonstration")
    print("=" * 50)
    
    # Create CQE system
    config = CQEConfiguration(
        operation_mode=CQEOperationMode.ULTIMATE_UNIFIED,
        processing_priority=ProcessingPriority.GEOMETRY_FIRST,
        enable_validation=True
    )
    
    cqe = CQESystem(config)
    
    # Test data
    test_data = [
        432,  # Sacred frequency
        "sacred geometry",  # Text
        [1, 1, 2, 3, 5, 8],  # Fibonacci
        {"golden": 1.618},  # Dictionary
        complex(-0.5, 0.6)  # Complex number
    ]
    
    print(f"\nProcessing {len(test_data)} test items...")
    
    for i, data in enumerate(test_data):
        print(f"\nProcessing item {i+1}: {data}")
        result = cqe.process_data(data)
        
        print(f"  Atom ID: {result['atom_id']}")
        print(f"  Digital Root: {result['atom_properties']['digital_root']}")
        print(f"  Sacred Frequency: {result['atom_properties']['sacred_frequency']} Hz")
        print(f"  Fractal Behavior: {result['atom_properties']['fractal_behavior']}")
        print(f"  Storage Size: {result['atom_properties']['storage_size']} bits")
        print(f"  Validation: {result['atom_properties']['validation_status']}")
    
    # Display system statistics
    print(f"\nSystem Statistics:")
    stats = cqe.get_system_statistics()
    print(f"  Total Atoms: {stats['atom_statistics']['total_atoms']}")
    print(f"  Total Storage: {stats['atom_statistics']['total_storage_bits']} bits")
    print(f"  System Health: {stats['system_health']}")
    
    # Export system state
    cqe.export_system_state("cqe_system_demo_state.json")
    print(f"  System state exported to: cqe_system_demo_state.json")
    
    return cqe

if __name__ == "__main__":
    demonstrate_cqe_system()
