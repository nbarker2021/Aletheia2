# CQE-MORSR API Reference

## Core Classes

### CQERunner

Main orchestrator for CQE system operations.

```python
class CQERunner:
    def __init__(self, e8_embedding_path: str = "embeddings/e8_248_embedding.json", 
                 config: Optional[Dict] = None)

    def solve_problem(self, problem_description: Dict, 
                     domain_type: str = "computational") -> Dict[str, Any]

    def run_test_suite(self) -> Dict[str, bool]

    def benchmark_performance(self, problem_sizes: List[int] = [10, 50, 100, 200]) -> Dict
```

### DomainAdapter

Converts problems to E₈-compatible feature vectors.

```python
class DomainAdapter:
    def embed_p_problem(self, instance_size: int, complexity_hint: int = 1) -> np.ndarray

    def embed_np_problem(self, instance_size: int, nondeterminism: float = 0.8) -> np.ndarray

    def embed_optimization_problem(self, variables: int, constraints: int,
                                  objective_type: str = "linear") -> np.ndarray

    def embed_scene_problem(self, scene_complexity: int, narrative_depth: int,
                           character_count: int) -> np.ndarray

    def hash_to_features(self, data: str) -> np.ndarray

    def validate_features(self, features: np.ndarray) -> bool
```

### E8Lattice

E₈ lattice operations and geometric computations.

```python
class E8Lattice:
    def __init__(self, embedding_path: str = "embeddings/e8_248_embedding.json")

    def nearest_root(self, vector: np.ndarray) -> Tuple[int, np.ndarray, float]

    def determine_chamber(self, vector: np.ndarray) -> Tuple[str, np.ndarray]

    def project_to_chamber(self, vector: np.ndarray, 
                          target_chamber: str = "11111111") -> np.ndarray

    def chamber_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float

    def root_embedding_quality(self, vector: np.ndarray) -> Dict[str, float]

    def generate_chamber_samples(self, chamber_sig: str, count: int = 10) -> np.ndarray
```

### ParityChannels

Parity extraction and error correction operations.

```python
class ParityChannels:
    def extract_channels(self, vector: np.ndarray) -> Dict[str, float]

    def enforce_parity(self, vector: np.ndarray, 
                      target_channels: Dict[str, float]) -> np.ndarray

    def calculate_parity_penalty(self, vector: np.ndarray, 
                               reference_channels: Dict[str, float]) -> float

    def golay_encode(self, data_bits: np.ndarray) -> np.ndarray

    def hamming_encode(self, data_bits: np.ndarray) -> np.ndarray

    def detect_syndrome(self, received: np.ndarray, 
                       code_type: str = "hamming") -> Tuple[bool, np.ndarray]

    def channel_statistics(self, vectors: List[np.ndarray]) -> Dict[str, Dict[str, float]]
```

### CQEObjectiveFunction

Multi-component objective function for optimization.

```python
class CQEObjectiveFunction:
    def __init__(self, e8_lattice: E8Lattice, parity_channels: ParityChannels)

    def evaluate(self, vector: np.ndarray, reference_channels: Dict[str, float],
                domain_context: Optional[Dict] = None) -> Dict[str, float]

    def gradient(self, vector: np.ndarray, reference_channels: Dict[str, float],
                domain_context: Optional[Dict] = None, epsilon: float = 1e-5) -> np.ndarray

    def suggest_improvement_direction(self, vector: np.ndarray,
                                    reference_channels: Dict[str, float],
                                    domain_context: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, str]]

    def set_weights(self, new_weights: Dict[str, float])
```

### MORSRExplorer

Multi-objective random search and repair algorithm.

```python
class MORSRExplorer:
    def __init__(self, objective_function: CQEObjectiveFunction,
                 parity_channels: ParityChannels, random_seed: Optional[int] = None)

    def explore(self, initial_vector: np.ndarray, reference_channels: Dict[str, float],
               max_iterations: int = 50, domain_context: Optional[Dict] = None,
               convergence_threshold: float = 1e-4) -> Tuple[np.ndarray, Dict[str, float], float]

    def pulse_exploration(self, vector: np.ndarray, reference_channels: Dict[str, float],
                         pulse_count: int = 10, domain_context: Optional[Dict] = None) -> List[Tuple[np.ndarray, float]]

    def set_parameters(self, pulse_size: Optional[float] = None,
                      repair_threshold: Optional[float] = None,
                      exploration_decay: Optional[float] = None,
                      parity_enforcement_strength: Optional[float] = None)

    def exploration_statistics(self, history: Dict) -> Dict[str, float]
```

### ChamberBoard

CBC enumeration and gate configuration management.

```python
class ChamberBoard:
    def enumerate_gates(self, max_count: Optional[int] = None) -> List[Dict]

    def generate_gate_vector(self, gate_config: Dict, index: int = 0) -> np.ndarray

    def explore_gate_sequence(self, gates: List[Dict], sequence_length: int = 5) -> List[np.ndarray]

    def analyze_gate_coverage(self, gates: List[Dict]) -> Dict[str, int]

    def validate_enumeration(self, gates: List[Dict]) -> Dict[str, bool]

    def reset_enumeration(self)
```

## Enumerations

### ConstructionType

```python
class ConstructionType(Enum):
    A = "A"  # Corner cells
    B = "B"  # Edge cells
    C = "C"  # Center cells  
    D = "D"  # Mixed patterns
```

### PolicyChannel

```python
class PolicyChannel(Enum):
    TYPE_1 = 1  # Linear progression
    TYPE_2 = 2  # Exponential progression
    TYPE_3 = 3  # Logarithmic progression
    TYPE_4 = 4  # Harmonic progression
    TYPE_5 = 5  # Fibonacci-like progression
    TYPE_6 = 6  # Prime-based progression
    TYPE_7 = 7  # Chaotic progression
    TYPE_8 = 8  # Balanced progression
```

## Data Structures

### Problem Description Format

```python
# Computational problems
{
    "size": int,                    # Problem instance size
    "complexity_class": str,        # "P", "NP", "PSPACE", etc.
    "complexity_hint": int,         # Additional complexity information
    "nondeterminism": float         # For NP problems (0.0 - 1.0)
}

# Optimization problems  
{
    "variables": int,               # Number of variables
    "constraints": int,             # Number of constraints
    "objective_type": str           # "linear", "quadratic", "nonlinear"
}

# Creative problems
{
    "scene_complexity": int,        # Scene complexity (1-100)
    "narrative_depth": int,         # Narrative depth (1-50)
    "character_count": int          # Number of characters
}
```

### Gate Configuration Format

```python
{
    "construction": ConstructionType,    # A, B, C, or D
    "policy_channel": PolicyChannel,     # TYPE_1 through TYPE_8
    "phase": int,                        # 1 or 2
    "gate_id": str,                      # Unique identifier (e.g., "A12")
    "cells": List[Tuple[int, int]],      # Conway frame cell coordinates
    "parameters": Dict[str, Any]         # Policy-specific parameters
}
```

## Constants

```python
# System limits
MAX_ITERATIONS = 1000
MAX_PULSE_COUNT = 100
CONVERGENCE_THRESHOLD = 1e-6

# E₈ parameters
E8_DIMENSION = 8
E8_ROOT_COUNT = 240
CARTAN_MATRIX_SIZE = 8

# Parity channels
PARITY_CHANNEL_COUNT = 8
GOLAY_CODE_LENGTH = 24
HAMMING_CODE_LENGTH = 7

# Conway frame
CONWAY_FRAME_SIZE = 4
TOTAL_GATE_COUNT = 64  # 4 constructions × 8 policies × 2 phases
```
