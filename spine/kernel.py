"""
CQE Kernel - Central Orchestrator

The kernel is the heart of the system. All operations flow through it.
It manages state, routes operations, and coordinates all components.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spine.speedlight import SpeedLight, get_speedlight, Receipt


class GaugeMode(Enum):
    """Available gauge rotation modes."""
    IDENTITY = "I"
    HADAMARD = "H8"
    QR = "QR"
    SIGNFLIP_HADAMARD = "signflip_H8"


@dataclass
class CQEAtom:
    """
    The universal data unit in the CQE system.
    
    Everything is an atom. Atoms have geometry, parity, and provenance.
    """
    lanes: np.ndarray  # 8D vector (E8 coordinates)
    parity: np.ndarray  # 8 parity channels
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: str = ""
    
    @property
    def parity_ok(self) -> bool:
        """Check if parity is valid (all channels sum to even)."""
        return int(np.sum(self.parity)) % 2 == 0
    
    def phi(self) -> float:
        """Calculate the Phi objective value."""
        return float(np.linalg.norm(self.lanes))
    
    def digital_root(self) -> int:
        """Calculate the digital root of the atom."""
        total = int(np.sum(np.abs(self.lanes) * 1000)) % 9
        return total if total != 0 else 9
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lanes": self.lanes.tolist(),
            "parity": self.parity.tolist(),
            "metadata": self.metadata,
            "provenance": self.provenance,
            "parity_ok": self.parity_ok,
            "phi": self.phi(),
            "digital_root": self.digital_root()
        }
    
    @classmethod
    def from_vector(cls, vec: List[float], provenance: str = "") -> 'CQEAtom':
        """Create an atom from a raw vector."""
        lanes = np.array(vec[:8] if len(vec) >= 8 else vec + [0.0] * (8 - len(vec)))
        parity = np.array([int(x >= 0) for x in lanes])
        return cls(lanes=lanes, parity=parity, provenance=provenance)


class CQEKernel:
    """
    The CQE Kernel - Central Orchestrator.
    
    All operations flow through the kernel. It:
    - Manages the current gauge (rotation mode)
    - Routes operations to appropriate handlers
    - Coordinates with SpeedLight for receipts
    - Enforces basic geometric constraints
    """
    
    def __init__(self, speedlight: Optional[SpeedLight] = None):
        self.speedlight = speedlight or get_speedlight()
        self.gauge = GaugeMode.IDENTITY
        self.tau_w = 0.03  # Boundary threshold
        self.tau_annih = 0.003  # Annihilation threshold
        self.slices: Dict[str, Callable] = {}
        self.operators: Dict[str, Callable] = {}
        self._current_atom: Optional[CQEAtom] = None
    
    def set_gauge(self, mode: GaugeMode):
        """Set the current gauge rotation mode."""
        self.gauge = mode
    
    def ratchet(self, factor: float = 0.9):
        """Tighten thresholds by the given factor."""
        self.tau_w *= factor
        self.tau_annih *= factor
    
    def register_slice(self, name: str, handler: Callable):
        """Register a CQE slice handler."""
        self.slices[name] = handler
    
    def register_operator(self, name: str, handler: Callable):
        """Register an operator."""
        self.operators[name] = handler
    
    def process(self, atom: CQEAtom, operation: str = "default", **kwargs) -> CQEAtom:
        """
        Main processing entry point.
        
        All operations go through here. SpeedLight receipt is generated.
        """
        old_phi = atom.phi()
        
        # Apply gauge rotation if needed
        if self.gauge != GaugeMode.IDENTITY:
            atom = self._apply_gauge(atom)
        
        # Route to appropriate handler
        if operation in self.slices:
            result = self.slices[operation](atom, **kwargs)
        elif operation in self.operators:
            result = self.operators[operation](atom, **kwargs)
        else:
            # Default: pass through
            result = atom
        
        # Ensure result is an atom
        if not isinstance(result, CQEAtom):
            result = CQEAtom.from_vector(list(result) if hasattr(result, '__iter__') else [float(result)])
        
        # Generate receipt
        new_phi = result.phi()
        self.speedlight.create_receipt(
            operation=operation,
            input_data=atom.to_dict(),
            output_data=result.to_dict(),
            old_phi=old_phi,
            new_phi=new_phi,
            parity_ok=result.parity_ok
        )
        
        self._current_atom = result
        return result
    
    def _apply_gauge(self, atom: CQEAtom) -> CQEAtom:
        """Apply the current gauge rotation to an atom."""
        if self.gauge == GaugeMode.HADAMARD:
            # Simplified Hadamard-like rotation
            H = np.ones((8, 8)) / np.sqrt(8)
            new_lanes = H @ atom.lanes
        elif self.gauge == GaugeMode.SIGNFLIP_HADAMARD:
            H = np.ones((8, 8)) / np.sqrt(8)
            signs = np.array([1, -1, 1, -1, 1, -1, 1, -1])
            new_lanes = (H @ atom.lanes) * signs
        else:
            new_lanes = atom.lanes
        
        return CQEAtom(
            lanes=new_lanes,
            parity=atom.parity,
            metadata=atom.metadata,
            provenance=atom.provenance
        )
    
    def validate(self, atom: CQEAtom) -> bool:
        """Check if an atom is geometrically valid."""
        # Check parity
        if not atom.parity_ok:
            return False
        # Check digital root (0 or 9 is valid)
        dr = atom.digital_root()
        if dr not in [0, 9]:
            # Not strictly invalid, but noted
            pass
        return True
    
    def route(self, atom: CQEAtom, slice_name: str, **kwargs) -> Any:
        """Route an atom to a specific slice."""
        if slice_name not in self.slices:
            raise ValueError(f"Unknown slice: {slice_name}")
        return self.process(atom, operation=slice_name, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current kernel status."""
        return {
            "gauge": self.gauge.value,
            "tau_w": self.tau_w,
            "tau_annih": self.tau_annih,
            "registered_slices": list(self.slices.keys()),
            "registered_operators": list(self.operators.keys()),
            "speedlight_summary": self.speedlight.get_summary()
        }
