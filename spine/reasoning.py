"""
Reasoning Engine - Slice Execution

Routes atoms to appropriate CQE slices for processing.
Manages the slice registry and execution.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import numpy as np

from spine.kernel import CQEAtom
from spine.speedlight import get_speedlight, receipted


@dataclass
class SliceResult:
    """Result from a slice execution."""
    atom: CQEAtom
    slice_name: str
    metadata: Dict[str, Any]
    success: bool
    message: str


class CQESlice:
    """Base class for CQE slices."""
    
    name: str = "base"
    description: str = "Base slice"
    
    def execute(self, atom: CQEAtom, **kwargs) -> SliceResult:
        """Execute the slice on an atom."""
        raise NotImplementedError


class MORSRSlice(CQESlice):
    """
    CQE-MORSR: Multi-Objective Randomized Search and Repair.
    
    Optimizes atom state through random exploration with constraints.
    """
    
    name = "morsr"
    description = "Gauge-Pose Ledger Build - optimization through random search"
    
    def __init__(self, operators: Optional[Dict[str, Callable]] = None):
        self.operators = operators or {}
    
    def execute(self, atom: CQEAtom, budget: int = 10, **kwargs) -> SliceResult:
        """Run MORSR optimization."""
        best_atom = atom
        best_phi = atom.phi()
        steps = 0
        accepts = 0
        
        for _ in range(budget):
            steps += 1
            # Random perturbation
            candidate_lanes = best_atom.lanes + np.random.randn(8) * 0.1
            candidate = CQEAtom(
                lanes=candidate_lanes,
                parity=np.array([int(x >= 0) for x in candidate_lanes]),
                metadata=best_atom.metadata,
                provenance=best_atom.provenance
            )
            
            # Accept if phi decreases (improvement)
            candidate_phi = candidate.phi()
            if candidate_phi < best_phi:
                best_atom = candidate
                best_phi = candidate_phi
                accepts += 1
        
        return SliceResult(
            atom=best_atom,
            slice_name=self.name,
            metadata={"steps": steps, "accepts": accepts, "final_phi": best_phi},
            success=True,
            message=f"MORSR completed: {accepts}/{steps} accepts"
        )


class SACNUMSlice(CQESlice):
    """
    CQE-SACNUM: Sacred Numerology.
    
    Applies sacred geometry and digital root analysis.
    """
    
    name = "sacnum"
    description = "Sacred Numerology - digital root and sacred frequency analysis"
    
    SACRED_FREQUENCIES = [111, 222, 333, 444, 555, 666, 777, 888, 999]
    
    def execute(self, atom: CQEAtom, **kwargs) -> SliceResult:
        """Analyze sacred numerology properties."""
        dr = atom.digital_root()
        phi_val = atom.phi()
        
        # Check for sacred frequency alignment
        phi_int = int(phi_val * 1000)
        sacred_match = None
        for freq in self.SACRED_FREQUENCIES:
            if phi_int % freq == 0:
                sacred_match = freq
                break
        
        metadata = {
            "digital_root": dr,
            "phi_value": phi_val,
            "sacred_match": sacred_match,
            "dr_valid": dr in [0, 9]
        }
        
        return SliceResult(
            atom=atom,
            slice_name=self.name,
            metadata=metadata,
            success=True,
            message=f"SACNUM: DR={dr}, sacred_match={sacred_match}"
        )


class SPECTRALSlice(CQESlice):
    """
    CQE-SPECTRAL: Graph Laplacian and Eigenroutes.
    
    Spectral analysis of geometric structure.
    """
    
    name = "spectral"
    description = "Graph Laplacian, Eigenroutes & Rayleigh Gates"
    
    def execute(self, atom: CQEAtom, **kwargs) -> SliceResult:
        """Perform spectral analysis."""
        # Create a simple Laplacian from the atom's lanes
        lanes = atom.lanes
        L = np.diag(np.abs(lanes)) - np.outer(lanes, lanes) / (np.linalg.norm(lanes) + 1e-6)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(L)
        
        metadata = {
            "eigenvalues": eigenvalues.tolist(),
            "spectral_gap": float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0,
            "trace": float(np.trace(L))
        }
        
        return SliceResult(
            atom=atom,
            slice_name=self.name,
            metadata=metadata,
            success=True,
            message=f"SPECTRAL: gap={metadata['spectral_gap']:.4f}"
        )


class ReasoningEngine:
    """
    Reasoning Engine - Routes atoms to CQE slices.
    
    Manages slice registration and execution routing.
    """
    
    def __init__(self):
        self.slices: Dict[str, CQESlice] = {}
        self.speedlight = get_speedlight()
        
        # Register default slices
        self.register_slice(MORSRSlice())
        self.register_slice(SACNUMSlice())
        self.register_slice(SPECTRALSlice())
    
    def register_slice(self, slice_instance: CQESlice):
        """Register a CQE slice."""
        self.slices[slice_instance.name] = slice_instance
    
    @receipted("slice_execution")
    def route(self, atom: CQEAtom, slice_name: str, **kwargs) -> SliceResult:
        """Route an atom to a specific slice."""
        if slice_name not in self.slices:
            return SliceResult(
                atom=atom,
                slice_name=slice_name,
                metadata={},
                success=False,
                message=f"Unknown slice: {slice_name}"
            )
        
        return self.slices[slice_name].execute(atom, **kwargs)
    
    def auto_route(self, atom: CQEAtom) -> SliceResult:
        """Automatically select and execute the best slice."""
        # Simple heuristic: run SACNUM first for analysis
        sacnum_result = self.route(atom, "sacnum")
        
        # If digital root is not valid, run MORSR to optimize
        if not sacnum_result.metadata.get("dr_valid", True):
            return self.route(atom, "morsr", budget=20)
        
        return sacnum_result
    
    def list_slices(self) -> List[Dict[str, str]]:
        """List all registered slices."""
        return [
            {"name": s.name, "description": s.description}
            for s in self.slices.values()
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get reasoning engine status."""
        return {
            "slices": list(self.slices.keys()),
            "active": True
        }
