"""
ALENA Operators - Integrated into Spine

Wraps the unified cqe_core operators and integrates them with the kernel.
"""

import sys
import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Import from unified
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified.cqe_core.operators import OperatorLibrary as BaseOperatorLibrary

from spine.kernel import CQEAtom
from spine.speedlight import get_speedlight, receipted


class ALENAOperators:
    """
    ALENA Operator Suite - Integrated with Spine.
    
    All operators are receipted and work on CQEAtom objects.
    """
    
    def __init__(self, seed: int = 42):
        self.base = BaseOperatorLibrary(seed)
        self.speedlight = get_speedlight()
    
    def _apply(self, atom: CQEAtom, op_name: str, **kwargs) -> CQEAtom:
        """Apply an operator and return new atom."""
        op_func = getattr(self.base, op_name)
        new_lanes, new_parity_ok = op_func(
            atom.lanes.tolist(), 
            atom.parity_ok, 
            **kwargs
        )
        
        new_atom = CQEAtom(
            lanes=np.array(new_lanes),
            parity=np.array([int(x >= 0) for x in new_lanes]),
            metadata={**atom.metadata, "last_op": op_name},
            provenance=f"{atom.provenance}>{op_name}"
        )
        
        # Override parity_ok based on operator result
        if not new_parity_ok:
            # Force parity to be odd to indicate broken state
            new_atom.parity[0] = 1 - new_atom.parity[0]
        
        return new_atom
    
    @receipted("R_theta")
    def R_theta(self, atom: CQEAtom, k: int = 1) -> CQEAtom:
        """Coxeter rotation - cyclic rotate lanes by k."""
        return self._apply(atom, "R_theta", k=k)
    
    @receipted("Weyl_reflect")
    def Weyl_reflect(self, atom: CQEAtom, idx: int = 0) -> CQEAtom:
        """Weyl reflection - reflect sign on one lane."""
        return self._apply(atom, "Weyl_reflect", idx=idx)
    
    @receipted("Midpoint")
    def Midpoint(self, atom: CQEAtom) -> CQEAtom:
        """Midpoint smoothing - pull lanes toward neighbor average."""
        return self._apply(atom, "Midpoint")
    
    @receipted("ECC_parity")
    def ECC_parity(self, atom: CQEAtom) -> CQEAtom:
        """ECC parity repair - quantize and fix parity."""
        return self._apply(atom, "ECC_parity")
    
    @receipted("SingleInsert")
    def SingleInsert(self, atom: CQEAtom, idx: int = 0, val: float = 0.0) -> CQEAtom:
        """Single insert - small perturbation at index."""
        return self._apply(atom, "SingleInsert", idx=idx, val=val)
    
    @receipted("ParityMirror")
    def ParityMirror(self, atom: CQEAtom, strength: float = 0.2) -> CQEAtom:
        """Parity mirror - mirror lanes across midpoint."""
        return self._apply(atom, "ParityMirror", strength=strength)
    
    def list_operators(self) -> List[str]:
        """List all available operators."""
        return self.base.shortlist()
    
    def apply_by_name(self, atom: CQEAtom, op_name: str, **kwargs) -> CQEAtom:
        """Apply an operator by name."""
        if op_name == "R_theta":
            return self.R_theta(atom, **kwargs)
        elif op_name == "Weyl_reflect":
            return self.Weyl_reflect(atom, **kwargs)
        elif op_name == "Midpoint":
            return self.Midpoint(atom)
        elif op_name == "ECC_parity":
            return self.ECC_parity(atom)
        elif op_name == "SingleInsert":
            return self.SingleInsert(atom, **kwargs)
        elif op_name == "ParityMirror":
            return self.ParityMirror(atom, **kwargs)
        else:
            raise ValueError(f"Unknown operator: {op_name}")


# Global instance
_operators: Optional[ALENAOperators] = None

def get_operators() -> ALENAOperators:
    """Get the global ALENA operators instance."""
    global _operators
    if _operators is None:
        _operators = ALENAOperators()
    return _operators
