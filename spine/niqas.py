"""
NIQAS - Non-Integer Quadratic Algebra System

Handles quadratic form operations and algebra for CQE atoms.
Provides the mathematical foundation for geometric operations.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from spine.kernel import CQEAtom
from spine.speedlight import get_speedlight, receipted


@dataclass
class QuadraticForm:
    """A quadratic form Q(x) = x^T A x."""
    matrix: np.ndarray
    signature: Tuple[int, int, int]  # (positive, negative, zero)
    determinant: float
    trace: float


@dataclass
class NIQASResult:
    """Result of NIQAS analysis."""
    form: QuadraticForm
    discriminant: float
    is_definite: bool
    is_positive_definite: bool
    eigenvalues: List[float]
    condition_number: float


class NIQASEngine:
    """
    NIQAS Engine - Quadratic Algebra Operations.
    
    Provides:
    - Quadratic form construction from atoms
    - Signature analysis
    - Definiteness checking
    - Eigenvalue decomposition
    """
    
    def __init__(self):
        self.speedlight = get_speedlight()
    
    @receipted("niqas_analyze")
    def analyze(self, atom: CQEAtom) -> NIQASResult:
        """Perform full NIQAS analysis on an atom."""
        lanes = atom.lanes
        
        # Construct quadratic form matrix from outer product
        A = np.outer(lanes, lanes)
        
        # Add diagonal regularization for stability
        A = A + np.eye(len(lanes)) * 0.01
        
        # Symmetrize
        A = (A + A.T) / 2
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(A)
        
        # Signature
        pos = int(np.sum(eigenvalues > 1e-10))
        neg = int(np.sum(eigenvalues < -1e-10))
        zero = len(eigenvalues) - pos - neg
        
        # Determinant and trace
        det = float(np.linalg.det(A))
        trace = float(np.trace(A))
        
        # Condition number
        cond = float(np.max(np.abs(eigenvalues)) / (np.min(np.abs(eigenvalues)) + 1e-10))
        
        form = QuadraticForm(
            matrix=A,
            signature=(pos, neg, zero),
            determinant=det,
            trace=trace
        )
        
        return NIQASResult(
            form=form,
            discriminant=det,
            is_definite=(neg == 0 or pos == 0),
            is_positive_definite=(neg == 0 and zero == 0),
            eigenvalues=eigenvalues.tolist(),
            condition_number=cond
        )
    
    @receipted("niqas_transform")
    def transform(self, atom: CQEAtom, matrix: np.ndarray) -> CQEAtom:
        """Apply a linear transformation to an atom."""
        new_lanes = matrix @ atom.lanes
        
        return CQEAtom(
            lanes=new_lanes,
            parity=np.array([int(x >= 0) for x in new_lanes]),
            metadata={**atom.metadata, "niqas_transform": True},
            provenance=f"{atom.provenance}>niqas_transform"
        )
    
    def gram_schmidt(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
        """Orthogonalize a set of vectors."""
        result = []
        for v in vectors:
            for u in result:
                v = v - np.dot(v, u) * u
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                result.append(v / norm)
        return result
    
    def project_to_subspace(self, atom: CQEAtom, basis: List[np.ndarray]) -> CQEAtom:
        """Project an atom onto a subspace defined by basis vectors."""
        # Orthonormalize basis
        ortho_basis = self.gram_schmidt(basis)
        
        # Project
        projection = np.zeros_like(atom.lanes)
        for b in ortho_basis:
            projection += np.dot(atom.lanes, b) * b
        
        return CQEAtom(
            lanes=projection,
            parity=np.array([int(x >= 0) for x in projection]),
            metadata={**atom.metadata, "niqas_projected": True},
            provenance=f"{atom.provenance}>niqas_project"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get NIQAS engine status."""
        return {
            "active": True,
            "operations": ["analyze", "transform", "gram_schmidt", "project_to_subspace"]
        }


# Global instance
_niqas: NIQASEngine = None

def get_niqas() -> NIQASEngine:
    """Get the global NIQAS engine."""
    global _niqas
    if _niqas is None:
        _niqas = NIQASEngine()
    return _niqas
