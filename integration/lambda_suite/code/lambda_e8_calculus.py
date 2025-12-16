"""
Extended Lambda Calculus (Λ⊗E₈)
================================

Lambda calculus extended to capture geometric transforms in E₈ space.
Integrates with:
- Geometric Transformer (captures transform operations as lambda)
- Token Object System (lambda IR in tokens)
- AGRM/MDHG (path operations as lambda composition)

Key features:
- Geometric operations as lambda terms
- E₈ lattice navigation as lambda composition
- Dihedral operations as lambda transformations
- Automatic derivation from system operations
- Type system for geometric constraints
"""
from pathlib import Path


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

# ============================================================================
# LAMBDA TERM TYPES
# ============================================================================

class LambdaType(Enum):
    """Types in the extended lambda calculus."""
    SCALAR = "scalar"           # Real number
    VECTOR = "vector"           # E₈ vector
    LATTICE = "lattice"         # E₈ lattice point
    TRANSFORM = "transform"     # Geometric transform
    PATH = "path"               # AGRM path
    TOKEN = "token"             # Token Object
    DIHEDRAL = "dihedral"       # Dihedral group element

@dataclass
class LambdaTerm:
    """
    A term in the extended lambda calculus.
    
    Grammar:
        t ::= x                     (variable)
            | λ x: τ. t            (abstraction)
            | t t                   (application)
            | (e8_embed t)          (E₈ embedding)
            | (e8_project t d)      (E₈ projection to dimension d)
            | (e8_navigate t w)     (Navigate E₈ via Weyl chamber w)
            | (dihedral_op N k t)   (Dihedral operation)
            | (path_compose t₁ t₂)  (AGRM path composition)
            | (conserve t)          (Apply conservation law)
    """
    term_type: str  # "var", "abs", "app", "e8_op", "dihedral_op", "path_op"
    content: Any    # Depends on term_type
    lambda_type: Optional[LambdaType] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_string(self) -> str:
        """Convert lambda term to string representation."""
        if self.term_type == "var":
            return self.content
        
        elif self.term_type == "abs":
            var, body = self.content
            type_annotation = f": {self.lambda_type.value}" if self.lambda_type else ""
            return f"(λ {var}{type_annotation}. {body.to_string()})"
        
        elif self.term_type == "app":
            func, arg = self.content
            return f"({func.to_string()} {arg.to_string()})"
        
        elif self.term_type == "e8_op":
            op_name, args = self.content
            arg_strs = [a.to_string() if isinstance(a, LambdaTerm) else str(a) for a in args]
            return f"({op_name} {' '.join(arg_strs)})"
        
        elif self.term_type == "dihedral_op":
            N, k, reflect, arg = self.content
            return f"(D_{N}^{k}{'*' if reflect else ''} {arg.to_string()})"
        
        elif self.term_type == "path_op":
            op_name, paths = self.content
            path_strs = [p.to_string() if isinstance(p, LambdaTerm) else str(p) for p in paths]
            return f"({op_name} {' '.join(path_strs)})"
        
        else:
            return f"<{self.term_type}>"

# ============================================================================
# LAMBDA CALCULUS BUILDER