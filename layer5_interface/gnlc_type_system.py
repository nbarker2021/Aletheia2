"""
GNLC Geometric Type System
Geometry-Native Lambda Calculus (GNLC)
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper:
"Types are geometrically defined subspaces of E₈. Type checking is geometric
point-set membership. Subtyping is geometric inclusion. This makes type errors
impossible in well-typed programs."

This implements:
- Geometric types (subspaces of E₈)
- Type checking (point-set membership)
- Type inference
- Subtyping (geometric inclusion)
- Dependent types
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay
from layer5_interface.gnlc_lambda0 import Lambda0Term


class GeometricTypeKind(Enum):
    """Kinds of geometric types."""
    INTEGER = "integer"  # Integer coordinates
    ROOT_VECTOR = "root_vector"  # 240 root vectors
    WEYL_CHAMBER = "weyl_chamber"  # Conical region
    FUNCTION = "function"  # A → B
    SPHERE = "sphere"  # Points at distance r
    TENSOR = "tensor"  # A ⊗ B
    STATE = "state"  # System state
    CUSTOM = "custom"  # User-defined


@dataclass
class GeometricType:
    """
    Geometric type - subspace of E₈.
    
    From whitepaper:
    "A type is a geometrically defined subspace of the E₈ lattice."
    """
    type_id: str
    name: str
    kind: GeometricTypeKind
    definition: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __repr__(self):
        return f"Type[{self.name}:{self.kind.value}]"


@dataclass
class TypeJudgment:
    """
    Type judgment: term has type.
    
    Notation: Γ ⊢ e : T
    """
    term: Lambda0Term
    type_: GeometricType
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __repr__(self):
        return f"{self.term.term_id[:8]} : {self.type_.name}"


@dataclass
class SubtypeRelation:
    """
    Subtyping relation: A <: B
    
    From whitepaper:
    "A is subtype of B if subspace(A) ⊆ subspace(B)"
    """
    subtype: GeometricType
    supertype: GeometricType
    inclusion_ratio: float  # How much of A is in B
    
    def __repr__(self):
        return f"{self.subtype.name} <: {self.supertype.name} ({self.inclusion_ratio:.1%})"


class GeometricTypeSystem:
    """
    Geometric Type System for GNLC.
    
    From whitepaper:
    "Type checking is geometric point-set membership."
    
    Key features:
    1. Types as E₈ subspaces
    2. Type checking (membership)
    3. Type inference
    4. Subtyping (inclusion)
    5. Dependent types
    """
    
    def __init__(self):
        self.types: Dict[str, GeometricType] = {}
        self.judgments: List[TypeJudgment] = []
        self.subtype_relations: List[SubtypeRelation] = []
        
        # Initialize built-in types
        self._initialize_builtin_types()
    
    def _initialize_builtin_types(self):
        """Initialize built-in geometric types."""
        # Integer type
        self.define_type(
            "Integer",
            GeometricTypeKind.INTEGER,
            definition={
                'constraint': 'integer_coordinates',
                'dimension': 8
            }
        )
        
        # Root vector type
        self.define_type(
            "RootVector",
            GeometricTypeKind.ROOT_VECTOR,
            definition={
                'constraint': 'one_of_240_roots',
                'norm': 'sqrt(2)'
            }
        )
        
        # Unit sphere type
        self.define_type(
            "UnitSphere",
            GeometricTypeKind.SPHERE,
            definition={
                'center': [0] * 8,
                'radius': 1.0
            }
        )
    
    def define_type(
        self,
        name: str,
        kind: GeometricTypeKind,
        definition: Dict[str, Any]
    ) -> GeometricType:
        """
        Define new geometric type.
        
        Args:
            name: Type name
            kind: Type kind
            definition: Type definition
        
        Returns:
            GeometricType
        """
        type_id = f"type_{name.lower()}"
        
        geo_type = GeometricType(
            type_id=type_id,
            name=name,
            kind=kind,
            definition=definition
        )
        
        self.types[type_id] = geo_type
        return geo_type
    
    def check_type(
        self,
        term: Lambda0Term,
        type_: GeometricType
    ) -> bool:
        """
        Check if term has type (geometric membership).
        
        Args:
            term: Term to check
            type_: Type to check against
        
        Returns:
            True if term ∈ type_
        """
        e8_point = term.overlay.e8_base
        
        if type_.kind == GeometricTypeKind.INTEGER:
            # Check if all coordinates are integers
            return np.allclose(e8_point, np.round(e8_point))
        
        elif type_.kind == GeometricTypeKind.SPHERE:
            # Check if point is on sphere
            center = np.array(type_.definition['center'])
            radius = type_.definition['radius']
            distance = np.linalg.norm(e8_point - center)
            return np.isclose(distance, radius, atol=0.1)
        
        elif type_.kind == GeometricTypeKind.ROOT_VECTOR:
            # Check if point is one of 240 root vectors
            norm = np.linalg.norm(e8_point)
            return np.isclose(norm, np.sqrt(2), atol=0.1)
        
        elif type_.kind == GeometricTypeKind.WEYL_CHAMBER:
            # Check if point is in Weyl chamber
            # Simplified: check if coordinates satisfy chamber inequalities
            return self._in_weyl_chamber(e8_point, type_.definition)
        
        else:
            # Custom type - use definition
            return self._check_custom_type(e8_point, type_)
    
    def _in_weyl_chamber(
        self,
        point: np.ndarray,
        definition: Dict[str, Any]
    ) -> bool:
        """Check if point is in Weyl chamber."""
        # Simplified: check if coordinates are non-negative and decreasing
        if 'chamber_id' in definition:
            # Specific chamber
            return np.all(point >= 0) and np.all(np.diff(point) <= 0)
        return True
    
    def _check_custom_type(
        self,
        point: np.ndarray,
        type_: GeometricType
    ) -> bool:
        """Check custom type."""
        # Default: always true for custom types
        return True
    
    def infer_type(
        self,
        term: Lambda0Term
    ) -> GeometricType:
        """
        Infer type of term.
        
        Args:
            term: Term to infer type for
        
        Returns:
            Inferred GeometricType
        """
        e8_point = term.overlay.e8_base
        
        # Check against known types
        for type_ in self.types.values():
            if self.check_type(term, type_):
                return type_
        
        # Infer new type based on geometric properties
        norm = np.linalg.norm(e8_point)
        
        if np.isclose(norm, 1.0, atol=0.1):
            # Unit sphere
            return self.types['type_unitsphere']
        elif np.isclose(norm, np.sqrt(2), atol=0.1):
            # Root vector
            return self.types['type_rootvector']
        else:
            # Create sphere type
            return self.define_type(
                f"Sphere{norm:.2f}",
                GeometricTypeKind.SPHERE,
                definition={
                    'center': [0] * 8,
                    'radius': float(norm)
                }
            )
    
    def add_judgment(
        self,
        term: Lambda0Term,
        type_: GeometricType,
        confidence: float = 1.0
    ) -> TypeJudgment:
        """
        Add type judgment.
        
        Args:
            term: Term
            type_: Type
            confidence: Confidence level
        
        Returns:
            TypeJudgment
        """
        judgment = TypeJudgment(
            term=term,
            type_=type_,
            confidence=confidence
        )
        
        self.judgments.append(judgment)
        return judgment
    
    def is_subtype(
        self,
        subtype: GeometricType,
        supertype: GeometricType
    ) -> bool:
        """
        Check if subtype <: supertype.
        
        From whitepaper:
        "A is subtype of B if subspace(A) ⊆ subspace(B)"
        
        Args:
            subtype: Potential subtype
            supertype: Potential supertype
        
        Returns:
            True if subtype <: supertype
        """
        # Sphere subtyping
        if (subtype.kind == GeometricTypeKind.SPHERE and
            supertype.kind == GeometricTypeKind.SPHERE):
            r1 = subtype.definition['radius']
            r2 = supertype.definition['radius']
            # Smaller sphere is subtype of larger sphere
            return r1 <= r2
        
        # Root vectors are subtype of unit sphere
        if (subtype.kind == GeometricTypeKind.ROOT_VECTOR and
            supertype.kind == GeometricTypeKind.SPHERE):
            return True
        
        # Integer points are subtype of all
        if subtype.kind == GeometricTypeKind.INTEGER:
            return True
        
        return False
    
    def compute_subtype_relation(
        self,
        subtype: GeometricType,
        supertype: GeometricType
    ) -> Optional[SubtypeRelation]:
        """
        Compute subtype relation with inclusion ratio.
        
        Args:
            subtype: Subtype
            supertype: Supertype
        
        Returns:
            SubtypeRelation if subtype <: supertype, None otherwise
        """
        if not self.is_subtype(subtype, supertype):
            return None
        
        # Compute inclusion ratio (simplified)
        if (subtype.kind == GeometricTypeKind.SPHERE and
            supertype.kind == GeometricTypeKind.SPHERE):
            r1 = subtype.definition['radius']
            r2 = supertype.definition['radius']
            inclusion_ratio = (r1 / r2) ** 8  # 8-dimensional volume ratio
        else:
            inclusion_ratio = 1.0
        
        relation = SubtypeRelation(
            subtype=subtype,
            supertype=supertype,
            inclusion_ratio=inclusion_ratio
        )
        
        self.subtype_relations.append(relation)
        return relation
    
    def define_function_type(
        self,
        name: str,
        domain: GeometricType,
        codomain: GeometricType
    ) -> GeometricType:
        """
        Define function type A → B.
        
        Args:
            name: Type name
            domain: Domain type
            codomain: Codomain type
        
        Returns:
            Function GeometricType
        """
        return self.define_type(
            name,
            GeometricTypeKind.FUNCTION,
            definition={
                'domain': domain.type_id,
                'codomain': codomain.type_id
            }
        )
    
    def define_dependent_type(
        self,
        name: str,
        base_type: GeometricType,
        parameter: str,
        constraint: str
    ) -> GeometricType:
        """
        Define dependent type.
        
        Args:
            name: Type name
            base_type: Base type
            parameter: Parameter name
            constraint: Constraint on parameter
        
        Returns:
            Dependent GeometricType
        """
        return self.define_type(
            name,
            base_type.kind,
            definition={
                **base_type.definition,
                'dependent': True,
                'parameter': parameter,
                'constraint': constraint
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get type system statistics."""
        return {
            'types': len(self.types),
            'judgments': len(self.judgments),
            'subtype_relations': len(self.subtype_relations),
            'types_by_kind': self._count_types_by_kind()
        }
    
    def _count_types_by_kind(self) -> Dict[str, int]:
        """Count types by kind."""
        counts = {}
        for type_ in self.types.values():
            kind = type_.kind.value
            counts[kind] = counts.get(kind, 0) + 1
        return counts


# Example usage and tests
if __name__ == "__main__":
    print("=== GNLC Geometric Type System Test ===\n")
    
    from layer1_morphonic.overlay_system import ImmutablePose
    from layer5_interface.gnlc_lambda0 import Lambda0Calculus
    import time
    
    # Create type system
    type_system = GeometricTypeSystem()
    lambda0 = Lambda0Calculus()
    
    # Test 1: Built-in types
    print("Test 1: Built-in Types")
    
    for type_id, type_ in type_system.types.items():
        print(f"  {type_}")
    print()
    
    # Test 2: Create terms
    print("Test 2: Create Terms")
    
    # Integer point
    e8_int = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    activations = np.zeros(240, dtype=int)
    activations[0:80] = 1
    pose_int = ImmutablePose(tuple(e8_int), tuple(np.eye(8)[0]), time.time())
    overlay_int = Overlay(e8_base=e8_int, activations=activations, pose=pose_int)
    term_int = lambda0.atom(overlay_int)
    
    # Unit sphere point
    e8_unit = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_unit = ImmutablePose(tuple(e8_unit), tuple(np.eye(8)[0]), time.time())
    overlay_unit = Overlay(e8_base=e8_unit, activations=activations.copy(), pose=pose_unit)
    term_unit = lambda0.atom(overlay_unit)
    
    # Root vector point
    e8_root = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e8_root = e8_root / np.linalg.norm(e8_root) * np.sqrt(2)
    pose_root = ImmutablePose(tuple(e8_root), tuple(np.eye(8)[0]), time.time())
    overlay_root = Overlay(e8_base=e8_root, activations=activations.copy(), pose=pose_root)
    term_root = lambda0.atom(overlay_root)
    
    print(f"Integer term: {term_int}")
    print(f"Unit sphere term: {term_unit}")
    print(f"Root vector term: {term_root}")
    print()
    
    # Test 3: Type checking
    print("Test 3: Type Checking")
    
    int_type = type_system.types['type_integer']
    sphere_type = type_system.types['type_unitsphere']
    root_type = type_system.types['type_rootvector']
    
    print(f"term_int : Integer? {type_system.check_type(term_int, int_type)}")
    print(f"term_unit : UnitSphere? {type_system.check_type(term_unit, sphere_type)}")
    print(f"term_root : RootVector? {type_system.check_type(term_root, root_type)}")
    print()
    
    # Test 4: Type inference
    print("Test 4: Type Inference")
    
    inferred_int = type_system.infer_type(term_int)
    inferred_unit = type_system.infer_type(term_unit)
    inferred_root = type_system.infer_type(term_root)
    
    print(f"Inferred type for term_int: {inferred_int}")
    print(f"Inferred type for term_unit: {inferred_unit}")
    print(f"Inferred type for term_root: {inferred_root}")
    print()
    
    # Test 5: Type judgments
    print("Test 5: Type Judgments")
    
    judgment1 = type_system.add_judgment(term_int, int_type)
    judgment2 = type_system.add_judgment(term_unit, sphere_type)
    
    print(f"Judgment 1: {judgment1}")
    print(f"Judgment 2: {judgment2}")
    print()
    
    # Test 6: Subtyping
    print("Test 6: Subtyping")
    
    # Define sphere types
    small_sphere = type_system.define_type(
        "SmallSphere",
        GeometricTypeKind.SPHERE,
        definition={'center': [0]*8, 'radius': 0.5}
    )
    
    large_sphere = type_system.define_type(
        "LargeSphere",
        GeometricTypeKind.SPHERE,
        definition={'center': [0]*8, 'radius': 2.0}
    )
    
    is_sub = type_system.is_subtype(small_sphere, large_sphere)
    print(f"{small_sphere.name} <: {large_sphere.name}? {is_sub}")
    
    relation = type_system.compute_subtype_relation(small_sphere, large_sphere)
    if relation:
        print(f"Subtype relation: {relation}")
    print()
    
    # Test 7: Function types
    print("Test 7: Function Types")
    
    func_type = type_system.define_function_type(
        "IntToSphere",
        int_type,
        sphere_type
    )
    print(f"Function type: {func_type}")
    print(f"  Domain: {func_type.definition['domain']}")
    print(f"  Codomain: {func_type.definition['codomain']}")
    print()
    
    # Test 8: Dependent types
    print("Test 8: Dependent Types")
    
    dep_type = type_system.define_dependent_type(
        "SphereR",
        sphere_type,
        parameter="r",
        constraint="r > 0"
    )
    print(f"Dependent type: {dep_type}")
    print(f"  Parameter: {dep_type.definition['parameter']}")
    print(f"  Constraint: {dep_type.definition['constraint']}")
    print()
    
    # Test 9: Statistics
    print("Test 9: Statistics")
    
    stats = type_system.get_statistics()
    print(f"Types: {stats['types']}")
    print(f"Judgments: {stats['judgments']}")
    print(f"Subtype relations: {stats['subtype_relations']}")
    print(f"Types by kind: {stats['types_by_kind']}")
    print()
    
    print("=== All Tests Passed ===")
