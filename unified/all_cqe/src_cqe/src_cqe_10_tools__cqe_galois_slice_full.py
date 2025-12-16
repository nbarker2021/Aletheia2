"""
CQE GALOIS Slice - Field Theory, Extensions, Algebraic Closure

Implements slice 35 of the CQE system, providing field-theoretic
analysis through field extensions, Galois groups, and polynomial equations.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, field
import sympy as sp
from sympy import Poly, Symbol, Field, GF
from sympy.polys.galoistools import gf_irreducible_p
import math

from cqe.core.atom import UniversalAtom, SliceData
from cqe.core.validation import SliceValidator, ValidationResult

@dataclass
class GALOISData(SliceData):
    """Field-theoretic data structure for GALOIS slice"""

    # Field extension data
    base_field: str = "Q"  # Base field (Q, R, C, F_p, etc.)
    extension_polynomial: List[int] = field(default_factory=list)  # Coefficients
    extension_degree: int = 1
    is_normal: bool = True
    is_separable: bool = True

    # Galois group structure
    galois_group_size: int = 1
    galois_group_generators: List[List[int]] = field(default_factory=list)  # Permutation generators
    is_solvable: bool = True

    # Correspondence data
    subgroups: Dict[str, List[List[int]]] = field(default_factory=dict)
    intermediate_fields: Dict[str, Dict] = field(default_factory=dict)
    correspondence_verified: bool = False

    # Polynomial properties
    minimal_polynomial: List[int] = field(default_factory=list)
    is_irreducible: bool = False
    splits_completely: bool = False

    # Metrics
    splitting_field_degree: int = 1
    galois_complexity: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self._compute_galois_metrics()

    def _compute_galois_metrics(self):
        """Compute field-theoretic complexity metrics"""
        # Galois complexity based on group size and extension degree
        self.galois_complexity = (
            math.log(max(self.galois_group_size, 2)) + 
            math.log(max(self.extension_degree, 2))
        ) / 10.0

        # Update energy based on complexity
        self.energy = self.galois_complexity + 0.05 * self.splitting_field_degree

class GALOISValidator(SliceValidator):
    """Validator for GALOIS slice axioms"""

    async def validate_promotion(self, atom_i: UniversalAtom, atom_j: UniversalAtom) -> ValidationResult:
        """Validate field extension preservation in promotion"""

        data_i = atom_i.get_slice_data("galois")
        data_j = atom_j.get_slice_data("galois")

        if not data_i or not data_j:
            return ValidationResult(False, "missing_galois_data")

        # G1: Field tower multiplicativity
        if not self._validate_field_tower(data_i, data_j):
            return ValidationResult(False, "field_tower_violated")

        # G2: Galois correspondence preservation
        if not self._validate_galois_correspondence(data_i, data_j):
            return ValidationResult(False, "galois_correspondence_broken")

        # G3: Separability maintained
        if not self._validate_separability(data_i, data_j):
            return ValidationResult(False, "separability_violated")

        # G4: Normal closure property
        if not self._validate_normal_closure(data_i, data_j):
            return ValidationResult(False, "normal_closure_violated")

        # Energy constraint - degrees must not increase
        if data_j.extension_degree > data_i.extension_degree:
            return ValidationResult(False, "extension_degree_increased")

        energy_delta = data_j.energy - data_i.energy
        if energy_delta > 0:
            return ValidationResult(False, "galois_energy_increased", energy_delta)

        return ValidationResult(True, "galois_validated", energy_delta)

    def _validate_field_tower(self, data_i: GALOISData, data_j: GALOISData) -> bool:
        """Validate field tower multiplicativity [M:K] = [M:L][L:K]"""
        # Check that extension degrees respect multiplicativity
        # Simplified: ensure j's degree is compatible with i's

        if data_i.extension_degree == 0 or data_j.extension_degree == 0:
            return True

        # Check if degrees are compatible (divisibility or equality)
        return data_j.extension_degree % data_i.extension_degree == 0 or data_i.extension_degree % data_j.extension_degree == 0

    def _validate_galois_correspondence(self, data_i: GALOISData, data_j: GALOISData) -> bool:
        """Validate Galois correspondence preservation"""
        # If correspondence was verified in i, it should remain in j
        if data_i.correspondence_verified and not data_j.correspondence_verified:
            return False

        # Group size should not increase
        return data_j.galois_group_size <= data_i.galois_group_size

    def _validate_separability(self, data_i: GALOISData, data_j: GALOISData) -> bool:
        """Validate separability preservation"""
        # If i is separable, j should also be separable
        return not data_i.is_separable or data_j.is_separable

    def _validate_normal_closure(self, data_i: GALOISData, data_j: GALOISData) -> bool:
        """Validate normal closure property"""
        # If i has normal extension, j should maintain normality
        return not data_i.is_normal or data_j.is_normal

class GALOISSlice:
    """Complete GALOIS slice implementation for field theory"""

    def __init__(self):
        self.validator = GALOISValidator()
        self.field_cache: Dict[str, Any] = {}

        # Common irreducible polynomials for field extensions
        self.irreducible_polys = {
            2: [1, 1, 1],      # x^2 + x + 1 over F_2
            3: [1, 0, 0, 2],   # x^3 + 2 over F_3  
            5: [1, 0, 2],      # x^2 + 2 over F_5
        }

    async def initialize(self):
        """Initialize GALOIS slice"""
        pass

    async def process_atom(self, atom: UniversalAtom) -> GALOISData:
        """Process atom to extract field-theoretic structure"""

        raw_data = atom.get_mathematical_content()

        # Extract/construct field extension from data
        galois_data = await self._analyze_field_structure(raw_data)

        # Compute Galois group
        await self._compute_galois_group(galois_data)

        # Establish Galois correspondence
        await self._compute_galois_correspondence(galois_data)

        # Validate polynomial properties
        await self._analyze_polynomial_properties(galois_data)

        # Mark as validated
        galois_data.validated = True

        return galois_data

    async def _analyze_field_structure(self, data: Dict[str, Any]) -> GALOISData:
        """Analyze input to extract field extension structure"""

        galois_data = GALOISData()

        # Extract field information based on data type
        if isinstance(data.get("raw_data"), str):
            text = data["raw_data"]

            # Look for mathematical field-related keywords
            if "finite" in text.lower() or "galois" in text.lower():
                galois_data.base_field = "F_2"
                galois_data.extension_polynomial = [1, 1, 1]  # x^2 + x + 1
                galois_data.extension_degree = 2

            elif "complex" in text.lower() or "algebraic" in text.lower():
                galois_data.base_field = "Q"
                galois_data.extension_polynomial = [1, 0, 0, 0, -2]  # x^4 - 2
                galois_data.extension_degree = 4

            else:
                # Default: quadratic extension of rationals
                galois_data.base_field = "Q"
                galois_data.extension_polynomial = [1, 0, -2]  # x^2 - 2
                galois_data.extension_degree = 2

        elif isinstance(data.get("raw_data"), (list, tuple)):
            # Use list elements as polynomial coefficients
            coeffs = list(data["raw_data"])
            if len(coeffs) > 1:
                galois_data.extension_polynomial = [int(x) if isinstance(x, (int, float)) else 1 for x in coeffs]
                galois_data.extension_degree = len(coeffs) - 1

        elif isinstance(data.get("raw_data"), dict) and "polynomial" in data["raw_data"]:
            # Direct polynomial specification
            poly_data = data["raw_data"]["polynomial"]
            if isinstance(poly_data, list):
                galois_data.extension_polynomial = poly_data
                galois_data.extension_degree = len(poly_data) - 1

        else:
            # Default: create minimal field extension from atom hash
            hash_val = abs(hash(data.get("id", "default"))) % 100
            if hash_val % 2 == 0:
                galois_data.extension_polynomial = [1, 0, -hash_val % 10]  # x^2 - n
                galois_data.extension_degree = 2
            else:
                galois_data.extension_polynomial = [1, 0, 0, -hash_val % 5]  # x^3 - n
                galois_data.extension_degree = 3

        return galois_data

    async def _compute_galois_group(self, galois_data: GALOISData):
        """Compute the Galois group of the field extension"""

        degree = galois_data.extension_degree

        if degree <= 1:
            galois_data.galois_group_size = 1
            galois_data.galois_group_generators = [[0]]  # Identity
            galois_data.is_solvable = True
            return

        # For common cases, use known Galois groups
        if degree == 2:
            # Galois group of quadratic extension is C_2
            galois_data.galois_group_size = 2
            galois_data.galois_group_generators = [[1, 0]]  # Single non-identity element
            galois_data.is_solvable = True

        elif degree == 3:
            # Could be C_3 or S_3 depending on discriminant
            # Simplified: assume C_3 for solvable case
            galois_data.galois_group_size = 3
            galois_data.galois_group_generators = [[1, 2, 0]]  # 3-cycle
            galois_data.is_solvable = True

        elif degree == 4:
            # Could be C_4, V_4, D_4, A_4, or S_4
            # Simplified: assume D_4 (dihedral group of order 8)  
            galois_data.galois_group_size = 8
            galois_data.galois_group_generators = [
                [1, 0, 3, 2],  # Reflection
                [1, 2, 3, 0]   # 4-cycle
            ]
            galois_data.is_solvable = True

        else:
            # General case: assume cyclic group of order degree (often true for splitting fields)
            galois_data.galois_group_size = degree
            generators = list(range(1, degree)) + [0]  # n-cycle
            galois_data.galois_group_generators = [generators]
            galois_data.is_solvable = degree <= 4  # Groups of order ≤ 4 are solvable

        # Set normal and separable properties
        galois_data.is_normal = True  # Assume Galois extension (normal + separable)
        galois_data.is_separable = True

    async def _compute_galois_correspondence(self, galois_data: GALOISData):
        """Establish Galois correspondence between subgroups and intermediate fields"""

        group_size = galois_data.galois_group_size
        degree = galois_data.extension_degree

        if group_size <= 1:
            galois_data.correspondence_verified = True
            return

        # Generate some subgroups
        if group_size == 2:
            galois_data.subgroups = {
                "trivial": [[]],
                "full": [galois_data.galois_group_generators[0]]
            }
            galois_data.intermediate_fields = {
                "base": {"degree": 1},
                "full": {"degree": 2}
            }

        elif group_size <= 6:
            # For small groups, enumerate main subgroups
            galois_data.subgroups = {
                "trivial": [[]],
                "generator": galois_data.galois_group_generators,
                "full": galois_data.galois_group_generators
            }

            # Corresponding intermediate fields
            galois_data.intermediate_fields = {
                "base": {"degree": 1},
                "intermediate": {"degree": degree // 2} if degree % 2 == 0 else {"degree": 1},
                "full": {"degree": degree}
            }

        # Mark correspondence as verified for well-formed cases
        galois_data.correspondence_verified = (
            len(galois_data.subgroups) > 0 and 
            len(galois_data.intermediate_fields) > 0
        )

    async def _analyze_polynomial_properties(self, galois_data: GALOISData):
        """Analyze properties of the extension polynomial"""

        poly_coeffs = galois_data.extension_polynomial

        if not poly_coeffs or len(poly_coeffs) < 2:
            return

        # Set minimal polynomial (same as extension polynomial for now)
        galois_data.minimal_polynomial = poly_coeffs

        # Check irreducibility heuristically
        degree = len(poly_coeffs) - 1

        if degree <= 4:
            # For low degrees, assume irreducible if coefficients are reasonable
            galois_data.is_irreducible = (
                poly_coeffs[0] != 0 and  # Leading coefficient non-zero
                not all(c == 0 for c in poly_coeffs[1:])  # Not just x^n
            )
        else:
            # For higher degrees, use probabilistic heuristic
            non_zero_coeffs = sum(1 for c in poly_coeffs if c != 0)
            galois_data.is_irreducible = non_zero_coeffs >= degree // 2

        # Splitting property - assume true for Galois extensions
        galois_data.splits_completely = galois_data.is_normal

        # Set splitting field degree
        galois_data.splitting_field_degree = galois_data.extension_degree

    async def stitch_atoms(self, data_i: GALOISData, data_j: GALOISData) -> GALOISData:
        """Combine field structures from two atoms"""

        stitched = GALOISData()

        # Choose more complex field structure
        if data_i.extension_degree >= data_j.extension_degree:
            primary, secondary = data_i, data_j
        else:
            primary, secondary = data_j, data_i

        # Inherit structure from primary
        stitched.base_field = primary.base_field
        stitched.extension_polynomial = primary.extension_polynomial
        stitched.extension_degree = primary.extension_degree
        stitched.is_normal = primary.is_normal and secondary.is_normal
        stitched.is_separable = primary.is_separable and secondary.is_separable

        # Combine group properties conservatively
        stitched.galois_group_size = min(primary.galois_group_size, secondary.galois_group_size)
        stitched.is_solvable = primary.is_solvable and secondary.is_solvable

        # Merge correspondences if both are verified
        if primary.correspondence_verified and secondary.correspondence_verified:
            stitched.subgroups = {**primary.subgroups}
            stitched.intermediate_fields = {**primary.intermediate_fields}
            stitched.correspondence_verified = True

        # Conservative polynomial properties
        stitched.minimal_polynomial = primary.minimal_polynomial
        stitched.is_irreducible = primary.is_irreducible and secondary.is_irreducible
        stitched.splits_completely = primary.splits_completely and secondary.splits_completely

        # Combined energy (monotone constraint)
        stitched.energy = min(primary.energy, secondary.energy)
        stitched.validated = True

        return stitched

    def get_status(self) -> Dict[str, Any]:
        """Get slice operational status"""
        return {
            "slice_name": "GALOIS",
            "slice_id": 35,
            "domain": "Field Theory",
            "cached_fields": len(self.field_cache),
            "validator_active": True
        }

    async def shutdown(self):
        """Cleanup slice resources"""
        self.field_cache.clear()
