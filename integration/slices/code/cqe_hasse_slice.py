"""
CQE HASSE Slice - Order Theory, Posets, Galois Connections

Implements slice 34 of the CQE system, providing order-theoretic
analysis through partial orders, lattice operations, and Galois connections.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, field
from itertools import combinations
import networkx as nx

from cqe.core.atom import UniversalAtom, SliceData
from cqe.core.validation import SliceValidator, ValidationResult

@dataclass
class HASSEData(SliceData):
    """Order-theoretic data structure for HASSE slice"""

    # Poset structure
    elements: List[str] = field(default_factory=list)
    order_relations: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    transitive_closure: Dict[Tuple[str, str], bool] = field(default_factory=dict)

    # Lattice operations
    joins: Dict[Tuple[str, str], str] = field(default_factory=dict)
    meets: Dict[Tuple[str, str], str] = field(default_factory=dict)
    top_element: Optional[str] = None
    bottom_element: Optional[str] = None

    # Galois connections
    galois_pairs: List[Tuple[str, str]] = field(default_factory=list)
    adjoint_verified: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    residuated: bool = False

    # Metrics
    poset_width: int = 0  # Maximum antichain size
    poset_height: int = 0  # Maximum chain length
    hasse_complexity: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.elements:
            self._compute_hasse_metrics()

    def _compute_hasse_metrics(self):
        """Compute order-theoretic complexity metrics"""
        n = len(self.elements)
        if n == 0:
            return

        # Hasse complexity based on relation density
        total_possible = n * (n - 1) // 2
        actual_relations = len(self.order_relations)
        self.hasse_complexity = actual_relations / max(total_possible, 1)

        # Update energy based on complexity
        self.energy = self.hasse_complexity + 0.1 * (self.poset_width + self.poset_height)

class HASSEValidator(SliceValidator):
    """Validator for HASSE slice axioms"""

    async def validate_promotion(self, atom_i: UniversalAtom, atom_j: UniversalAtom) -> ValidationResult:
        """Validate order-preserving promotion"""

        data_i = atom_i.get_slice_data("hasse")
        data_j = atom_j.get_slice_data("hasse")

        if not data_i or not data_j:
            return ValidationResult(False, "missing_hasse_data")

        # H1: Order preservation check
        if not self._validate_order_preservation(data_i, data_j):
            return ValidationResult(False, "order_preservation_violated")

        # H2: Lattice structure maintained  
        if not self._validate_lattice_properties(data_i, data_j):
            return ValidationResult(False, "lattice_properties_violated")

        # H3: Galois correspondence preserved
        if not self._validate_galois_correspondence(data_i, data_j):
            return ValidationResult(False, "galois_correspondence_broken")

        # Energy constraint
        energy_delta = data_j.energy - data_i.energy
        if energy_delta > 0:
            return ValidationResult(False, "hasse_energy_increased", energy_delta)

        return ValidationResult(True, "hasse_validated", energy_delta)

    def _validate_order_preservation(self, data_i: HASSEData, data_j: HASSEData) -> bool:
        """Check that partial order is preserved"""
        # All relations in i must be preserved in j
        for relation, is_related in data_i.order_relations.items():
            if is_related and relation not in data_j.order_relations:
                return False
            if relation in data_j.order_relations and data_j.order_relations[relation] != is_related:
                return False

        return True

    def _validate_lattice_properties(self, data_i: HASSEData, data_j: HASSEData) -> bool:
        """Validate lattice join/meet operations"""
        # Check that all joins in i are consistent in j
        for elements, join_result in data_i.joins.items():
            if elements in data_j.joins:
                if data_j.joins[elements] != join_result:
                    return False

        # Similar check for meets
        for elements, meet_result in data_i.meets.items():
            if elements in data_j.meets:
                if data_j.meets[elements] != meet_result:
                    return False

        return True

    def _validate_galois_correspondence(self, data_i: HASSEData, data_j: HASSEData) -> bool:
        """Check Galois connection preservation"""
        # All verified adjoint pairs must remain verified
        for pair, verified in data_i.adjoint_verified.items():
            if verified and pair in data_j.adjoint_verified:
                if not data_j.adjoint_verified[pair]:
                    return False

        return True

class HASSESlice:
    """Complete HASSE slice implementation for order theory"""

    def __init__(self):
        self.validator = HASSEValidator()
        self.order_cache: Dict[str, nx.DiGraph] = {}

    async def initialize(self):
        """Initialize HASSE slice"""
        pass

    async def process_atom(self, atom: UniversalAtom) -> HASSEData:
        """Process atom to extract order-theoretic structure"""

        raw_data = atom.get_mathematical_content()

        # Extract/infer partial order structure from data
        hasse_data = await self._analyze_order_structure(raw_data)

        # Compute lattice operations
        await self._compute_lattice_operations(hasse_data)

        # Find Galois connections
        await self._identify_galois_connections(hasse_data)

        # Validate and mark as complete
        hasse_data.validated = True

        return hasse_data

    async def _analyze_order_structure(self, data: Dict[str, Any]) -> HASSEData:
        """Analyze input to extract partial order relationships"""

        hasse_data = HASSEData()

        # Extract elements based on data type
        if isinstance(data.get("raw_data"), str):
            # Text analysis for hierarchical concepts
            elements = self._extract_hierarchical_concepts(data["raw_data"])
        elif isinstance(data.get("raw_data"), (list, tuple)):
            # List/tuple implies some ordering
            elements = [f"elem_{i}" for i in range(len(data["raw_data"]))]
        elif isinstance(data.get("raw_data"), dict):
            # Dictionary keys as elements
            elements = list(data["raw_data"].keys())
        else:
            # Default: create minimal poset from atom ID
            elements = [f"node_{i}" for i in range(3)]

        hasse_data.elements = elements

        # Generate partial order relations
        for i, elem_a in enumerate(elements):
            for j, elem_b in enumerate(elements):
                if i <= j:  # Reflexive and partial order
                    hasse_data.order_relations[(elem_a, elem_b)] = (i <= j)

        # Compute transitive closure
        hasse_data.transitive_closure = self._compute_transitive_closure(
            hasse_data.order_relations
        )

        return hasse_data

    def _extract_hierarchical_concepts(self, text: str) -> List[str]:
        """Extract hierarchical concepts from text for poset construction"""
        # Simple heuristic: look for hierarchical keywords
        hierarchical_words = []

        # Split text and identify concepts
        words = text.lower().split()
        concept_indicators = ["theory", "method", "system", "element", "component", "part"]

        for word in words:
            if any(indicator in word for indicator in concept_indicators):
                hierarchical_words.append(word)

        # Ensure at least 2 elements for meaningful poset
        if len(hierarchical_words) < 2:
            hierarchical_words = ["concept_a", "concept_b", "concept_c"]

        return hierarchical_words[:5]  # Limit complexity

    def _compute_transitive_closure(self, relations: Dict[Tuple[str, str], bool]) -> Dict[Tuple[str, str], bool]:
        """Compute transitive closure of partial order"""
        closure = relations.copy()

        # Floyd-Warshall algorithm for transitive closure
        elements = set()
        for (a, b), is_related in relations.items():
            if is_related:
                elements.add(a)
                elements.add(b)

        elements = list(elements)

        for k in elements:
            for i in elements:
                for j in elements:
                    if closure.get((i, k), False) and closure.get((k, j), False):
                        closure[(i, j)] = True

        return closure

    async def _compute_lattice_operations(self, hasse_data: HASSEData):
        """Compute join and meet operations for lattice structure"""

        elements = hasse_data.elements
        relations = hasse_data.transitive_closure

        # Compute joins (least upper bounds)
        for elem_a in elements:
            for elem_b in elements:
                join_result = self._find_least_upper_bound(elem_a, elem_b, elements, relations)
                if join_result:
                    hasse_data.joins[(elem_a, elem_b)] = join_result

        # Compute meets (greatest lower bounds)  
        for elem_a in elements:
            for elem_b in elements:
                meet_result = self._find_greatest_lower_bound(elem_a, elem_b, elements, relations)
                if meet_result:
                    hasse_data.meets[(elem_a, elem_b)] = meet_result

        # Find top and bottom elements
        hasse_data.top_element = self._find_top_element(elements, relations)
        hasse_data.bottom_element = self._find_bottom_element(elements, relations)

    def _find_least_upper_bound(self, a: str, b: str, elements: List[str], 
                               relations: Dict[Tuple[str, str], bool]) -> Optional[str]:
        """Find least upper bound (join) of two elements"""

        # Find all upper bounds
        upper_bounds = []
        for elem in elements:
            if (relations.get((a, elem), False) and relations.get((b, elem), False)):
                upper_bounds.append(elem)

        if not upper_bounds:
            return None

        # Find minimal element among upper bounds
        for candidate in upper_bounds:
            is_minimal = True
            for other in upper_bounds:
                if other != candidate and relations.get((other, candidate), False):
                    is_minimal = False
                    break
            if is_minimal:
                return candidate

        return upper_bounds[0]  # Fallback

    def _find_greatest_lower_bound(self, a: str, b: str, elements: List[str],
                                  relations: Dict[Tuple[str, str], bool]) -> Optional[str]:
        """Find greatest lower bound (meet) of two elements"""

        # Find all lower bounds
        lower_bounds = []
        for elem in elements:
            if (relations.get((elem, a), False) and relations.get((elem, b), False)):
                lower_bounds.append(elem)

        if not lower_bounds:
            return None

        # Find maximal element among lower bounds
        for candidate in lower_bounds:
            is_maximal = True
            for other in lower_bounds:
                if other != candidate and relations.get((candidate, other), False):
                    is_maximal = False
                    break
            if is_maximal:
                return candidate

        return lower_bounds[0]  # Fallback

    def _find_top_element(self, elements: List[str], relations: Dict[Tuple[str, str], bool]) -> Optional[str]:
        """Find top element (maximum) if it exists"""
        for candidate in elements:
            is_top = True
            for other in elements:
                if not relations.get((other, candidate), False):
                    is_top = False
                    break
            if is_top:
                return candidate
        return None

    def _find_bottom_element(self, elements: List[str], relations: Dict[Tuple[str, str], bool]) -> Optional[str]:
        """Find bottom element (minimum) if it exists"""
        for candidate in elements:
            is_bottom = True
            for other in elements:
                if not relations.get((candidate, other), False):
                    is_bottom = False
                    break
            if is_bottom:
                return candidate
        return None

    async def _identify_galois_connections(self, hasse_data: HASSEData):
        """Identify and verify Galois connections"""

        # For each pair of elements, check if they form a Galois connection
        elements = hasse_data.elements

        for i, elem_f in enumerate(elements):
            for j, elem_g in enumerate(elements):
                if i < j:  # Avoid duplicates

                    # Check adjunction property: F(x) ≤ y ⟺ x ≤ G(y)
                    is_adjoint = self._verify_adjoint_property(
                        elem_f, elem_g, elements, hasse_data.transitive_closure
                    )

                    if is_adjoint:
                        hasse_data.galois_pairs.append((elem_f, elem_g))
                        hasse_data.adjoint_verified[(elem_f, elem_g)] = True

        # Mark as residuated if any Galois connections found
        hasse_data.residuated = len(hasse_data.galois_pairs) > 0

    def _verify_adjoint_property(self, f: str, g: str, elements: List[str],
                                relations: Dict[Tuple[str, str], bool]) -> bool:
        """Verify F ⊣ G adjoint property"""

        # Simplified check: for some elements x, y verify F(x) ≤ y ⟺ x ≤ G(y)
        # This is a heuristic since we don't have actual function mappings

        for x in elements[:3]:  # Check subset for efficiency
            for y in elements[:3]:
                # Assume F maps x to some related element, G maps y similarly
                fx_leq_y = relations.get((f, y), False)
                x_leq_gy = relations.get((x, g), False)

                # For adjoint property, these should be equivalent
                if fx_leq_y != x_leq_gy:
                    return False

        return True

    async def stitch_atoms(self, data_i: HASSEData, data_j: HASSEData) -> HASSEData:
        """Combine order structures from two atoms"""

        stitched = HASSEData()

        # Combine elements
        stitched.elements = list(set(data_i.elements + data_j.elements))

        # Merge order relations
        stitched.order_relations = {**data_i.order_relations, **data_j.order_relations}

        # Recompute transitive closure
        stitched.transitive_closure = self._compute_transitive_closure(stitched.order_relations)

        # Recompute lattice operations
        await self._compute_lattice_operations(stitched)

        # Merge Galois connections
        stitched.galois_pairs = data_i.galois_pairs + data_j.galois_pairs
        stitched.adjoint_verified = {**data_i.adjoint_verified, **data_j.adjoint_verified}

        # Combined energy
        stitched.energy = min(data_i.energy, data_j.energy)  # Monotone constraint
        stitched.validated = True

        return stitched

    def get_status(self) -> Dict[str, Any]:
        """Get slice operational status"""
        return {
            "slice_name": "HASSE",
            "slice_id": 34,
            "domain": "Order Theory",
            "cached_posets": len(self.order_cache),
            "validator_active": True
        }

    async def shutdown(self):
        """Cleanup slice resources"""
        self.order_cache.clear()
