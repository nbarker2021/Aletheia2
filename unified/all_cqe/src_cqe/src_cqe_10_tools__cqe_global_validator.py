"""
CQE Global Validator - 37-Slice Promotion DSL Engine

Implements the extended global promotion rule that validates operations
across all mathematical slices using the CQE promotion DSL.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from .atom import UniversalAtom, SliceData

@dataclass
class ValidationResult:
    """Result of a validation operation"""
    is_valid: bool
    reason: str
    energy_delta: float = 0.0
    approved_slices: List[str] = None
    failed_constraints: List[str] = None

    def __post_init__(self):
        if self.approved_slices is None:
            self.approved_slices = []
        if self.failed_constraints is None:
            self.failed_constraints = []

class SliceValidator(ABC):
    """Abstract base class for slice-specific validators"""

    @abstractmethod
    async def validate_promotion(self, atom_i: UniversalAtom, atom_j: UniversalAtom) -> ValidationResult:
        """Validate promotion from atom_i to atom_j according to slice axioms"""
        pass

class GlobalValidator:
    """
    Global CQE Promotion Validator

    Implements the extended DSL:
    PROMOTE(i,j) iff
      PARITY.ok(edge_ij) AND
      NOETHER.Conservation.ok(i,j, bands(8,24)) AND  
      GROTH.Sheaf.glueOK(i,j, overlays≥2, witness_cover≥3) AND
      GOVERNANCE.deltaBandOK(i,j, max=1) AND
      (SLICE_VALIDATORS_37_OR_COMBINATION) AND
      Φ_total_new ≤ Φ_total_prev
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.slice_validators: Dict[str, SliceValidator] = {}

        # Validation statistics  
        self.validations_performed = 0
        self.validations_passed = 0
        self.validation_failures: Dict[str, int] = {}

    async def initialize(self, slices: Dict[str, Any]):
        """Initialize validator with all slice validators"""
        self.logger.info("Initializing Global Validator with 37 slice validators")

        for slice_name, slice_instance in slices.items():
            if hasattr(slice_instance, 'validator'):
                self.slice_validators[slice_name] = slice_instance.validator
                self.logger.debug(f"Registered validator for slice: {slice_name}")

    async def validate_atom(self, atom: UniversalAtom) -> ValidationResult:
        """Validate a single atom for consistency across all slices"""
        try:
            # Check basic atom integrity
            if not atom.id or not atom.sha256:
                return ValidationResult(False, "atom_integrity_fail")

            # Verify E8 coordinates
            if len(atom.e8_coordinates) != 8:
                return ValidationResult(False, "invalid_e8_coordinates")

            # Check slice data consistency
            inconsistent_slices = []
            for slice_name, slice_data in atom.slice_data.items():
                if not slice_data.validated:
                    inconsistent_slices.append(slice_name)

            if inconsistent_slices:
                return ValidationResult(
                    False, 
                    "slice_data_unvalidated",
                    failed_constraints=inconsistent_slices
                )

            return ValidationResult(True, "atom_validated")

        except Exception as e:
            self.logger.error(f"Atom validation error: {e}")
            return ValidationResult(False, f"validation_exception: {e}")

    async def validate_promotion(self, atom_i: UniversalAtom, atom_j: UniversalAtom) -> ValidationResult:
        """
        Validate promotion from atom_i to atom_j using complete 37-slice DSL

        This is the core CQE promotion validation that implements the
        mathematical framework for slice stitching operations.
        """
        try:
            self.validations_performed += 1

            # 1. PARITY.ok(edge_ij) - 64-bit parity lane verification
            if not self._validate_parity(atom_i, atom_j):
                self._record_failure("parity_violation")
                return ValidationResult(False, "parity_lanes_incompatible")

            # 2. NOETHER.Conservation.ok(i,j, bands(8,24)) - Conservation laws  
            if not self._validate_noether_conservation(atom_i, atom_j):
                self._record_failure("noether_violation")
                return ValidationResult(False, "noether_conservation_violated")

            # 3. GROTH.Sheaf.glueOK(i,j, overlays≥2, witness_cover≥3) - Sheaf cohomology
            if not self._validate_grothendieck_sheaf_glue(atom_i, atom_j):
                self._record_failure("grothendieck_violation")
                return ValidationResult(False, "sheaf_gluing_failed")

            # 4. GOVERNANCE.deltaBandOK(i,j, max=1) - Governance band constraints
            if not self._validate_governance_bands(atom_i, atom_j):
                self._record_failure("governance_violation")  
                return ValidationResult(False, "governance_band_violation")

            # 5. Slice validator checks - At least one slice must approve
            approved_slices = []
            slice_results = {}

            for slice_name, validator in self.slice_validators.items():
                try:
                    result = await validator.validate_promotion(atom_i, atom_j)
                    slice_results[slice_name] = result

                    if result.is_valid:
                        approved_slices.append(slice_name)

                except Exception as e:
                    self.logger.warning(f"Slice validator {slice_name} error: {e}")
                    slice_results[slice_name] = ValidationResult(False, f"validator_error: {e}")

            if not approved_slices:
                self._record_failure("no_slice_approval")
                return ValidationResult(
                    False, 
                    "no_slice_validators_approved",
                    failed_constraints=list(slice_results.keys())
                )

            # 6. Global energy constraint: Φ_total_new ≤ Φ_total_prev  
            energy_delta = atom_j.get_total_energy() - atom_i.get_total_energy()
            if energy_delta > 0:
                self._record_failure("energy_increase")
                return ValidationResult(False, "global_energy_increased", energy_delta)

            # All constraints satisfied - promotion approved
            self.validations_passed += 1

            return ValidationResult(
                True, 
                "promotion_approved",
                energy_delta=energy_delta,
                approved_slices=approved_slices
            )

        except Exception as e:
            self.logger.error(f"Promotion validation error: {e}")
            return ValidationResult(False, f"validation_exception: {e}")

    def _validate_parity(self, atom_i: UniversalAtom, atom_j: UniversalAtom) -> bool:
        """Validate 64-bit parity lanes compatibility"""
        # XOR the parity lanes - should maintain even parity
        parity_xor = bytes(a ^ b for a, b in zip(atom_i.parity_lanes, atom_j.parity_lanes))

        # Check even parity across all lanes
        total_bits = sum(bin(byte).count('1') for byte in parity_xor)
        return total_bits % 2 == 0

    def _validate_noether_conservation(self, atom_i: UniversalAtom, atom_j: UniversalAtom) -> bool:
        """Validate Noether conservation laws across governance bands"""
        # Conservation in band8 and band24
        band8_conserved = (
            atom_i.governance_bands.get("band8", 0) == 
            atom_j.governance_bands.get("band8", 0)
        )

        band24_conserved = (
            atom_i.governance_bands.get("band24", 0) == 
            atom_j.governance_bands.get("band24", 0)
        )

        return band8_conserved and band24_conserved

    def _validate_grothendieck_sheaf_glue(self, atom_i: UniversalAtom, atom_j: UniversalAtom) -> bool:
        """Validate Grothendieck sheaf gluing conditions"""
        # Require at least 2 overlapping slices with 3+ witness coverage
        common_slices = set(atom_i.get_active_slices()) & set(atom_j.get_active_slices())

        if len(common_slices) < 2:
            return False

        # Check witness coverage - simplified heuristic  
        witness_coverage = len(common_slices) + len(atom_i.merkle_path) + len(atom_j.merkle_path)
        return witness_coverage >= 3

    def _validate_governance_bands(self, atom_i: UniversalAtom, atom_j: UniversalAtom) -> bool:
        """Validate governance band delta constraints"""
        # Maximum band change of 1
        for band_type in ["band8", "band24", "tile4096"]:
            delta = abs(
                atom_j.governance_bands.get(band_type, 0) - 
                atom_i.governance_bands.get(band_type, 0)
            )
            if delta > 1:
                return False

        return True

    def _record_failure(self, failure_type: str):
        """Record validation failure statistics"""
        self.validation_failures[failure_type] = self.validation_failures.get(failure_type, 0) + 1

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        success_rate = (self.validations_passed / self.validations_performed 
                       if self.validations_performed > 0 else 0.0)

        return {
            "validations_performed": self.validations_performed,
            "validations_passed": self.validations_passed, 
            "success_rate": success_rate,
            "failure_breakdown": dict(self.validation_failures)
        }
