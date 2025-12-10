"""
QuadraticLawHarness - Validation Framework
Cartan-Quadratic Equivalence (CQE) System

From Master Document:
"The QuadraticLawHarness serves as a critical validation platform for the
theoretical underpinnings of Cartan Quadratic Equivalence."

Tests performed:
1. CNF Path-Independence
2. Boundary-Only Emission
3. Φ-Probe Determinism
4. CRT Defect Detection
5. Receipt Schema Compliance
6. Ledger Sanity
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay
from layer1_morphonic.alena_operators import ALENAOperators, OperationResult
from layer1_morphonic.acceptance_rules import AcceptanceRule, ParitySignature


@dataclass
class TestResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    details: Dict[str, Any]
    error: Optional[str] = None


class QuadraticLawHarness:
    """
    Validation framework for CQE theoretical underpinnings.
    
    Performs six critical tests to ensure system integrity:
    1. CNF path-independence
    2. Boundary-only emission
    3. Φ-probe determinism
    4. CRT defect detection
    5. Receipt schema compliance
    6. Ledger sanity
    """
    
    def __init__(self):
        self.alena = ALENAOperators()
        self.acceptance_rule = AcceptanceRule()
        self.test_results: List[TestResult] = []
    
    def test_cnf_path_independence(
        self,
        overlay: Overlay,
        operations: List[Tuple[str, dict]]
    ) -> TestResult:
        """
        Test 1: CNF Path-Independence
        
        From whitepaper:
        "CNF path-independence: verified using two different operation orders
        (emit→couple vs couple→emit) with governance stripped; CNF tuples matched."
        
        Verifies that operation order doesn't affect final state.
        
        Args:
            overlay: Initial overlay
            operations: List of (operation_name, parameters) tuples
        
        Returns:
            TestResult
        """
        test_name = "CNF Path-Independence"
        
        try:
            # Path 1: Apply operations in given order
            current1 = overlay
            for op_name, params in operations:
                if op_name == "rotate":
                    result = self.alena.rotate(current1, **params)
                elif op_name == "weyl_reflect":
                    result = self.alena.weyl_reflect(current1, **params)
                elif op_name == "midpoint":
                    result = self.alena.midpoint(current1, current1, **params)
                elif op_name == "parity_mirror":
                    result = self.alena.parity_mirror(current1)
                else:
                    raise ValueError(f"Unknown operation: {op_name}")
                current1 = result.overlay
            
            # Path 2: Apply operations in reverse order
            current2 = overlay
            for op_name, params in reversed(operations):
                if op_name == "rotate":
                    result = self.alena.rotate(current2, **params)
                elif op_name == "weyl_reflect":
                    result = self.alena.weyl_reflect(current2, **params)
                elif op_name == "midpoint":
                    result = self.alena.midpoint(current2, current2, **params)
                elif op_name == "parity_mirror":
                    result = self.alena.parity_mirror(current2)
                current2 = result.overlay
            
            # Compare final states (should be equivalent under CQE)
            # Check if activations match
            activations_match = np.array_equal(current1.activations, current2.activations)
            
            # Check if parity signatures match
            parity1 = ParitySignature.from_overlay(current1)
            parity2 = ParitySignature.from_overlay(current2)
            parity_match = (parity1 == parity2)
            
            # Check if e8_base is close (allow small numerical error)
            position_close = np.allclose(current1.e8_base, current2.e8_base, atol=1e-6)
            
            passed = activations_match and parity_match and position_close
            
            result = TestResult(
                test_name=test_name,
                passed=passed,
                details={
                    'activations_match': activations_match,
                    'parity_match': parity_match,
                    'position_close': position_close,
                    'path1_id': current1.overlay_id,
                    'path2_id': current2.overlay_id,
                    'parity1': parity1.to_dict(),
                    'parity2': parity2.to_dict()
                }
            )
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                passed=False,
                details={},
                error=str(e)
            )
        
        self.test_results.append(result)
        return result
    
    def test_boundary_only_emission(
        self,
        overlay: Overlay,
        num_interior_ops: int = 10
    ) -> TestResult:
        """
        Test 2: Boundary-Only Emission
        
        From whitepaper:
        "Boundary-only emission: a toy Z² walk stayed ΔS-free inside a single
        cell and only registered events at cell crossings."
        
        Verifies that interior operations generate no entropy.
        
        Args:
            overlay: Initial overlay
            num_interior_ops: Number of interior operations to test
        
        Returns:
            TestResult
        """
        test_name = "Boundary-Only Emission"
        
        try:
            # Perform interior operations (small rotations within cell)
            current = overlay
            interior_deltas = []
            
            for i in range(num_interior_ops):
                # Small rotation (interior operation)
                theta = 0.01 * (i + 1)  # Small angles
                result = self.alena.rotate(current, theta=theta)
                interior_deltas.append(abs(result.delta_phi))
                current = result.overlay
            
            # Check that all interior deltas are near zero
            max_interior_delta = max(interior_deltas)
            interior_entropy_free = max_interior_delta < 1e-6
            
            # Now perform boundary operation (large transformation)
            boundary_result = self.alena.parity_mirror(current)
            boundary_delta = abs(boundary_result.delta_phi)
            
            # Boundary should have significant delta
            boundary_has_entropy = boundary_delta > 1e-6
            
            passed = interior_entropy_free and boundary_has_entropy
            
            result = TestResult(
                test_name=test_name,
                passed=passed,
                details={
                    'num_interior_ops': num_interior_ops,
                    'max_interior_delta': max_interior_delta,
                    'interior_entropy_free': interior_entropy_free,
                    'boundary_delta': boundary_delta,
                    'boundary_has_entropy': boundary_has_entropy,
                    'interior_deltas': interior_deltas
                }
            )
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                passed=False,
                details={},
                error=str(e)
            )
        
        self.test_results.append(result)
        return result
    
    def test_phi_probe_determinism(
        self,
        overlay1: Overlay,
        overlay2: Overlay,
        num_permutations: int = 10
    ) -> TestResult:
        """
        Test 3: Φ-Probe Determinism
        
        From whitepaper:
        "Φ-probe determinism: confirmed a stable winner across many
        permutations under a φ-weighted tie salt."
        
        Verifies that tie-breaking is deterministic.
        
        Args:
            overlay1: First overlay
            overlay2: Second overlay
            num_permutations: Number of permutations to test
        
        Returns:
            TestResult
        """
        test_name = "Φ-Probe Determinism"
        
        try:
            # Compute phi for both overlays
            phi1 = self.alena._compute_phi(overlay1)
            phi2 = self.alena._compute_phi(overlay2)
            
            # Perform midpoint operation multiple times with permutations
            winners = []
            
            for i in range(num_permutations):
                # Add small random perturbation (tie salt)
                weight = 0.5 + 0.01 * np.random.randn()
                weight = np.clip(weight, 0.0, 1.0)
                
                result = self.alena.midpoint(overlay1, overlay2, weight=weight)
                
                # Determine winner based on phi
                phi_mid = self.alena._compute_phi(result.overlay)
                if abs(phi_mid - phi1) < abs(phi_mid - phi2):
                    winners.append(1)
                else:
                    winners.append(2)
            
            # Check if winner is consistent
            unique_winners = set(winners)
            deterministic = len(unique_winners) == 1
            
            result = TestResult(
                test_name=test_name,
                passed=deterministic,
                details={
                    'num_permutations': num_permutations,
                    'unique_winners': list(unique_winners),
                    'winner_counts': {
                        1: winners.count(1),
                        2: winners.count(2)
                    },
                    'deterministic': deterministic,
                    'phi1': phi1,
                    'phi2': phi2
                }
            )
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                passed=False,
                details={},
                error=str(e)
            )
        
        self.test_results.append(result)
        return result
    
    def test_crt_defect_detection(
        self,
        moduli: List[int] = [3, 6, 9]
    ) -> TestResult:
        """
        Test 4: CRT Defect Detection
        
        From whitepaper:
        "CRT defect + Bézout: detected non-uniqueness (gcd>1) and produced
        a valid Bézout witness."
        
        Verifies CRT defect detection and Bézout witness generation.
        
        Args:
            moduli: List of moduli to test
        
        Returns:
            TestResult
        """
        test_name = "CRT Defect Detection"
        
        try:
            # Check if moduli are pairwise coprime
            defects = []
            bezout_witnesses = []
            
            for i in range(len(moduli)):
                for j in range(i+1, len(moduli)):
                    m1, m2 = moduli[i], moduli[j]
                    gcd = np.gcd(m1, m2)
                    
                    if gcd > 1:
                        # Defect detected!
                        defects.append((m1, m2, gcd))
                        
                        # Compute Bézout coefficients: a*m1 + b*m2 = gcd
                        # Using extended Euclidean algorithm
                        def extended_gcd(a, b):
                            if a == 0:
                                return b, 0, 1
                            gcd, x1, y1 = extended_gcd(b % a, a)
                            x = y1 - (b // a) * x1
                            y = x1
                            return gcd, x, y
                        
                        gcd_calc, a, b = extended_gcd(m1, m2)
                        
                        # Verify Bézout identity
                        bezout_valid = (a * m1 + b * m2 == gcd_calc)
                        bezout_witnesses.append({
                            'm1': m1,
                            'm2': m2,
                            'gcd': gcd,
                            'a': a,
                            'b': b,
                            'valid': bezout_valid
                        })
            
            # Test passes if defects were detected and Bézout witnesses are valid
            passed = len(defects) > 0 and all(w['valid'] for w in bezout_witnesses)
            
            result = TestResult(
                test_name=test_name,
                passed=passed,
                details={
                    'moduli': moduli,
                    'defects_found': len(defects),
                    'defects': defects,
                    'bezout_witnesses': bezout_witnesses
                }
            )
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                passed=False,
                details={},
                error=str(e)
            )
        
        self.test_results.append(result)
        return result
    
    def test_receipt_schema_compliance(
        self,
        overlay: Overlay
    ) -> TestResult:
        """
        Test 5: Receipt Schema Compliance
        
        From whitepaper:
        "Receipt schema compliance: generated a synthetic CNF boundary
        receipt and validated core fields against your schema; saved it."
        
        Verifies that receipts conform to schema.
        
        Args:
            overlay: Overlay to generate receipt for
        
        Returns:
            TestResult
        """
        test_name = "Receipt Schema Compliance"
        
        try:
            # Generate synthetic CNF boundary receipt
            receipt = {
                'overlay_id': overlay.overlay_id,
                'timestamp': overlay.creation_time,
                'e8_base': overlay.e8_base.tolist(),
                'activations': overlay.activations.tolist(),
                'num_active': int(np.sum(overlay.activations)),
                'parity_signature': ParitySignature.from_overlay(overlay).to_dict(),
                'phi': self.alena._compute_phi(overlay),
                'metadata': overlay.metadata
            }
            
            # Validate required fields
            required_fields = [
                'overlay_id', 'timestamp', 'e8_base', 'activations',
                'num_active', 'parity_signature', 'phi'
            ]
            
            missing_fields = [f for f in required_fields if f not in receipt]
            has_all_fields = len(missing_fields) == 0
            
            # Validate field types
            type_checks = {
                'overlay_id': isinstance(receipt['overlay_id'], str),
                'timestamp': isinstance(receipt['timestamp'], (int, float)),
                'e8_base': isinstance(receipt['e8_base'], list),
                'activations': isinstance(receipt['activations'], list),
                'num_active': isinstance(receipt['num_active'], int),
                'parity_signature': isinstance(receipt['parity_signature'], dict),
                'phi': isinstance(receipt['phi'], (int, float))
            }
            
            all_types_valid = all(type_checks.values())
            
            passed = has_all_fields and all_types_valid
            
            result = TestResult(
                test_name=test_name,
                passed=passed,
                details={
                    'has_all_fields': has_all_fields,
                    'missing_fields': missing_fields,
                    'type_checks': type_checks,
                    'all_types_valid': all_types_valid,
                    'receipt': receipt
                }
            )
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                passed=False,
                details={},
                error=str(e)
            )
        
        self.test_results.append(result)
        return result
    
    def test_ledger_sanity(
        self,
        ledger_data: List[Dict[str, Any]]
    ) -> TestResult:
        """
        Test 6: Ledger Sanity
        
        From whitepaper:
        "Ledger sanity: parsed your existing ledger.json and checked
        non-negative ΔS, boundary-only logging, and ISO-clean timestamps."
        
        Verifies ledger integrity.
        
        Args:
            ledger_data: List of ledger entries
        
        Returns:
            TestResult
        """
        test_name = "Ledger Sanity"
        
        try:
            checks = {
                'non_negative_delta_s': True,
                'boundary_only': True,
                'iso_timestamps': True
            }
            
            issues = []
            
            for i, entry in enumerate(ledger_data):
                # Check non-negative ΔS
                if 'delta_phi' in entry:
                    if entry['delta_phi'] < -1e-6:  # Allow small numerical error
                        checks['non_negative_delta_s'] = False
                        issues.append(f"Entry {i}: negative ΔΦ = {entry['delta_phi']}")
                
                # Check boundary-only logging
                if 'operation' in entry:
                    # Interior operations should have near-zero delta
                    if entry['operation'] in ['rotate', 'weyl_reflect']:
                        if 'delta_phi' in entry and abs(entry['delta_phi']) > 1e-3:
                            checks['boundary_only'] = False
                            issues.append(f"Entry {i}: interior op with large ΔΦ")
                
                # Check ISO timestamps
                if 'timestamp' in entry:
                    if not isinstance(entry['timestamp'], (int, float)):
                        checks['iso_timestamps'] = False
                        issues.append(f"Entry {i}: invalid timestamp type")
            
            passed = all(checks.values())
            
            result = TestResult(
                test_name=test_name,
                passed=passed,
                details={
                    'num_entries': len(ledger_data),
                    'checks': checks,
                    'issues': issues
                }
            )
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                passed=False,
                details={},
                error=str(e)
            )
        
        self.test_results.append(result)
        return result
    
    def run_all_tests(
        self,
        overlay1: Overlay,
        overlay2: Optional[Overlay] = None
    ) -> Dict[str, Any]:
        """
        Run all validation tests.
        
        Args:
            overlay1: Primary test overlay
            overlay2: Secondary test overlay (optional)
        
        Returns:
            Summary of all test results
        """
        print("=== Running QuadraticLawHarness Validation ===\n")
        
        # Test 1: CNF Path-Independence
        print("Test 1: CNF Path-Independence...")
        operations = [
            ("rotate", {"theta": 0.1}),
            ("weyl_reflect", {"root_index": 5})
        ]
        result1 = self.test_cnf_path_independence(overlay1, operations)
        print(f"  Result: {'PASS' if result1.passed else 'FAIL'}\n")
        
        # Test 2: Boundary-Only Emission
        print("Test 2: Boundary-Only Emission...")
        result2 = self.test_boundary_only_emission(overlay1)
        print(f"  Result: {'PASS' if result2.passed else 'FAIL'}\n")
        
        # Test 3: Φ-Probe Determinism
        print("Test 3: Φ-Probe Determinism...")
        if overlay2 is None:
            overlay2 = overlay1.clone(phase=0.5)
        result3 = self.test_phi_probe_determinism(overlay1, overlay2)
        print(f"  Result: {'PASS' if result3.passed else 'FAIL'}\n")
        
        # Test 4: CRT Defect Detection
        print("Test 4: CRT Defect Detection...")
        result4 = self.test_crt_defect_detection()
        print(f"  Result: {'PASS' if result4.passed else 'FAIL'}\n")
        
        # Test 5: Receipt Schema Compliance
        print("Test 5: Receipt Schema Compliance...")
        result5 = self.test_receipt_schema_compliance(overlay1)
        print(f"  Result: {'PASS' if result5.passed else 'FAIL'}\n")
        
        # Test 6: Ledger Sanity
        print("Test 6: Ledger Sanity...")
        # Generate synthetic ledger from operation history
        ledger_data = [
            {
                'operation': op.operation,
                'delta_phi': op.delta_phi,
                'timestamp': op.overlay.creation_time
            }
            for op in self.alena.get_history()
        ]
        result6 = self.test_ledger_sanity(ledger_data)
        print(f"  Result: {'PASS' if result6.passed else 'FAIL'}\n")
        
        # Summary
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.passed)
        
        summary = {
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / total if total > 0 else 0.0,
            'results': [
                {
                    'test': r.test_name,
                    'passed': r.passed,
                    'error': r.error
                }
                for r in self.test_results
            ]
        }
        
        print(f"=== Summary ===")
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Pass rate: {summary['pass_rate']:.1%}")
        
        return summary


# Example usage
if __name__ == "__main__":
    from layer1_morphonic.overlay_system import ImmutablePose
    import time
    
    # Create test overlays
    e8_base1 = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    activations1 = np.zeros(240, dtype=int)
    activations1[0:120] = 1
    
    pose1 = ImmutablePose(
        position=tuple(e8_base1),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    overlay1 = Overlay(e8_base=e8_base1, activations=activations1, pose=pose1)
    
    # Run harness
    harness = QuadraticLawHarness()
    summary = harness.run_all_tests(overlay1)
    
    print(f"\n=== QuadraticLawHarness Complete ===")
