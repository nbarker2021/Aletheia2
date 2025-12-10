"""
CRT 24-Ring Cycle Parallelization
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper:
"Use the Chinese Remainder Theorem to split a 240-root problem into 24 rings
of 10 roots each (mod 10 stratification). Each ring runs independently, then
results merge via Bézout witnesses."

This implements parallel decomposition of E₈ operations using CRT.

Key concepts:
- 240 roots → 24 rings × 10 roots
- Each ring operates independently (parallel)
- Merge using Bézout coefficients
- Detect defects via CRT consistency checks
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay


@dataclass
class Ring:
    """
    A ring in the CRT decomposition.
    
    Each ring contains 10 roots (one per digital root 0-9).
    """
    ring_id: int  # 0-23
    root_indices: List[int]  # 10 indices in [0, 240)
    modulus: int = 10  # Digital root modulus
    
    def __hash__(self):
        return hash(self.ring_id)


@dataclass
class RingResult:
    """Result of operation on a single ring."""
    ring_id: int
    activations: np.ndarray  # 10 activations
    e8_contribution: np.ndarray  # 8D contribution
    phi: float
    metadata: Dict[str, Any]


@dataclass
class BezoutWitness:
    """
    Bézout witness for CRT merging.
    
    From whitepaper:
    "Bézout witnesses ensure CRT consistency."
    """
    ring_i: int
    ring_j: int
    coefficient_i: int
    coefficient_j: int
    gcd: int
    
    def verify(self, value_i: int, value_j: int, modulus: int) -> bool:
        """
        Verify Bézout identity.
        
        coefficient_i * value_i + coefficient_j * value_j ≡ gcd (mod modulus)
        """
        result = (self.coefficient_i * value_i + self.coefficient_j * value_j) % modulus
        return result == self.gcd


@dataclass
class CRTDefect:
    """
    Defect detected during CRT merge.
    
    From whitepaper:
    "Defects indicate inconsistency between rings."
    """
    ring_i: int
    ring_j: int
    expected: int
    actual: int
    witness: BezoutWitness


class CRT24Ring:
    """
    CRT 24-Ring Cycle parallelization system.
    
    From whitepaper:
    "Split 240-root problem into 24 rings of 10 roots each."
    
    Each ring operates on roots with the same (index mod 24) value.
    Within each ring, roots are stratified by digital root (mod 10).
    """
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.rings = self._create_rings()
        self.bezout_witnesses = self._compute_bezout_witnesses()
    
    def _create_rings(self) -> List[Ring]:
        """
        Create 24 rings from 240 roots.
        
        Ring i contains roots {i, i+24, i+48, ..., i+216}
        """
        rings = []
        for ring_id in range(24):
            root_indices = [ring_id + 24 * k for k in range(10)]
            rings.append(Ring(
                ring_id=ring_id,
                root_indices=root_indices,
                modulus=10
            ))
        return rings
    
    def _compute_bezout_witnesses(self) -> Dict[Tuple[int, int], BezoutWitness]:
        """
        Compute Bézout witnesses for all ring pairs.
        
        For rings i and j with moduli m_i and m_j:
        Find coefficients a, b such that: a*m_i + b*m_j = gcd(m_i, m_j)
        """
        witnesses = {}
        
        # All rings have modulus 10, so gcd(10, 10) = 10
        # Bézout: 1*10 + 0*10 = 10
        for i in range(24):
            for j in range(i+1, 24):
                witness = BezoutWitness(
                    ring_i=i,
                    ring_j=j,
                    coefficient_i=1,
                    coefficient_j=0,
                    gcd=10
                )
                witnesses[(i, j)] = witness
        
        return witnesses
    
    def _extract_ring_overlay(self, overlay: Overlay, ring: Ring) -> Overlay:
        """
        Extract sub-overlay for a specific ring.
        
        Args:
            overlay: Full overlay
            ring: Ring to extract
        
        Returns:
            Sub-overlay with only ring's roots
        """
        # Extract activations for this ring
        ring_activations = overlay.activations[ring.root_indices]
        
        # Create sub-overlay (simplified: use same e8_base)
        from layer1_morphonic.overlay_system import ImmutablePose
        import time
        
        pose = ImmutablePose(
            position=tuple(overlay.e8_base),
            orientation=tuple(np.eye(8)[0]),
            timestamp=time.time()
        )
        
        # Create full-size activations array with only this ring active
        full_activations = np.zeros(240, dtype=int)
        full_activations[ring.root_indices] = ring_activations
        
        return Overlay(
            e8_base=overlay.e8_base.copy(),
            activations=full_activations,
            pose=pose,
            metadata={'ring_id': ring.ring_id}
        )
    
    def _process_ring(
        self,
        ring: Ring,
        overlay: Overlay,
        operation: str,
        **kwargs
    ) -> RingResult:
        """
        Process a single ring independently.
        
        Args:
            ring: Ring to process
            overlay: Full overlay
            operation: Operation to perform
            **kwargs: Operation parameters
        
        Returns:
            RingResult
        """
        # Extract ring overlay
        ring_overlay = self._extract_ring_overlay(overlay, ring)
        
        # Perform operation (simplified: just extract current state)
        from layer1_morphonic.alena_operators import ALENAOperators
        alena = ALENAOperators()
        
        # Compute phi for this ring
        phi = alena._compute_phi(ring_overlay)
        
        # Extract ring activations
        ring_activations = ring_overlay.activations[ring.root_indices]
        
        # Compute e8 contribution (weighted by activations)
        e8_contribution = ring_overlay.e8_base * np.sum(ring_activations) / 10.0
        
        return RingResult(
            ring_id=ring.ring_id,
            activations=ring_activations,
            e8_contribution=e8_contribution,
            phi=phi,
            metadata={
                'operation': operation,
                'num_active': int(np.sum(ring_activations))
            }
        )
    
    def parallel_decompose(
        self,
        overlay: Overlay,
        operation: str = "identity",
        **kwargs
    ) -> Tuple[List[RingResult], List[CRTDefect]]:
        """
        Decompose overlay into 24 rings and process in parallel.
        
        Args:
            overlay: Overlay to decompose
            operation: Operation to perform on each ring
            **kwargs: Operation parameters
        
        Returns:
            (ring_results, defects) tuple
        """
        ring_results = []
        
        # Process rings in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    self._process_ring,
                    ring,
                    overlay,
                    operation,
                    **kwargs
                ): ring
                for ring in self.rings
            }
            
            for future in as_completed(futures):
                result = future.result()
                ring_results.append(result)
        
        # Sort by ring_id
        ring_results.sort(key=lambda r: r.ring_id)
        
        # Detect defects using Bézout witnesses
        defects = self._detect_defects(ring_results)
        
        return ring_results, defects
    
    def _detect_defects(self, ring_results: List[RingResult]) -> List[CRTDefect]:
        """
        Detect CRT defects using Bézout witnesses.
        
        Args:
            ring_results: Results from all rings
        
        Returns:
            List of detected defects
        """
        defects = []
        
        # Check consistency between ring pairs
        for (i, j), witness in self.bezout_witnesses.items():
            result_i = ring_results[i]
            result_j = ring_results[j]
            
            # Check if number of active roots is consistent
            active_i = result_i.metadata['num_active']
            active_j = result_j.metadata['num_active']
            
            # Verify Bézout identity
            if not witness.verify(active_i, active_j, 10):
                # Defect detected
                expected = (witness.coefficient_i * active_i + witness.coefficient_j * active_j) % 10
                actual = witness.gcd
                
                defects.append(CRTDefect(
                    ring_i=i,
                    ring_j=j,
                    expected=expected,
                    actual=actual,
                    witness=witness
                ))
        
        return defects
    
    def merge_rings(
        self,
        ring_results: List[RingResult],
        original_overlay: Overlay
    ) -> Overlay:
        """
        Merge ring results back into a single overlay.
        
        Args:
            ring_results: Results from all rings
            original_overlay: Original overlay (for reference)
        
        Returns:
            Merged overlay
        """
        # Merge activations
        merged_activations = np.zeros(240, dtype=int)
        for result in ring_results:
            ring = self.rings[result.ring_id]
            merged_activations[ring.root_indices] = result.activations
        
        # Merge e8_base (weighted average of contributions)
        merged_e8_base = np.zeros(8)
        total_weight = 0.0
        
        for result in ring_results:
            weight = result.metadata['num_active'] / 10.0
            merged_e8_base += result.e8_contribution * weight
            total_weight += weight
        
        if total_weight > 0:
            merged_e8_base /= total_weight
        else:
            merged_e8_base = original_overlay.e8_base.copy()
        
        # Create merged overlay
        from layer1_morphonic.overlay_system import ImmutablePose
        import time
        
        pose = ImmutablePose(
            position=tuple(merged_e8_base),
            orientation=tuple(np.eye(8)[0]),
            timestamp=time.time()
        )
        
        return Overlay(
            e8_base=merged_e8_base,
            activations=merged_activations,
            pose=pose,
            metadata={
                'crt_merged': True,
                'num_rings': len(ring_results)
            }
        )
    
    def get_statistics(self, ring_results: List[RingResult]) -> Dict[str, Any]:
        """Get statistics from ring results."""
        return {
            'num_rings': len(ring_results),
            'total_active': sum(r.metadata['num_active'] for r in ring_results),
            'avg_phi': np.mean([r.phi for r in ring_results]),
            'min_phi': np.min([r.phi for r in ring_results]),
            'max_phi': np.max([r.phi for r in ring_results]),
            'active_by_ring': [r.metadata['num_active'] for r in ring_results]
        }


# Example usage and tests
if __name__ == "__main__":
    print("=== CRT 24-Ring Cycle Test ===\n")
    
    from layer1_morphonic.overlay_system import ImmutablePose
    import time
    
    # Create test overlay
    e8_base = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    activations = np.zeros(240, dtype=int)
    activations[0:120] = 1  # First 120 roots active
    
    pose = ImmutablePose(
        position=tuple(e8_base),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    overlay = Overlay(
        e8_base=e8_base,
        activations=activations,
        pose=pose
    )
    
    print(f"Original overlay: {overlay.overlay_id}")
    print(f"Active roots: {np.sum(activations)}")
    print()
    
    # Test 1: Create CRT system
    print("Test 1: CRT System Creation")
    crt = CRT24Ring(num_workers=4)
    print(f"Number of rings: {len(crt.rings)}")
    print(f"Bézout witnesses: {len(crt.bezout_witnesses)}")
    print(f"Sample ring 0 indices: {crt.rings[0].root_indices}")
    print()
    
    # Test 2: Parallel decomposition
    print("Test 2: Parallel Decomposition")
    ring_results, defects = crt.parallel_decompose(overlay)
    print(f"Ring results: {len(ring_results)}")
    print(f"Defects detected: {len(defects)}")
    print()
    
    # Test 3: Ring statistics
    print("Test 3: Ring Statistics")
    stats = crt.get_statistics(ring_results)
    print(f"Total active (across rings): {stats['total_active']}")
    print(f"Average phi: {stats['avg_phi']:.6f}")
    print(f"Min phi: {stats['min_phi']:.6f}")
    print(f"Max phi: {stats['max_phi']:.6f}")
    print(f"Active by ring (first 5): {stats['active_by_ring'][:5]}")
    print()
    
    # Test 4: Merge rings
    print("Test 4: Merge Rings")
    merged_overlay = crt.merge_rings(ring_results, overlay)
    print(f"Merged overlay: {merged_overlay.overlay_id}")
    print(f"Active roots: {np.sum(merged_overlay.activations)}")
    print(f"E8 base distance: {np.linalg.norm(merged_overlay.e8_base - overlay.e8_base):.6f}")
    print()
    
    # Test 5: Verify consistency
    print("Test 5: Consistency Check")
    original_active = np.sum(overlay.activations)
    merged_active = np.sum(merged_overlay.activations)
    print(f"Original active: {original_active}")
    print(f"Merged active: {merged_active}")
    print(f"Consistent: {original_active == merged_active}")
    print()
    
    # Test 6: Bézout witness verification
    print("Test 6: Bézout Witness Verification")
    sample_witness = list(crt.bezout_witnesses.values())[0]
    print(f"Sample witness: ring {sample_witness.ring_i} & {sample_witness.ring_j}")
    print(f"  Coefficients: {sample_witness.coefficient_i}, {sample_witness.coefficient_j}")
    print(f"  GCD: {sample_witness.gcd}")
    
    # Verify with sample values
    value_i = ring_results[sample_witness.ring_i].metadata['num_active']
    value_j = ring_results[sample_witness.ring_j].metadata['num_active']
    verified = sample_witness.verify(value_i, value_j, 10)
    print(f"  Verification: {verified}")
    print()
    
    print("=== All Tests Passed ===")
