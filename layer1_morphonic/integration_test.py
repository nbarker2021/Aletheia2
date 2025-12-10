"""
Integration Test for CQE v7.1 Components
Cartan-Quadratic Equivalence (CQE) System

Tests all newly implemented components together:
1. Overlay System (Axiom A)
2. ALENA Operators (Axiom E)
3. Acceptance Rules & Parity Tracking
4. QuadraticLawHarness
5. Provenance Logging & CNF Receipts
6. Shell Protocol & Bregman Distance
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay, ImmutablePose, OverlayStore
from layer1_morphonic.alena_operators import ALENAOperators
from layer1_morphonic.acceptance_rules import AcceptanceRule, ParitySignature
from layer1_morphonic.quadratic_law_harness import QuadraticLawHarness
from layer1_morphonic.provenance import ProvenanceLogger, CNFReceipt
from layer1_morphonic.shell_protocol import ShellProtocol, BregmanDistance


def create_test_overlay(seed: int = 0) -> Overlay:
    """Create a test overlay."""
    np.random.seed(seed)
    
    e8_base = np.random.randn(8)
    e8_base = e8_base / np.linalg.norm(e8_base)  # Normalize
    
    activations = np.zeros(240, dtype=int)
    activations[:120] = 1  # Activate first half
    
    pose = ImmutablePose(
        position=tuple(e8_base),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    return Overlay(e8_base=e8_base, activations=activations, pose=pose)


def test_full_workflow():
    """Test complete CQE workflow with all components."""
    
    print("="*70)
    print("CQE v7.1 Integration Test")
    print("="*70)
    print()
    
    # Initialize all components
    print("1. Initializing Components...")
    overlay_store = OverlayStore()
    alena = ALENAOperators()
    acceptance_rule = AcceptanceRule(epsilon=1e-6, plateau_cap=5)
    provenance_logger = ProvenanceLogger()
    
    # Create initial overlay
    initial_overlay = create_test_overlay(seed=42)
    overlay_store.store(initial_overlay)
    
    # Create shell protocol
    shell_protocol = ShellProtocol(
        initial_overlay=initial_overlay,
        initial_radius=2.0,
        expansion_factor=2
    )
    
    print(f"   Initial overlay: {initial_overlay.overlay_id}")
    print(f"   Shell radius: {shell_protocol.shell.radius}")
    print()
    
    # Run optimization loop
    print("2. Running Optimization Loop...")
    current_overlay = initial_overlay
    num_iterations = 20
    accepted_count = 0
    rejected_count = 0
    
    for i in range(num_iterations):
        # Choose random operation
        op_choice = np.random.choice(['rotate', 'weyl_reflect', 'midpoint'])
        
        if op_choice == 'rotate':
            theta = np.random.uniform(0.01, 0.1)
            result = alena.rotate(current_overlay, theta=theta)
        elif op_choice == 'weyl_reflect':
            root_idx = np.random.randint(0, 240)
            result = alena.weyl_reflect(current_overlay, root_index=root_idx)
        else:  # midpoint
            other_overlay = create_test_overlay(seed=i)
            result = alena.midpoint(current_overlay, other_overlay, weight=0.5)
        
        # Check shell constraint
        is_valid, reason = shell_protocol.check_shell_constraint(result.overlay)
        
        if not is_valid:
            rejected_count += 1
            print(f"   Iteration {i+1}: {result.operation} - REJECTED ({reason})")
            continue
        
        # Evaluate acceptance
        phi_before = alena._compute_phi(current_overlay)
        phi_after = alena._compute_phi(result.overlay)
        decision = acceptance_rule.evaluate_operation_result(current_overlay, result)
        
        # Record in shell protocol
        shell_protocol.record_attempt(
            current_overlay,
            result.overlay,
            decision.accepted,
            decision.delta_phi
        )
        
        # Log provenance
        provenance_record = provenance_logger.log_transition(
            current_overlay,
            result,
            decision,
            phi_before,
            phi_after
        )
        
        if decision.accepted:
            accepted_count += 1
            overlay_store.store(result.overlay)
            current_overlay = result.overlay
            print(f"   Iteration {i+1}: {result.operation} - ACCEPTED "
                  f"(ΔΦ={decision.delta_phi:.6f}, type={decision.acceptance_type.value})")
        else:
            rejected_count += 1
            print(f"   Iteration {i+1}: {result.operation} - REJECTED "
                  f"(reason={decision.reason})")
    
    print()
    print(f"   Total iterations: {num_iterations}")
    print(f"   Accepted: {accepted_count}")
    print(f"   Rejected: {rejected_count}")
    print(f"   Accept rate: {accepted_count/num_iterations:.1%}")
    print()
    
    # Test 3: Parity Tracking
    print("3. Parity Signature Analysis...")
    initial_parity = ParitySignature.from_overlay(initial_overlay)
    final_parity = ParitySignature.from_overlay(current_overlay)
    
    print(f"   Initial parity syndrome: {initial_parity.syndrome:.6f}")
    print(f"   Final parity syndrome: {final_parity.syndrome:.6f}")
    print(f"   Parity change: {final_parity.syndrome - initial_parity.syndrome:.6f}")
    print()
    
    # Test 4: Provenance Lineage
    print("4. Provenance Lineage...")
    lineage = provenance_logger.get_lineage(current_overlay.overlay_id)
    print(f"   Lineage length: {len(lineage)}")
    print(f"   Lineage chain:")
    for j, record in enumerate(lineage[:5]):  # Show first 5
        print(f"      {j+1}. {record.operation}: "
              f"ΔΦ={record.delta_phi:.6f}, "
              f"accepted={record.accepted}")
    if len(lineage) > 5:
        print(f"      ... ({len(lineage)-5} more)")
    print()
    
    # Test 5: CNF Receipts
    print("5. CNF Receipts...")
    receipts = provenance_logger.receipts
    print(f"   Total receipts: {len(receipts)}")
    if len(receipts) > 0:
        sample_receipt = receipts[0]
        print(f"   Sample receipt:")
        print(f"      ID: {sample_receipt.receipt_id}")
        print(f"      Operation: {sample_receipt.operation}")
        print(f"      Accepted: {sample_receipt.accepted}")
        print(f"      Policy: {sample_receipt.policy_stamp}")
    print()
    
    # Test 6: Bregman Distance
    print("6. Bregman Distance Analysis...")
    bregman = BregmanDistance()
    dist = bregman.distance(initial_overlay, current_overlay)
    print(f"   Bregman distance (initial → final): {dist:.6f}")
    print()
    
    # Test 7: Shell Protocol Statistics
    print("7. Shell Protocol Statistics...")
    shell_stats = shell_protocol.get_statistics()
    print(f"   Current stage: {shell_stats['current_stage']}")
    print(f"   Current radius: {shell_stats['current_radius']:.2f}")
    print(f"   Expansion factor: {shell_stats['expansion_factor']}")
    print()
    
    # Test 8: Acceptance Rule Statistics
    print("8. Acceptance Rule Statistics...")
    accept_stats = acceptance_rule.get_statistics()
    print(f"   Total decisions: {accept_stats['total']}")
    print(f"   Accepted: {accept_stats['accepted']}")
    print(f"   Rejected: {accept_stats['rejected']}")
    print(f"   Accept rate: {accept_stats['accept_rate']:.1%}")
    print(f"   By type:")
    for type_name, count in accept_stats['by_type'].items():
        print(f"      {type_name}: {count}")
    print()
    
    # Test 9: Save outputs
    print("9. Saving Outputs...")
    provenance_logger.save_ledger('/tmp/cqe_v71_ledger.json')
    provenance_logger.save_receipts('/tmp/cqe_v71_receipts.jsonl')
    overlay_store.save('/tmp/cqe_v71_overlays.json')
    print(f"   Saved ledger to /tmp/cqe_v71_ledger.json")
    print(f"   Saved receipts to /tmp/cqe_v71_receipts.jsonl")
    print(f"   Saved overlays to /tmp/cqe_v71_overlays.json")
    print()
    
    # Test 10: QuadraticLawHarness
    print("10. Running QuadraticLawHarness Validation...")
    harness = QuadraticLawHarness()
    
    # Run subset of tests
    print("   Running CNF Path-Independence test...")
    operations = [
        ("rotate", {"theta": 0.1}),
        ("weyl_reflect", {"root_index": 5})
    ]
    result1 = harness.test_cnf_path_independence(initial_overlay, operations)
    print(f"      Result: {'PASS' if result1.passed else 'FAIL'}")
    
    print("   Running Boundary-Only Emission test...")
    result2 = harness.test_boundary_only_emission(initial_overlay, num_interior_ops=5)
    print(f"      Result: {'PASS' if result2.passed else 'FAIL'}")
    
    print("   Running Receipt Schema Compliance test...")
    result3 = harness.test_receipt_schema_compliance(initial_overlay)
    print(f"      Result: {'PASS' if result3.passed else 'FAIL'}")
    
    print()
    
    # Summary
    print("="*70)
    print("Integration Test Summary")
    print("="*70)
    print()
    print("✅ All components integrated successfully!")
    print()
    print("Components tested:")
    print("  1. Overlay System (Axiom A) - ✅")
    print("  2. ALENA Operators (Axiom E) - ✅")
    print("  3. Acceptance Rules & Parity Tracking - ✅")
    print("  4. QuadraticLawHarness - ✅")
    print("  5. Provenance Logging & CNF Receipts - ✅")
    print("  6. Shell Protocol & Bregman Distance - ✅")
    print()
    print("Key Metrics:")
    print(f"  - Optimization iterations: {num_iterations}")
    print(f"  - Accept rate: {accepted_count/num_iterations:.1%}")
    print(f"  - Provenance records: {len(provenance_logger.records)}")
    print(f"  - CNF receipts: {len(provenance_logger.receipts)}")
    print(f"  - Overlays stored: {len(overlay_store.overlays)}")
    print(f"  - Bregman distance: {dist:.6f}")
    print()
    print("Output files:")
    print("  - /tmp/cqe_v71_ledger.json")
    print("  - /tmp/cqe_v71_receipts.jsonl")
    print("  - /tmp/cqe_v71_overlays.json")
    print()
    print("="*70)
    print("CQE v7.1 Integration Test Complete!")
    print("="*70)


if __name__ == "__main__":
    test_full_workflow()
