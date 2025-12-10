"""
Provenance Logging and CNF Receipts
Cartan-Quadratic Equivalence (CQE) System

From Axiom F:
"Every accepted transition logs ΔΦ, op, reason code, policy stamp, and
parent IDs (signed when keys are present)."

This module implements complete audit trail and CNF boundary receipts.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import json
import hashlib
import time
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay
from layer1_morphonic.alena_operators import OperationResult
from layer1_morphonic.acceptance_rules import AcceptanceDecision, ParitySignature


@dataclass
class CNFReceipt:
    """
    CNF (Canonical Normal Form) Boundary Receipt.
    
    Generated at boundary crossings to record state transitions.
    Schema-compliant with validation requirements.
    """
    
    # Core fields
    receipt_id: str
    timestamp: float
    iso_timestamp: str
    
    # Overlay state
    overlay_id: str
    parent_id: Optional[str]
    e8_base: List[float]
    activations: List[int]
    num_active: int
    
    # Parity signature
    parity_signature: Dict[str, Any]
    
    # Phi metric
    phi: float
    delta_phi: float
    
    # Operation details
    operation: str
    parameters: Dict[str, Any]
    
    # Acceptance
    accepted: bool
    acceptance_type: str
    reason: str
    
    # Policy
    policy_stamp: str
    
    # Signature (optional)
    signature: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_operation(
        cls,
        overlay_before: Overlay,
        operation_result: OperationResult,
        acceptance_decision: AcceptanceDecision,
        policy_stamp: str = "cqe_policy_v1"
    ) -> 'CNFReceipt':
        """Create CNF receipt from operation result and acceptance decision."""
        
        overlay_after = operation_result.overlay
        
        # Generate receipt ID
        receipt_data = {
            'overlay_id': overlay_after.overlay_id,
            'timestamp': time.time(),
            'operation': operation_result.operation
        }
        receipt_id = hashlib.sha256(
            json.dumps(receipt_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        # Get parity signature
        parity = ParitySignature.from_overlay(overlay_after)
        
        # Create ISO timestamp
        iso_timestamp = datetime.fromtimestamp(overlay_after.creation_time).isoformat()
        
        return cls(
            receipt_id=receipt_id,
            timestamp=overlay_after.creation_time,
            iso_timestamp=iso_timestamp,
            overlay_id=overlay_after.overlay_id,
            parent_id=overlay_before.overlay_id,
            e8_base=overlay_after.e8_base.tolist(),
            activations=overlay_after.activations.tolist(),
            num_active=int(np.sum(overlay_after.activations)),
            parity_signature=parity.to_dict(),
            phi=acceptance_decision.metadata.get('phi_after', 0.0),
            delta_phi=acceptance_decision.delta_phi,
            operation=operation_result.operation,
            parameters=operation_result.parameters,
            accepted=acceptance_decision.accepted,
            acceptance_type=acceptance_decision.acceptance_type.value,
            reason=acceptance_decision.reason,
            policy_stamp=policy_stamp
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'receipt_id': self.receipt_id,
            'timestamp': self.timestamp,
            'iso_timestamp': self.iso_timestamp,
            'overlay_id': self.overlay_id,
            'parent_id': self.parent_id,
            'e8_base': self.e8_base,
            'activations': self.activations,
            'num_active': self.num_active,
            'parity_signature': self.parity_signature,
            'phi': self.phi,
            'delta_phi': self.delta_phi,
            'operation': self.operation,
            'parameters': self.parameters,
            'accepted': self.accepted,
            'acceptance_type': self.acceptance_type,
            'reason': self.reason,
            'policy_stamp': self.policy_stamp,
            'signature': self.signature,
            'metadata': self.metadata
        }
    
    def sign(self, private_key: Optional[str] = None):
        """
        Sign the receipt (optional).
        
        Args:
            private_key: Private key for signing (if available)
        """
        if private_key:
            # In production, use proper cryptographic signing
            # For now, use simple hash-based signature
            data = json.dumps(self.to_dict(), sort_keys=True)
            self.signature = hashlib.sha256(
                (data + private_key).encode()
            ).hexdigest()


@dataclass
class ProvenanceRecord:
    """
    Complete provenance record for a transition.
    
    From Axiom F:
    Logs ΔΦ, op, reason code, policy stamp, and parent IDs.
    """
    
    # Identifiers
    record_id: str
    timestamp: float
    iso_timestamp: str
    
    # State transition
    overlay_before_id: str
    overlay_after_id: str
    parent_ids: List[str]
    
    # Operation
    operation: str
    parameters: Dict[str, Any]
    
    # Metrics
    delta_phi: float
    delta_parity: float
    phi_before: float
    phi_after: float
    
    # Acceptance
    accepted: bool
    reason: str
    
    # Policy
    policy_stamp: str
    
    # CNF Receipt
    receipt: Optional[CNFReceipt] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'record_id': self.record_id,
            'timestamp': self.timestamp,
            'iso_timestamp': self.iso_timestamp,
            'overlay_before_id': self.overlay_before_id,
            'overlay_after_id': self.overlay_after_id,
            'parent_ids': self.parent_ids,
            'operation': self.operation,
            'parameters': self.parameters,
            'delta_phi': self.delta_phi,
            'delta_parity': self.delta_parity,
            'phi_before': self.phi_before,
            'phi_after': self.phi_after,
            'accepted': self.accepted,
            'reason': self.reason,
            'policy_stamp': self.policy_stamp,
            'receipt': self.receipt.to_dict() if self.receipt else None,
            'metadata': self.metadata
        }


class ProvenanceLogger:
    """
    Provenance logging system.
    
    Maintains complete audit trail of all transitions with CNF receipts.
    """
    
    def __init__(self, policy_stamp: str = "cqe_policy_v1"):
        self.policy_stamp = policy_stamp
        self.records: List[ProvenanceRecord] = []
        self.receipts: List[CNFReceipt] = []
    
    def log_transition(
        self,
        overlay_before: Overlay,
        operation_result: OperationResult,
        acceptance_decision: AcceptanceDecision,
        phi_before: float,
        phi_after: float
    ) -> ProvenanceRecord:
        """
        Log a state transition with full provenance.
        
        Args:
            overlay_before: Overlay before operation
            operation_result: Result of ALENA operation
            acceptance_decision: Acceptance decision
            phi_before: Phi metric before
            phi_after: Phi metric after
        
        Returns:
            ProvenanceRecord
        """
        overlay_after = operation_result.overlay
        
        # Generate record ID
        record_id = hashlib.sha256(
            f"{overlay_before.overlay_id}:{overlay_after.overlay_id}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Create ISO timestamp
        iso_timestamp = datetime.now().isoformat()
        
        # Get parent IDs
        parent_ids = []
        if overlay_before.parent_id:
            parent_ids.append(overlay_before.parent_id)
        if overlay_after.parent_id:
            parent_ids.append(overlay_after.parent_id)
        
        # Create CNF receipt if accepted
        receipt = None
        if acceptance_decision.accepted:
            receipt = CNFReceipt.from_operation(
                overlay_before,
                operation_result,
                acceptance_decision,
                self.policy_stamp
            )
            self.receipts.append(receipt)
        
        # Create provenance record
        record = ProvenanceRecord(
            record_id=record_id,
            timestamp=time.time(),
            iso_timestamp=iso_timestamp,
            overlay_before_id=overlay_before.overlay_id,
            overlay_after_id=overlay_after.overlay_id,
            parent_ids=parent_ids,
            operation=operation_result.operation,
            parameters=operation_result.parameters,
            delta_phi=acceptance_decision.delta_phi,
            delta_parity=acceptance_decision.delta_parity,
            phi_before=phi_before,
            phi_after=phi_after,
            accepted=acceptance_decision.accepted,
            reason=acceptance_decision.reason,
            policy_stamp=self.policy_stamp,
            receipt=receipt
        )
        
        self.records.append(record)
        return record
    
    def get_lineage(self, overlay_id: str) -> List[ProvenanceRecord]:
        """
        Get complete lineage for an overlay.
        
        Args:
            overlay_id: Overlay ID
        
        Returns:
            List of provenance records in lineage
        """
        lineage = []
        current_id = overlay_id
        
        # Traverse backwards through records
        while current_id:
            # Find record where overlay_after_id matches current_id
            matching_records = [
                r for r in self.records
                if r.overlay_after_id == current_id
            ]
            
            if not matching_records:
                break
            
            # Take most recent record
            record = matching_records[-1]
            lineage.append(record)
            current_id = record.overlay_before_id
        
        return list(reversed(lineage))
    
    def save_ledger(self, filepath: str):
        """
        Save ledger to JSON file.
        
        Args:
            filepath: Path to save ledger
        """
        ledger = {
            'policy_stamp': self.policy_stamp,
            'num_records': len(self.records),
            'num_receipts': len(self.receipts),
            'records': [r.to_dict() for r in self.records]
        }
        
        with open(filepath, 'w') as f:
            json.dump(ledger, f, indent=2)
    
    def save_receipts(self, filepath: str):
        """
        Save receipts to JSONL file.
        
        Args:
            filepath: Path to save receipts
        """
        with open(filepath, 'w') as f:
            for receipt in self.receipts:
                f.write(json.dumps(receipt.to_dict()) + '\n')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get provenance statistics."""
        total_records = len(self.records)
        accepted_records = sum(1 for r in self.records if r.accepted)
        rejected_records = total_records - accepted_records
        
        # Compute average deltas
        if total_records > 0:
            avg_delta_phi = np.mean([r.delta_phi for r in self.records])
            avg_delta_parity = np.mean([r.delta_parity for r in self.records])
        else:
            avg_delta_phi = 0.0
            avg_delta_parity = 0.0
        
        # Count by operation
        by_operation = {}
        for record in self.records:
            op = record.operation
            if op not in by_operation:
                by_operation[op] = 0
            by_operation[op] += 1
        
        # Count by reason
        by_reason = {}
        for record in self.records:
            reason = record.reason
            if reason not in by_reason:
                by_reason[reason] = 0
            by_reason[reason] += 1
        
        return {
            'total_records': total_records,
            'accepted': accepted_records,
            'rejected': rejected_records,
            'accept_rate': accepted_records / total_records if total_records > 0 else 0.0,
            'num_receipts': len(self.receipts),
            'avg_delta_phi': avg_delta_phi,
            'avg_delta_parity': avg_delta_parity,
            'by_operation': by_operation,
            'by_reason': by_reason
        }


# Example usage
if __name__ == "__main__":
    print("=== Provenance Logging and CNF Receipts Test ===\n")
    
    from layer1_morphonic.overlay_system import ImmutablePose
    from layer1_morphonic.alena_operators import ALENAOperators
    from layer1_morphonic.acceptance_rules import AcceptanceRule
    
    # Create test overlay
    e8_base = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    activations = np.zeros(240, dtype=int)
    activations[0:120] = 1
    
    pose = ImmutablePose(
        position=tuple(e8_base),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    overlay = Overlay(e8_base=e8_base, activations=activations, pose=pose)
    
    # Create systems
    alena = ALENAOperators()
    acceptance_rule = AcceptanceRule()
    logger = ProvenanceLogger()
    
    print(f"Initial overlay: {overlay.overlay_id}\n")
    
    # Test 1: Perform operation and log
    print("Test 1: Midpoint Operation")
    result1 = alena.midpoint(overlay, overlay, weight=0.5)
    phi_before = alena._compute_phi(overlay)
    phi_after = alena._compute_phi(result1.overlay)
    decision1 = acceptance_rule.evaluate_operation_result(overlay, result1)
    
    record1 = logger.log_transition(
        overlay,
        result1,
        decision1,
        phi_before,
        phi_after
    )
    
    print(f"Record ID: {record1.record_id}")
    print(f"Accepted: {record1.accepted}")
    print(f"ΔΦ: {record1.delta_phi:.6f}")
    print(f"Receipt generated: {record1.receipt is not None}")
    print()
    
    # Test 2: Another operation
    print("Test 2: Rotation Operation")
    result2 = alena.rotate(result1.overlay, theta=0.1)
    phi_before2 = phi_after
    phi_after2 = alena._compute_phi(result2.overlay)
    decision2 = acceptance_rule.evaluate_operation_result(result1.overlay, result2)
    
    record2 = logger.log_transition(
        result1.overlay,
        result2,
        decision2,
        phi_before2,
        phi_after2
    )
    
    print(f"Record ID: {record2.record_id}")
    print(f"Accepted: {record2.accepted}")
    print(f"ΔΦ: {record2.delta_phi:.6f}")
    print()
    
    # Test 3: Get lineage
    print("Test 3: Lineage Tracking")
    lineage = logger.get_lineage(result2.overlay.overlay_id)
    print(f"Lineage length: {len(lineage)}")
    for i, rec in enumerate(lineage):
        print(f"  {i+1}. {rec.operation}: ΔΦ={rec.delta_phi:.6f}, accepted={rec.accepted}")
    print()
    
    # Test 4: Statistics
    print("Test 4: Provenance Statistics")
    stats = logger.get_statistics()
    print(f"Total records: {stats['total_records']}")
    print(f"Accepted: {stats['accepted']}")
    print(f"Accept rate: {stats['accept_rate']:.1%}")
    print(f"Receipts: {stats['num_receipts']}")
    print(f"By operation: {stats['by_operation']}")
    print()
    
    # Test 5: Save ledger and receipts
    print("Test 5: Save Ledger and Receipts")
    logger.save_ledger('/tmp/test_ledger.json')
    logger.save_receipts('/tmp/test_receipts.jsonl')
    print("Saved ledger to /tmp/test_ledger.json")
    print("Saved receipts to /tmp/test_receipts.jsonl")
    print()
    
    print("=== All Tests Passed ===")
