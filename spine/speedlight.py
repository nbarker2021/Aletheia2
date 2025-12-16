"""
SpeedLight - Mandatory Receipt Generation System

NOTHING passes through the system without a receipt.
This is the non-negotiable audit trail for all operations.
"""

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable, Tuple
from functools import wraps


def sha256(obj: Any) -> str:
    """Generate SHA256 hash of any JSON-serializable object."""
    data = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class Receipt:
    """A single operation receipt."""
    timestamp: float
    operation: str
    input_hash: str
    output_hash: str
    delta_phi: float
    parity_ok: bool
    provenance: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def is_valid(self) -> bool:
        """Check if receipt represents a valid operation."""
        return self.delta_phi <= 0  # Monotonic improvement required


class SpeedLight:
    """
    The SpeedLight receipt generation system.
    
    Every operation MUST be wrapped by SpeedLight to generate receipts.
    Operations without receipts are invalid and will be rejected.
    """
    
    def __init__(self, ledger_path: Optional[str] = None):
        self.ledger: List[Receipt] = []
        self.ledger_path = ledger_path
        self._file = None
        self.session_id = sha256({"start": time.time()})
        
    def open(self):
        """Open the ledger file for writing."""
        if self.ledger_path:
            self._file = open(self.ledger_path, 'a', encoding='utf-8')
    
    def close(self):
        """Close the ledger file."""
        if self._file:
            self._file.close()
            self._file = None
    
    def emit(self, receipt: Receipt):
        """Write a receipt to the ledger."""
        self.ledger.append(receipt)
        if self._file:
            self._file.write(json.dumps(receipt.to_dict()) + '\n')
            self._file.flush()
    
    def create_receipt(
        self,
        operation: str,
        input_data: Any,
        output_data: Any,
        old_phi: float = 0.0,
        new_phi: float = 0.0,
        parity_ok: bool = True,
        metadata: Optional[Dict] = None
    ) -> Receipt:
        """Create a new receipt for an operation."""
        receipt = Receipt(
            timestamp=time.time(),
            operation=operation,
            input_hash=sha256(input_data),
            output_hash=sha256(output_data),
            delta_phi=new_phi - old_phi,
            parity_ok=parity_ok,
            provenance=self.session_id,
            metadata=metadata or {}
        )
        self.emit(receipt)
        return receipt
    
    def wrap(self, operation_name: str, phi_func: Optional[Callable] = None):
        """
        Decorator to wrap any operation with receipt generation.
        
        Usage:
            @speedlight.wrap("my_operation")
            def my_operation(input_data):
                return process(input_data)
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Capture input
                input_data = {"args": args, "kwargs": kwargs}
                old_phi = 0.0
                
                # Calculate old phi if function provided
                if phi_func and args:
                    try:
                        old_phi = phi_func(args[0])
                    except:
                        pass
                
                # Execute operation
                result = func(*args, **kwargs)
                
                # Calculate new phi
                new_phi = 0.0
                if phi_func and result is not None:
                    try:
                        new_phi = phi_func(result)
                    except:
                        pass
                
                # Determine parity
                parity_ok = True
                if hasattr(result, 'parity_ok'):
                    parity_ok = result.parity_ok
                elif isinstance(result, dict) and 'parity_ok' in result:
                    parity_ok = result['parity_ok']
                
                # Create receipt
                self.create_receipt(
                    operation=operation_name,
                    input_data=input_data,
                    output_data=result,
                    old_phi=old_phi,
                    new_phi=new_phi,
                    parity_ok=parity_ok
                )
                
                return result
            return wrapper
        return decorator
    
    def verify_chain(self) -> bool:
        """Verify the entire receipt chain is valid."""
        for receipt in self.ledger:
            if not receipt.is_valid():
                return False
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current ledger."""
        if not self.ledger:
            return {"count": 0, "valid": True}
        
        return {
            "count": len(self.ledger),
            "valid": self.verify_chain(),
            "total_delta_phi": sum(r.delta_phi for r in self.ledger),
            "parity_violations": sum(1 for r in self.ledger if not r.parity_ok),
            "session_id": self.session_id,
        }


# Global instance for convenience
_global_speedlight: Optional[SpeedLight] = None


def get_speedlight() -> SpeedLight:
    """Get the global SpeedLight instance."""
    global _global_speedlight
    if _global_speedlight is None:
        _global_speedlight = SpeedLight()
    return _global_speedlight


def receipted(operation_name: str):
    """Convenience decorator using global SpeedLight."""
    return get_speedlight().wrap(operation_name)


class MerkleTree:
    """
    Merkle tree for receipt chain verification.
    
    Provides cryptographic proof of receipt chain integrity.
    """
    
    def __init__(self):
        self.leaves: List[str] = []
        self.root: Optional[str] = None
    
    def add_receipt(self, receipt: Receipt):
        """Add a receipt to the tree."""
        leaf = sha256(receipt.to_dict())
        self.leaves.append(leaf)
        self._rebuild()
    
    def _rebuild(self):
        """Rebuild the Merkle tree."""
        if not self.leaves:
            self.root = None
            return
        
        level = self.leaves[:]
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = level[i] + level[i + 1]
                else:
                    combined = level[i] + level[i]
                next_level.append(sha256(combined))
            level = next_level
        
        self.root = level[0] if level else None
    
    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """Get Merkle proof for a leaf at given index."""
        if index >= len(self.leaves):
            return []
        
        proof = []
        level = self.leaves[:]
        idx = index
        
        while len(level) > 1:
            sibling_idx = idx ^ 1  # XOR to get sibling
            if sibling_idx < len(level):
                direction = 'L' if idx % 2 == 1 else 'R'
                proof.append((direction, level[sibling_idx]))
            
            # Move up
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = level[i] + level[i + 1]
                else:
                    combined = level[i] + level[i]
                next_level.append(sha256(combined))
            level = next_level
            idx = idx // 2
        
        return proof
    
    def verify_proof(self, leaf: str, proof: List[Tuple[str, str]]) -> bool:
        """Verify a Merkle proof."""
        current = leaf
        for direction, sibling in proof:
            if direction == 'L':
                current = sha256(sibling + current)
            else:
                current = sha256(current + sibling)
        return current == self.root


# Add Merkle tree to SpeedLight
SpeedLight.merkle = None

def _enhanced_emit(self, receipt: Receipt):
    """Enhanced emit with Merkle tree."""
    self.ledger.append(receipt)
    
    # Update Merkle tree
    if self.merkle is None:
        self.merkle = MerkleTree()
    self.merkle.add_receipt(receipt)
    
    if self._file:
        self._file.write(json.dumps(receipt.to_dict()) + '\n')
        self._file.flush()

SpeedLight.emit = _enhanced_emit

def get_merkle_root(self) -> Optional[str]:
    """Get the current Merkle root."""
    if self.merkle is None:
        return None
    return self.merkle.root

SpeedLight.get_merkle_root = get_merkle_root
