"""
Ledger - Transaction Ledger with Provenance Tracking

Persistent storage for all CQE transactions with full provenance chain.
Integrates with SpeedLight for receipt generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json
import time
import hashlib
import os


def sha256_json(obj: Any) -> str:
    """Generate SHA256 hash of JSON-serializable object."""
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


@dataclass
class Provenance:
    """Tracks the origin and lineage of data."""
    manifold_id: str
    started_ts: float = field(default_factory=time.time)
    run_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def start(self) -> str:
        """Initialize provenance and generate run ID."""
        base = {
            "manifold": self.manifold_id,
            "started_ts": self.started_ts,
        }
        self.run_id = sha256_json(base)[:12]
        return self.run_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifold_id": self.manifold_id,
            "started_ts": self.started_ts,
            "run_id": self.run_id,
            "meta": self.meta
        }


@dataclass
class TransactionRecord:
    """A single transaction in the ledger."""
    tx_id: str
    timestamp: float
    operation: str
    input_hash: str
    output_hash: str
    delta_phi: float
    parity_ok: bool
    provenance: Provenance
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "delta_phi": self.delta_phi,
            "parity_ok": self.parity_ok,
            "provenance": self.provenance.to_dict(),
            "metadata": self.metadata
        }


@dataclass
class IntegrityPanel:
    """Tracks integrity metrics for validation."""
    accepts: int = 0
    rejects: int = 0
    plateau_ticks: int = 0
    parity_fixes: int = 0
    parity_breaks: int = 0
    phi_series: List[float] = field(default_factory=list)
    
    def record_accept(self, phi: float):
        self.accepts += 1
        self.phi_series.append(phi)
        self.plateau_ticks = 0
    
    def record_reject(self, phi: float):
        self.rejects += 1
        self.phi_series.append(phi)
        self.plateau_ticks += 1
    
    def record_parity_fix(self):
        self.parity_fixes += 1
    
    def record_parity_break(self):
        self.parity_breaks += 1
    
    def to_dict(self) -> Dict[str, Any]:
        import statistics
        return {
            "accepts": self.accepts,
            "rejects": self.rejects,
            "accept_rate": self.accepts / max(1, self.accepts + self.rejects),
            "plateau_ticks": self.plateau_ticks,
            "parity_fixes": self.parity_fixes,
            "parity_breaks": self.parity_breaks,
            "phi_mean": statistics.mean(self.phi_series) if self.phi_series else None,
            "phi_last": self.phi_series[-1] if self.phi_series else None,
            "phi_min": min(self.phi_series) if self.phi_series else None,
        }


class Ledger:
    """
    Persistent transaction ledger with provenance tracking.
    
    Stores all CQE transactions to disk with full audit trail.
    """
    
    def __init__(self, path_dir: str = None):
        from .config import get_config
        self.path_dir = path_dir or str(get_config().ledger_path)
        self.file = None
        self.count = 0
        self.transactions: List[TransactionRecord] = []
        self.integrity = IntegrityPanel()
        self.current_provenance: Optional[Provenance] = None
    
    def open(self, manifold_id: str) -> str:
        """Open ledger for a specific manifold."""
        os.makedirs(os.path.join(self.path_dir, manifold_id), exist_ok=True)
        p = os.path.join(self.path_dir, manifold_id, "receipts.jsonl")
        self.file = open(p, "a", encoding="utf-8")
        
        # Initialize provenance
        self.current_provenance = Provenance(manifold_id=manifold_id)
        run_id = self.current_provenance.start()
        return run_id
    
    def append(self, record: TransactionRecord):
        """Append a transaction to the ledger."""
        if not self.file:
            raise RuntimeError("Ledger not opened")
        
        self.transactions.append(record)
        self.file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self.file.flush()
        self.count += 1
        
        # Update integrity panel
        if record.delta_phi <= 0:
            self.integrity.record_accept(record.delta_phi)
        else:
            self.integrity.record_reject(record.delta_phi)
        
        if record.parity_ok:
            self.integrity.record_parity_fix()
        else:
            self.integrity.record_parity_break()
    
    def create_transaction(
        self,
        operation: str,
        input_data: Any,
        output_data: Any,
        old_phi: float,
        new_phi: float,
        parity_ok: bool = True,
        metadata: Optional[Dict] = None
    ) -> TransactionRecord:
        """Create and append a new transaction."""
        tx_id = sha256_json({
            "ts": time.time(),
            "op": operation,
            "in": str(input_data)[:100]
        })[:16]
        
        record = TransactionRecord(
            tx_id=tx_id,
            timestamp=time.time(),
            operation=operation,
            input_hash=sha256_json(input_data)[:16],
            output_hash=sha256_json(output_data)[:16],
            delta_phi=new_phi - old_phi,
            parity_ok=parity_ok,
            provenance=self.current_provenance or Provenance("unknown"),
            metadata=metadata or {}
        )
        
        self.append(record)
        return record
    
    def close(self):
        """Close the ledger file."""
        if self.file:
            self.file.close()
            self.file = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get ledger summary."""
        return {
            "transaction_count": self.count,
            "integrity": self.integrity.to_dict(),
            "provenance": self.current_provenance.to_dict() if self.current_provenance else None
        }
    
    def verify_chain(self) -> bool:
        """Verify all transactions have valid delta_phi."""
        for tx in self.transactions:
            if tx.delta_phi > 0:
                return False
        return True


# Global instance
_global_ledger: Optional[Ledger] = None


def get_ledger(path_dir: str = None) -> Ledger:
    """Get the global ledger instance."""
    global _global_ledger
    if _global_ledger is None:
        _global_ledger = Ledger(path_dir)
    return _global_ledger
