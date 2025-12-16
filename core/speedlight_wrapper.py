#!/usr/bin/env python3
"""
SpeedLight Wrapper - Mandatory Receipt Layer
=============================================

This module wraps all CQE operations to ensure receipt generation is mandatory.
No operation passes through the system without being logged and documented.

The SpeedLight system is the audit and data ingress layer that:
1. Generates receipts for all operations
2. Performs transforms and tokenization
3. Handles all embedding generation
4. Anchors receipts with SHA256 hashes at regular intervals
"""

import json
import hashlib
import time
import os
from typing import Any, Dict, Optional, Callable
from functools import wraps

class SpeedLightReceipts:
    """
    Receipt generation system with SHA256-anchored audit trail.
    
    All operations in the CQE system MUST pass through this layer.
    """
    
    def __init__(self, path: str = "./data/speedlight_ledger.jsonl", anchor_period: float = 3600.0):
        """
        Initialize the SpeedLight receipt system.
        
        Args:
            path: Path to the JSONL ledger file
            anchor_period: Seconds between SHA256 anchor points (default 1 hour)
        """
        self.path = path
        self.anchor_period = anchor_period
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        self._roll = []
        self._t0 = time.time()
        self._operation_count = 0
    
    def write(self, kind: str, what: Any, gov: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write a receipt for an operation.
        
        Args:
            kind: Type of operation (e.g., "transform", "embed", "process")
            what: The data or operation being recorded
            gov: Governance metadata (layer, constraints, etc.)
        
        Returns:
            The receipt record
        """
        rec = {
            "ts": time.time(),
            "kind": kind,
            "WHAT": what,
            "GOV": gov,
            "seq": self._operation_count
        }
        line = json.dumps(rec, sort_keys=True)
        self._f.write(line + "\n")
        self._f.flush()
        self._roll.append(line.encode())
        self._operation_count += 1
        
        # Check if we need to anchor
        if time.time() - self._t0 >= self.anchor_period:
            self.anchor()
        
        return rec
    
    def anchor(self) -> Optional[str]:
        """
        Create a SHA256 anchor point for all receipts since last anchor.
        
        Returns:
            The anchor hash, or None if no receipts to anchor
        """
        if not self._roll:
            return None
        
        h = hashlib.sha256()
        for ln in self._roll:
            h.update(ln)
        
        anchor_hash = h.hexdigest()
        anchor_rec = {
            "ts": time.time(),
            "kind": "anchor",
            "root": anchor_hash,
            "count": len(self._roll)
        }
        self._f.write(json.dumps(anchor_rec) + "\n")
        self._f.flush()
        self._roll.clear()
        self._t0 = time.time()
        
        return anchor_hash
    
    def close(self):
        """Close the receipt file, anchoring any pending receipts."""
        self.anchor()
        self._f.close()


# Global SpeedLight instance
_speedlight: Optional[SpeedLightReceipts] = None


def get_speedlight(path: str = "./data/speedlight_ledger.jsonl") -> SpeedLightReceipts:
    """Get or create the global SpeedLight instance."""
    global _speedlight
    if _speedlight is None:
        _speedlight = SpeedLightReceipts(path)
    return _speedlight


def requires_receipt(kind: str, layer: str = "unknown"):
    """
    Decorator that enforces receipt generation for any function.
    
    Args:
        kind: The type of operation being performed
        layer: The CQE layer this operation belongs to
    
    Usage:
        @requires_receipt("transform", layer="L2")
        def my_transform_function(data):
            return transformed_data
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            sl = get_speedlight()
            
            # Record the operation start
            start_time = time.time()
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Generate receipt
            gov = {
                "layer": layer,
                "function": func.__name__,
                "module": func.__module__,
                "duration_ms": (time.time() - start_time) * 1000
            }
            
            # Summarize what was processed
            what = {
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
                "result_type": type(result).__name__
            }
            
            sl.write(kind, what, gov)
            
            return result
        return wrapper
    return decorator


class SpeedLightContext:
    """
    Context manager for SpeedLight operations.
    
    Usage:
        with SpeedLightContext("batch_process", layer="L3") as ctx:
            # All operations here are tracked
            result = process_data(data)
            ctx.log("intermediate", {"step": 1, "status": "complete"})
    """
    
    def __init__(self, operation: str, layer: str = "unknown", metadata: Optional[Dict] = None):
        self.operation = operation
        self.layer = layer
        self.metadata = metadata or {}
        self.sl = get_speedlight()
        self.start_time = None
        self.logs = []
    
    def __enter__(self):
        self.start_time = time.time()
        self.sl.write(f"{self.operation}_start", self.metadata, {"layer": self.layer})
        return self
    
    def log(self, sub_operation: str, data: Any):
        """Log an intermediate step within the context."""
        self.logs.append({
            "sub_op": sub_operation,
            "data": data,
            "ts": time.time()
        })
        self.sl.write(f"{self.operation}_{sub_operation}", data, {"layer": self.layer})
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        gov = {
            "layer": self.layer,
            "duration_ms": duration * 1000,
            "steps": len(self.logs),
            "success": exc_type is None
        }
        if exc_type:
            gov["error"] = str(exc_val)
        
        self.sl.write(f"{self.operation}_end", {"logs": len(self.logs)}, gov)
        return False  # Don't suppress exceptions


# Convenience functions for common operations
def log_transform(data: Any, transform_type: str, layer: str = "L2") -> Dict:
    """Log a transformation operation."""
    return get_speedlight().write("transform", {
        "type": transform_type,
        "data_type": type(data).__name__
    }, {"layer": layer})


def log_embedding(vector_dim: int, source: str, layer: str = "L2") -> Dict:
    """Log an embedding generation."""
    return get_speedlight().write("embedding", {
        "dimension": vector_dim,
        "source": source
    }, {"layer": layer})


def log_governance_check(check_type: str, passed: bool, layer: str = "L4") -> Dict:
    """Log a governance check."""
    return get_speedlight().write("governance", {
        "check": check_type,
        "passed": passed
    }, {"layer": layer})


def log_lattice_operation(lattice_type: str, operation: str, layer: str = "L2") -> Dict:
    """Log a lattice operation."""
    return get_speedlight().write("lattice", {
        "lattice": lattice_type,
        "operation": operation
    }, {"layer": layer})
