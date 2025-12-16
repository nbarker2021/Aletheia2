"""
Extended CQE Slices - Domain-Specific Processing

Additional slices beyond the core MORSR, SACNUM, SPECTRAL.
Includes mathematical domain slices and governance slices.
"""

import numpy as np
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from spine.kernel import CQEAtom
from spine.speedlight import get_speedlight, receipted
from spine.reasoning import CQESlice, SliceResult


# ============================================================
# MATHEMATICAL DOMAIN SLICES
# ============================================================

class RIEMANNSlice(CQESlice):
    """
    CQE-RIEMANN: Complex Analysis Slice.
    
    Validates Riemann hypothesis related properties.
    Domain: Complex Analysis
    """
    
    name = "riemann"
    description = "Complex Analysis - Riemann zeta and analytic continuation"
    
    def execute(self, atom: CQEAtom, **kwargs) -> SliceResult:
        """Perform Riemann analysis on atom."""
        lanes = atom.lanes
        
        # Compute zeta-like properties
        # Using simplified approximation
        s_real = float(np.mean(lanes[:4]))
        s_imag = float(np.mean(lanes[4:]))
        
        # Check critical strip (0 < Re(s) < 1)
        in_critical_strip = 0 < s_real < 1
        
        # Approximate zeta value (very simplified)
        zeta_approx = sum(1.0 / (n ** (s_real + 1j * s_imag)) for n in range(1, 20))
        zeta_magnitude = abs(zeta_approx)
        
        # Check if near zero (RH connection)
        near_zero = zeta_magnitude < 0.1
        
        metadata = {
            "s_real": s_real,
            "s_imag": s_imag,
            "in_critical_strip": in_critical_strip,
            "zeta_magnitude": zeta_magnitude,
            "near_zero": near_zero
        }
        
        return SliceResult(
            atom=atom,
            slice_name=self.name,
            metadata=metadata,
            success=True,
            message=f"RIEMANN: Re(s)={s_real:.3f}, |ζ(s)|={zeta_magnitude:.4f}"
        )


class GALOISSlice(CQESlice):
    """
    CQE-GALOIS: Algebraic Structures Slice.
    
    Analyzes group-theoretic properties.
    Domain: Abstract Algebra
    """
    
    name = "galois"
    description = "Algebraic Structures - Group theory and field extensions"
    
    def execute(self, atom: CQEAtom, **kwargs) -> SliceResult:
        """Perform Galois analysis on atom."""
        lanes = atom.lanes
        
        # Compute permutation signature
        perm = np.argsort(lanes)
        inversions = 0
        for i in range(len(perm)):
            for j in range(i + 1, len(perm)):
                if perm[i] > perm[j]:
                    inversions += 1
        
        # Parity of permutation
        perm_parity = "even" if inversions % 2 == 0 else "odd"
        
        # Cycle structure
        visited = [False] * len(perm)
        cycles = []
        for i in range(len(perm)):
            if not visited[i]:
                cycle = []
                j = i
                while not visited[j]:
                    visited[j] = True
                    cycle.append(j)
                    j = perm[j]
                cycles.append(len(cycle))
        
        # Group order estimate (factorial of distinct cycle lengths)
        from math import factorial
        group_order = factorial(len(set(cycles)))
        
        metadata = {
            "inversions": inversions,
            "perm_parity": perm_parity,
            "cycle_structure": cycles,
            "group_order_estimate": group_order
        }
        
        return SliceResult(
            atom=atom,
            slice_name=self.name,
            metadata=metadata,
            success=True,
            message=f"GALOIS: {perm_parity} permutation, cycles={cycles}"
        )


class HASSESlice(CQESlice):
    """
    CQE-HASSE: Number Theory Slice.
    
    Analyzes Hasse principle and local-global properties.
    Domain: Algebraic Number Theory
    """
    
    name = "hasse"
    description = "Number Theory - Local-global principle and Hasse invariants"
    
    def execute(self, atom: CQEAtom, **kwargs) -> SliceResult:
        """Perform Hasse analysis on atom."""
        lanes = atom.lanes
        
        # Compute local invariants at various primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19]
        local_invariants = {}
        
        for p in primes:
            # Simplified p-adic valuation
            val = sum(int(abs(x * 1000)) % p for x in lanes)
            local_invariants[p] = val % p
        
        # Check Hasse principle (product of local invariants)
        hasse_product = 1
        for v in local_invariants.values():
            hasse_product *= (v + 1)
        hasse_product %= 1000
        
        # Global obstruction check
        global_obstruction = hasse_product == 0
        
        metadata = {
            "local_invariants": local_invariants,
            "hasse_product": hasse_product,
            "global_obstruction": global_obstruction
        }
        
        return SliceResult(
            atom=atom,
            slice_name=self.name,
            metadata=metadata,
            success=True,
            message=f"HASSE: product={hasse_product}, obstruction={global_obstruction}"
        )


class LEGENDRESlice(CQESlice):
    """
    CQE-LEGENDRE: Quadratic Residues Slice.
    
    Analyzes quadratic reciprocity and Legendre symbols.
    Domain: Elementary Number Theory
    """
    
    name = "legendre"
    description = "Quadratic Residues - Legendre symbols and reciprocity"
    
    def execute(self, atom: CQEAtom, **kwargs) -> SliceResult:
        """Perform Legendre analysis on atom."""
        lanes = atom.lanes
        
        # Convert to integers for modular arithmetic
        ints = [int(abs(x * 1000)) for x in lanes]
        
        # Compute Legendre symbols for first few primes
        def legendre(a, p):
            if a % p == 0:
                return 0
            return 1 if pow(a, (p - 1) // 2, p) == 1 else -1
        
        primes = [3, 5, 7, 11, 13]
        symbols = {}
        for p in primes:
            for i, a in enumerate(ints):
                if a > 0:
                    symbols[f"({a % p}/{p})"] = legendre(a % p, p)
        
        # Quadratic character sum
        char_sum = sum(symbols.values())
        
        metadata = {
            "legendre_symbols": symbols,
            "character_sum": char_sum,
            "qr_count": sum(1 for v in symbols.values() if v == 1)
        }
        
        return SliceResult(
            atom=atom,
            slice_name=self.name,
            metadata=metadata,
            success=True,
            message=f"LEGENDRE: char_sum={char_sum}"
        )


# ============================================================
# GOVERNANCE SLICES
# ============================================================

class DWMCharterSlice(CQESlice):
    """
    DWM Charter Initialization Slice.
    
    Initializes the overlay registry and hyperperm oracle.
    """
    
    name = "dwm_charter"
    description = "DWM Charter - Initialize overlay and oracle registries"
    
    def __init__(self):
        self.overlay_registry = {}
        self.hyperperm_oracle = {"items": {}}
    
    def execute(self, atom: CQEAtom, **kwargs) -> SliceResult:
        """Initialize DWM charter."""
        import time
        
        # Initialize overlay registry if empty
        if not self.overlay_registry:
            self.overlay_registry = {
                "A": {"glyph": "A", "category": "TRIANGLE", "ops": ["×", "÷", "#2"], "status": "PENDING"},
                "B": {"glyph": "B", "category": "MULTI_STROKE", "ops": ["%", "÷", "~"], "status": "PENDING"},
                "C": {"glyph": "C", "category": "LOOP", "ops": ["÷", "~"], "status": "PENDING"}
            }
        
        metadata = {
            "started_at": time.time(),
            "overlays": len(self.overlay_registry),
            "oracle_items": len(self.hyperperm_oracle.get("items", {}))
        }
        
        return SliceResult(
            atom=atom,
            slice_name=self.name,
            metadata=metadata,
            success=True,
            message=f"DWM Charter: {len(self.overlay_registry)} overlays initialized"
        )


class MS2LabelSlice(CQESlice):
    """
    MS2 Label and Hyperperm Slice.
    
    Labels data and manages the hyperperm oracle.
    """
    
    name = "ms2_label"
    description = "MS2 - Label data and manage hyperperm oracle"
    
    def __init__(self):
        self.oracle = {"items": {}, "sigs": set(), "channels": set()}
    
    def execute(self, atom: CQEAtom, **kwargs) -> SliceResult:
        """Process labeling."""
        # Generate signature from atom
        sig = hashlib.sha256(json.dumps(atom.to_dict(), sort_keys=True).encode()).hexdigest()[:16]
        
        # Update oracle
        self.oracle["sigs"].add(sig)
        channel = kwargs.get("channel", "default")
        self.oracle["channels"].add(channel)
        
        # Check lock threshold
        locked = len(self.oracle["sigs"]) >= 8 and len(self.oracle["channels"]) >= 4
        
        metadata = {
            "sig": sig,
            "channel": channel,
            "total_sigs": len(self.oracle["sigs"]),
            "total_channels": len(self.oracle["channels"]),
            "locked": locked
        }
        
        return SliceResult(
            atom=atom,
            slice_name=self.name,
            metadata=metadata,
            success=True,
            message=f"MS2: sig={sig[:8]}, locked={locked}"
        )


class MS3GovernanceSlice(CQESlice):
    """
    MS3 Governance Manager Slice.
    
    Manages witnesses, quorum, and seals.
    """
    
    name = "ms3_governance"
    description = "MS3 - Governance manager with witnesses and seals"
    
    def __init__(self):
        self.witnesses = []
        self.seals = []
    
    def execute(self, atom: CQEAtom, **kwargs) -> SliceResult:
        """Process governance."""
        import time
        
        # Create witnesses from atom state
        w_parity = {"id": "wit:parity", "vote": "YES" if atom.parity_ok else "NO", "weight": 1.0}
        w_phi = {"id": "wit:phi", "vote": "YES" if atom.phi() < 1.0 else "NO", "weight": 1.0}
        w_dr = {"id": "wit:dr", "vote": "YES" if atom.digital_root() in [0, 9] else "NO", "weight": 0.5}
        
        witnesses = [w_parity, w_phi, w_dr]
        
        # Tally votes
        yes_votes = sum(w["weight"] for w in witnesses if w["vote"] == "YES")
        no_votes = sum(w["weight"] for w in witnesses if w["vote"] == "NO")
        
        # Decision
        decision = "ACCEPT" if yes_votes > no_votes else "REVIEW"
        
        # Create seal
        seal_data = json.dumps({"atom": atom.to_dict(), "witnesses": witnesses}, sort_keys=True)
        seal_hash = hashlib.sha256(seal_data.encode()).hexdigest()[:16]
        
        metadata = {
            "witnesses": witnesses,
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "decision": decision,
            "seal_hash": seal_hash,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        return SliceResult(
            atom=atom,
            slice_name=self.name,
            metadata=metadata,
            success=True,
            message=f"MS3: decision={decision}, seal={seal_hash}"
        )


# ============================================================
# SLICE REGISTRY
# ============================================================

def get_all_slices() -> Dict[str, CQESlice]:
    """Get all available slices."""
    return {
        # Mathematical domain slices
        "riemann": RIEMANNSlice(),
        "galois": GALOISSlice(),
        "hasse": HASSESlice(),
        "legendre": LEGENDRESlice(),
        # Governance slices
        "dwm_charter": DWMCharterSlice(),
        "ms2_label": MS2LabelSlice(),
        "ms3_governance": MS3GovernanceSlice(),
    }


def register_all_slices(reasoning_engine):
    """Register all extended slices with the reasoning engine."""
    for name, slice_instance in get_all_slices().items():
        reasoning_engine.register_slice(slice_instance)
