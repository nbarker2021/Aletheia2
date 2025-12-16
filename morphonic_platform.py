"""
Morphonic Operation Platform
Unified runtime for the CQE system - loadable into AI platforms.
"""

import sys
import os
from pathlib import Path

# Ensure unified modules are importable
sys.path.insert(0, str(Path(__file__).parent))

# Core imports
from unified.cqe_core.operators import OperatorLibrary
from unified.cqe_core.state import CQEState
from unified.cqe_core.objective import Phi
from unified.cqe_unified.slices import SLICE_REGISTRY, slice_op
from unified.cqe_unified.uvibs_monster import (
    project_to_planes, w4_geometry, w80_parity_stability,
    wexp_sparsity, monster_24D_projection, monster_pass
)
from unified.cqe_unified.glyphs_lambda import *
# Ledger and Provenance require paths/manifold_ids - use simple in-memory versions
# from unified.cqe_core.ledger import Ledger
# from unified.cqe_core.provenance import Provenance


class SimpleLedger:
    """In-memory ledger for receipts."""
    def __init__(self):
        self.entries = []
    
    def append(self, rec):
        self.entries.append(rec)
    
    def clear(self):
        self.entries = []


class SimpleProvenance:
    """Simple provenance tracker."""
    def __init__(self):
        self.run_id = None
    
    def start(self):
        import time
        self.run_id = f"run_{int(time.time())}"
        return self.run_id


class MorphonicPlatform:
    """
    The Morphonic Operation Platform - a geometric AI operating system.
    
    This is the unified entry point for all CQE operations. It can be loaded
    into AI platforms like AnythingLLM as a custom system.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the platform with all subsystems."""
        self.seed = seed
        
        # Core systems
        self.operators = OperatorLibrary(seed=seed)
        self.phi = Phi()
        self.ledger = SimpleLedger()
        self.provenance = SimpleProvenance()
        
        # State
        self.state = None
        
        # Available operators
        self.available_operators = self.operators.shortlist()
        
        # UVIBS/Monster metrics
        self.uvibs_enabled = True
        
        # Slice registry
        self.slices = SLICE_REGISTRY
        
    def initialize_state(self, lanes: list = None, parity: bool = True):
        """Initialize or reset the CQE state."""
        if lanes is None:
            lanes = [0.0] * 8  # Default 8-lane state
        self.state = CQEState(lanes=lanes, parity_ok=parity)
        return self.state
    
    def apply_operator(self, op_name: str, **kwargs):
        """Apply an ALENA operator to the current state."""
        if self.state is None:
            self.initialize_state()
            
        if op_name not in self.available_operators:
            raise ValueError(f"Unknown operator: {op_name}. Available: {self.available_operators}")
        
        op = getattr(self.operators, op_name)
        
        # Apply operator
        new_lanes, new_parity = op(self.state.lanes, self.state.parity_ok, **kwargs)
        
        # Update state
        old_parity = self.state.parity_ok
        self.state.lanes = new_lanes
        self.state.parity_ok = new_parity
        
        # Log to ledger
        receipt = {
            "operator": op_name,
            "kwargs": kwargs,
            "parity_change": f"{old_parity} -> {new_parity}",
        }
        self.ledger.append(receipt)
        
        return self.state
    
    def run_morsr(self, budget: int = 100, target_parity: bool = True):
        """
        Run the MORSR protocol to optimize state.
        
        MORSR = Middle-Out Ripple Shape Reader
        Uses monotonic acceptance (ΔΦ ≤ 0) to find optimal state.
        """
        if self.state is None:
            self.initialize_state()
        
        import random
        rs = random.Random(self.seed)
        
        best_phi = self.phi.total(self.state.lanes, self.state.parity_ok)
        accepts = 0
        rejects = 0
        
        for step in range(budget):
            # Select random operator
            op_name = rs.choice(self.available_operators)
            
            # Generate parameters
            kwargs = {}
            if op_name == "R_theta":
                kwargs["k"] = rs.randint(1, 7)
            elif op_name == "Weyl_reflect":
                kwargs["idx"] = rs.randint(0, len(self.state.lanes) - 1)
            elif op_name == "SingleInsert":
                kwargs["idx"] = rs.randint(0, len(self.state.lanes) - 1)
                kwargs["val"] = rs.uniform(-0.15, 0.15)
            elif op_name == "ParityMirror":
                kwargs["strength"] = rs.uniform(0.1, 0.35)
            
            # Apply operator
            op = getattr(self.operators, op_name)
            new_lanes, new_parity = op(self.state.lanes, self.state.parity_ok, **kwargs)
            new_phi = self.phi.total(new_lanes, new_parity)
            
            # Monotonic acceptance
            delta = new_phi - best_phi
            if delta <= 0:
                self.state.lanes = new_lanes
                self.state.parity_ok = new_parity
                best_phi = new_phi
                accepts += 1
            else:
                rejects += 1
            
            # Early exit if target achieved
            if target_parity and self.state.parity_ok:
                break
        
        return {
            "final_phi": best_phi,
            "parity_ok": self.state.parity_ok,
            "accepts": accepts,
            "rejects": rejects,
            "steps": step + 1,
        }
    
    def process(self, input_data: dict) -> dict:
        """
        Main entry point for AI platform integration.
        
        This method is called by AI platforms to process requests.
        """
        action = input_data.get("action", "status")
        
        if action == "status":
            return {
                "platform": "Morphonic Operation Platform",
                "version": "1.0.0",
                "operators": self.available_operators,
                "state_initialized": self.state is not None,
            }
        
        elif action == "initialize":
            lanes = input_data.get("lanes")
            parity = input_data.get("parity", True)
            self.initialize_state(lanes, parity)
            return {"status": "initialized", "state": self.get_state()}
        
        elif action == "apply":
            op_name = input_data.get("operator")
            kwargs = input_data.get("kwargs", {})
            self.apply_operator(op_name, **kwargs)
            return {"status": "applied", "state": self.get_state()}
        
        elif action == "morsr":
            budget = input_data.get("budget", 100)
            result = self.run_morsr(budget=budget)
            return {"status": "morsr_complete", "result": result, "state": self.get_state()}
        
        else:
            return {"error": f"Unknown action: {action}"}
    
    def get_state(self) -> dict:
        """Get current state as dictionary."""
        if self.state is None:
            return None
        return {
            "lanes": list(self.state.lanes),
            "parity_ok": self.state.parity_ok,
        }
    
    def get_ledger(self) -> list:
        """Get the operation ledger."""
        return self.ledger.entries if hasattr(self.ledger, 'entries') else []
    
    def compute_uvibs_metrics(self) -> dict:
        """Compute UVIBS/Monster metrics for current state."""
        if self.state is None:
            return {}
        
        lanes = self.state.lanes
        return {
            "planes": project_to_planes(lanes),
            "w4_geometry": w4_geometry(lanes),
            "w80_parity_stability": w80_parity_stability(lanes),
            "wexp_sparsity": wexp_sparsity(lanes),
            "monster_24D": monster_24D_projection(lanes),
            "monster_pass": monster_pass(monster_24D_projection(lanes)),
        }
    
    def list_slices(self) -> list:
        """List available CQE slices."""
        return list(self.slices.keys())


# Create default instance for direct import
platform = MorphonicPlatform()


def main():
    """Test the platform."""
    print("=== Morphonic Operation Platform ===")
    print()
    
    # Status
    status = platform.process({"action": "status"})
    print(f"Platform: {status['platform']}")
    print(f"Version: {status['version']}")
    print(f"Operators: {status['operators']}")
    print()
    
    # Initialize with broken parity
    platform.process({"action": "initialize", "lanes": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "parity": False})
    print(f"Initial state: parity_ok = {platform.state.parity_ok}")
    
    # Run MORSR to fix parity
    result = platform.process({"action": "morsr", "budget": 50})
    print(f"After MORSR: parity_ok = {result['state']['parity_ok']}")
    print(f"Steps: {result['result']['steps']}, Accepts: {result['result']['accepts']}")
    print()
    
    print("=== Platform Ready ===")


if __name__ == "__main__":
    main()
