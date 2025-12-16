#!/usr/bin/env python3
"""
Morphonic Operation Platform - Unified Runtime
===============================================

This is the main entry point for the Morphonic Operation Platform.
It integrates all subsystems into a cohesive, functional system.

Architecture:
- Layer 1 (Morphonic): Atomic operations, Lambda calculus, ALENA operators
- Layer 2 (Geometric): E8/Niemeier/Leech lattices, GeoTransformer, GeoTokenizer
- Layer 3 (Operational): MORSR, Conservation laws, Phi metric
- Layer 4 (Governance): Policy enforcement, Sacred geometry governance
- Layer 5 (Interface): SpeedLight sidecar, API endpoints, CLI

All operations are tracked via SpeedLight receipt generation.
"""

import sys
import os
import json
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Add core to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules
from core.speedlight_wrapper import (
    SpeedLightReceipts,
    SpeedLightContext,
    get_speedlight,
    requires_receipt,
    log_transform,
    log_embedding,
    log_governance_check,
    log_lattice_operation
)
from core.geo_pipeline import GeoPipeline, GeoTokenizer, GeoTransformer, E8Projector
from core.lattice_engine import LatticeEngine, E8Lattice, LeechLattice, get_lattice_engine


class Layer(Enum):
    """CQE Layer enumeration."""
    L1_MORPHONIC = "L1"
    L2_GEOMETRIC = "L2"
    L3_OPERATIONAL = "L3"
    L4_GOVERNANCE = "L4"
    L5_INTERFACE = "L5"


@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    success: bool
    data: Any
    layer: Layer
    receipts: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class MorphonicRuntime:
    """
    Morphonic Operation Platform Runtime.
    
    This is the central orchestrator that coordinates all subsystems
    and ensures proper data flow through the layer stack.
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the Morphonic Runtime.
        
        Args:
            data_dir: Directory for data storage and receipts
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize SpeedLight (mandatory)
        self.speedlight = get_speedlight(os.path.join(data_dir, "speedlight_ledger.jsonl"))
        
        # Initialize Layer 2 systems
        self.geo_pipeline = GeoPipeline()
        self.lattice_engine = get_lattice_engine()
        
        # Track initialization
        self._log_init()
    
    def _log_init(self):
        """Log runtime initialization."""
        self.speedlight.write("runtime_init", {
            "version": "4.0",
            "data_dir": self.data_dir,
            "systems": ["speedlight", "geo_pipeline", "lattice_engine"]
        }, {"layer": "L5"})
    
    @requires_receipt("process", layer="L5")
    def process(self, input_data: Union[str, Dict, List], operation: str = "auto") -> ProcessingResult:
        """
        Process input data through the Morphonic platform.
        
        Args:
            input_data: Input to process (text, dict, or list)
            operation: Type of operation ("auto", "embed", "transform", "validate")
        
        Returns:
            ProcessingResult with output data and receipts
        """
        with SpeedLightContext("process", layer="L5", metadata={"operation": operation}) as ctx:
            try:
                # Determine operation type
                if operation == "auto":
                    if isinstance(input_data, str):
                        operation = "embed"
                    elif isinstance(input_data, (list, dict)):
                        operation = "transform"
                
                ctx.log("operation_selected", {"type": operation})
                
                # Route to appropriate handler
                if operation == "embed":
                    result = self._embed(input_data, ctx)
                elif operation == "transform":
                    result = self._transform(input_data, ctx)
                elif operation == "validate":
                    result = self._validate(input_data, ctx)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                return ProcessingResult(
                    success=True,
                    data=result,
                    layer=Layer.L5_INTERFACE,
                    metadata={"operation": operation}
                )
                
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    data=None,
                    layer=Layer.L5_INTERFACE,
                    errors=[str(e)],
                    metadata={"operation": operation}
                )
    
    def _embed(self, text: str, ctx: SpeedLightContext) -> Dict[str, Any]:
        """Generate embeddings for text input."""
        ctx.log("embed_start", {"text_len": len(text)})
        
        # Process through geometric pipeline
        result = self.geo_pipeline.process(text)
        
        # Project final embedding through E8
        import numpy as np
        final_emb = np.array(result["final_embedding"])
        e8_projected = self.lattice_engine.embed_to_e8(final_emb)
        
        result["e8_embedding"] = e8_projected.tolist()
        
        ctx.log("embed_complete", {"embedding_dim": len(result["final_embedding"])})
        
        return result
    
    def _transform(self, data: Union[Dict, List], ctx: SpeedLightContext) -> Dict[str, Any]:
        """Transform structured data through the geometric engine."""
        import numpy as np
        
        ctx.log("transform_start", {"data_type": type(data).__name__})
        
        # Convert to array
        if isinstance(data, dict):
            arr = np.array(list(data.values()), dtype=float)
        else:
            arr = np.array(data, dtype=float)
        
        # Apply E8 projection
        projected = self.lattice_engine.embed_to_e8(arr)
        
        ctx.log("transform_complete", {"shape": list(projected.shape)})
        
        return {
            "original": arr.tolist(),
            "projected": projected.tolist(),
            "lattice": "e8"
        }
    
    def _validate(self, data: Any, ctx: SpeedLightContext) -> Dict[str, Any]:
        """Validate data against geometric constraints."""
        import numpy as np
        
        ctx.log("validate_start", {"data_type": type(data).__name__})
        
        # Convert to array
        if isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=float)
        else:
            arr = np.array([data], dtype=float)
        
        # Validate against E8
        validation = self.lattice_engine.validate_geometric_constraint(arr, "e8")
        
        ctx.log("validate_complete", {"valid": validation["valid"]})
        
        return validation
    
    def get_status(self) -> Dict[str, Any]:
        """Get current runtime status."""
        return {
            "version": "4.0",
            "data_dir": self.data_dir,
            "lattice_info": self.lattice_engine.get_lattice_info(),
            "speedlight_active": True
        }
    
    def close(self):
        """Shutdown the runtime gracefully."""
        self.speedlight.write("runtime_shutdown", {
            "timestamp": time.time()
        }, {"layer": "L5"})
        self.speedlight.close()


# Global runtime instance
_runtime: Optional[MorphonicRuntime] = None


def get_runtime(data_dir: str = "./data") -> MorphonicRuntime:
    """Get or create the global runtime instance."""
    global _runtime
    if _runtime is None:
        _runtime = MorphonicRuntime(data_dir)
    return _runtime


def process(input_data: Union[str, Dict, List], operation: str = "auto") -> ProcessingResult:
    """
    Quick function to process data through the Morphonic platform.
    
    Args:
        input_data: Input to process
        operation: Type of operation
    
    Returns:
        ProcessingResult
    """
    runtime = get_runtime()
    return runtime.process(input_data, operation)


def embed(text: str) -> Dict[str, Any]:
    """Quick function to generate embeddings."""
    result = process(text, "embed")
    return result.data if result.success else {"error": result.errors}


def validate(data: Any) -> Dict[str, Any]:
    """Quick function to validate data."""
    result = process(data, "validate")
    return result.data if result.success else {"error": result.errors}


# CLI interface
def main():
    """Command-line interface for the Morphonic Runtime."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Morphonic Operation Platform")
    parser.add_argument("command", choices=["status", "embed", "validate", "process"],
                       help="Command to execute")
    parser.add_argument("--input", "-i", help="Input data (text or JSON)")
    parser.add_argument("--file", "-f", help="Input file path")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--data-dir", "-d", default="./data", help="Data directory")
    
    args = parser.parse_args()
    
    runtime = get_runtime(args.data_dir)
    
    if args.command == "status":
        result = runtime.get_status()
        print(json.dumps(result, indent=2))
    
    elif args.command in ["embed", "validate", "process"]:
        # Get input
        if args.file:
            with open(args.file, "r") as f:
                input_data = f.read()
                try:
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    pass  # Keep as string
        elif args.input:
            input_data = args.input
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                pass
        else:
            print("Error: --input or --file required")
            return
        
        # Process
        result = runtime.process(input_data, args.command)
        
        # Output
        output = {
            "success": result.success,
            "data": result.data,
            "errors": result.errors
        }
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
        else:
            print(json.dumps(output, indent=2))
    
    runtime.close()


if __name__ == "__main__":
    main()
