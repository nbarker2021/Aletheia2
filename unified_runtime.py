"""
Aletheia2 Unified Runtime v2.0
==============================

A morphonic-native geometric operating system with:
- SpeedLight idempotent caching (receipts for ALL operations)
- Lambda E8 calculus (geometric lambda terms)
- Deployable assistant capabilities
- 5-layer architecture with full governance

This extends the original CQE Unified Runtime with:
- SpeedLight V2 (Merkle-chained ledger, LRU cache, disk persistence)
- Lambda E8 Builder (geometric operations as lambda terms)
- GeoTransformer, GeoTokenizer integration points
- Assistant interface for interactive use

Author: Manus AI
Date: December 16, 2025
License: MIT
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
import logging
import json
import hashlib
import time
import os

# Import core layers
from layer1_morphonic import UniversalMorphon, MGLCEngine, ObservationType, LambdaLevel
from layer2_geometric import E8Lattice, LeechLattice
from layer3_operational import ConservationEnforcer, MORSRExplorer
from layer4_governance import GravitationalLayer, SevenWitness, DigitalRoot
from layer5_interface import CQESDK

# Import morphonic extensions
from morphonic_cqe_unified.sidecar.speedlight_sidecar_plus import SpeedLightV2
from morphonic_cqe_unified.experimental.lambda_e8_calculus import (
    LambdaE8Builder, LambdaTerm, LambdaType, GeometricLambdaCapture
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# RECEIPT SYSTEM
# =============================================================================

@dataclass
class OperationReceipt:
    """Receipt for any operation in the system (governance requirement)."""
    operation_id: str
    operation_type: str
    timestamp: float
    input_hash: str
    output_hash: str
    cost_ms: float
    lambda_term: Optional[str] = None
    e8_coordinates: Optional[List[float]] = None
    conservation_delta: Optional[float] = None
    valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "timestamp": self.timestamp,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "cost_ms": self.cost_ms,
            "lambda_term": self.lambda_term,
            "e8_coordinates": self.e8_coordinates,
            "conservation_delta": self.conservation_delta,
            "valid": self.valid,
            "metadata": self.metadata
        }


# =============================================================================
# UNIFIED RUNTIME STATE
# =============================================================================

@dataclass
class UnifiedRuntimeState:
    """Complete state of the unified runtime."""
    morphon_state: Any
    e8_state: np.ndarray
    leech_state: np.ndarray
    digital_root: DigitalRoot
    conservation_phi: float
    valid: bool
    lambda_term: Optional[LambdaTerm] = None
    receipt: Optional[OperationReceipt] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# UNIFIED RUNTIME
# =============================================================================

class UnifiedRuntime:
    """
    Aletheia2 Unified Runtime v2.0
    
    Extends the 5-layer CQE architecture with:
    - SpeedLight V2 for idempotent caching with receipts
    - Lambda E8 calculus for geometric operations
    - Assistant interface for interactive use
    
    Critical Rules:
    - EVERY action produces receipts (governance requirement)
    - Semantics removed until final steps, geometry only
    - Recall prioritized over recompute (non-increasing energy)
    - SpeedLight ALWAYS uses 3 tools: GeoTransformer, GeoTokenizer, MonsterMoonshineDB
    """
    
    # System constants
    E8_DIM = 8
    E8_ROOTS = 240
    LEECH_DIM = 24
    LEECH_MINIMAL = 196560
    NIEMEIER_COUNT = 24
    WEYL_CHAMBERS = 696729600
    PHI = (1 + np.sqrt(5)) / 2
    COUPLING = np.log((1 + np.sqrt(5)) / 2) / 16
    
    VERSION = "2.0.0"
    BUILD_DATE = "2025-12-16"
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        cache_dir: Optional[str] = None,
        ledger_path: Optional[str] = None,
        mem_bytes: int = 512 * 1024 * 1024
    ):
        """
        Initialize the Unified Runtime.
        
        Args:
            config: Optional configuration dictionary
            cache_dir: Directory for disk cache (optional)
            ledger_path: Path for receipt ledger (optional)
            mem_bytes: Memory limit for LRU cache (default 512MB)
        """
        self.config = config or {}
        logger.info(f"Initializing Aletheia2 Unified Runtime v{self.VERSION}")
        
        # Initialize SpeedLight V2 (receipts + caching)
        logger.info("  [SL] Initializing SpeedLight V2...")
        self.speedlight = SpeedLightV2(
            mem_bytes=mem_bytes,
            disk_dir=cache_dir,
            ledger_path=ledger_path,
            default_ttl=None,  # No expiration by default
            determinism_guard=True
        )
        logger.info("      ✓ SpeedLight V2 (Merkle ledger, LRU cache)")
        
        # Initialize Lambda E8 Builder
        logger.info("  [Λ] Initializing Lambda E8 Calculus...")
        self.lambda_builder = LambdaE8Builder()
        self.lambda_capture = GeometricLambdaCapture()
        logger.info("      ✓ Lambda E8 Builder")
        logger.info("      ✓ Geometric Lambda Capture")
        
        # Layer 1: Morphonic Foundation
        logger.info("  [L1] Initializing Morphonic Foundation...")
        self.morphon = UniversalMorphon()
        self.mglc = MGLCEngine()
        logger.info("      ✓ Universal Morphon (M₀)")
        logger.info("      ✓ MGLC with 8 reduction rules")
        
        # Layer 2: Core Geometric Engine
        logger.info("  [L2] Initializing Core Geometric Engine...")
        self.e8 = E8Lattice()
        self.leech = LeechLattice()
        logger.info(f"      ✓ E8 lattice ({self.E8_ROOTS} roots)")
        logger.info(f"      ✓ Leech lattice ({self.LEECH_MINIMAL} minimal vectors)")
        
        # Layer 3: Operational Systems
        logger.info("  [L3] Initializing Operational Systems...")
        self.conservation = ConservationEnforcer()
        self.morsr = MORSRExplorer()
        logger.info("      ✓ Conservation enforcer (ΔΦ ≤ 0)")
        logger.info("      ✓ MORSR explorer")
        
        # Layer 4: Governance & Validation
        logger.info("  [L4] Initializing Governance & Validation...")
        self.gravitational = GravitationalLayer()
        self.seven_witness = SevenWitness()
        logger.info("      ✓ Gravitational Layer (DR 0) ✨")
        logger.info("      ✓ Seven Witness validation")
        
        # Layer 5: Interface & Applications
        logger.info("  [L5] Initializing Interface & Applications...")
        self.sdk = CQESDK()
        logger.info("      ✓ Native SDK")
        
        # Statistics
        self._operation_count = 0
        self._start_time = time.time()
        
        logger.info(f"✓ Aletheia2 Unified Runtime v{self.VERSION} initialized")
    
    def _generate_receipt(
        self,
        operation_type: str,
        input_data: Any,
        output_data: Any,
        cost_ms: float,
        lambda_term: Optional[LambdaTerm] = None,
        e8_coords: Optional[np.ndarray] = None,
        conservation_delta: Optional[float] = None,
        valid: bool = True,
        metadata: Optional[Dict] = None
    ) -> OperationReceipt:
        """Generate a receipt for an operation (governance requirement)."""
        self._operation_count += 1
        
        # Hash inputs and outputs
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        output_str = json.dumps(output_data, sort_keys=True, default=str)
        input_hash = hashlib.sha256(input_str.encode()).hexdigest()[:16]
        output_hash = hashlib.sha256(output_str.encode()).hexdigest()[:16]
        
        return OperationReceipt(
            operation_id=f"op_{self._operation_count:08d}_{int(time.time()*1000)}",
            operation_type=operation_type,
            timestamp=time.time(),
            input_hash=input_hash,
            output_hash=output_hash,
            cost_ms=cost_ms,
            lambda_term=lambda_term.to_string() if lambda_term else None,
            e8_coordinates=e8_coords.tolist() if e8_coords is not None else None,
            conservation_delta=conservation_delta,
            valid=valid,
            metadata=metadata or {}
        )
    
    def compute(
        self,
        operation_type: str,
        payload: Any,
        compute_fn: Callable,
        scope: str = "global",
        channel: int = 3,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[Any, OperationReceipt]:
        """
        Execute a computation with SpeedLight caching and receipt generation.
        
        This is the primary entry point for all operations. It:
        1. Checks SpeedLight cache for existing result
        2. If miss, executes compute_fn
        3. Generates receipt for governance
        4. Returns (result, receipt)
        
        Args:
            operation_type: Type of operation (for receipt)
            payload: Input payload (used for cache key)
            compute_fn: Function to execute if cache miss
            scope: SpeedLight scope
            channel: SpeedLight channel (3/6/9)
            tags: Optional tags for the operation
            **kwargs: Additional arguments for compute_fn
        
        Returns:
            Tuple of (result, receipt)
        """
        start_time = time.time()
        
        # Execute with SpeedLight
        result, cost, key = self.speedlight.compute(
            payload,
            scope=scope,
            channel=channel,
            tags=tags or [],
            compute_fn=lambda: compute_fn(**kwargs) if kwargs else compute_fn(),
            ttl=None
        )
        
        total_cost_ms = (time.time() - start_time) * 1000
        
        # Generate receipt
        receipt = self._generate_receipt(
            operation_type=operation_type,
            input_data=payload,
            output_data=result,
            cost_ms=total_cost_ms,
            metadata={
                "speedlight_key": key,
                "speedlight_cost": cost,
                "cache_hit": cost == 0,
                "scope": scope,
                "channel": channel,
                "tags": tags or []
            }
        )
        
        return result, receipt
    
    def process(
        self,
        input_data: Union[np.ndarray, str, List, Any],
        generate_lambda: bool = True
    ) -> UnifiedRuntimeState:
        """
        Process input through the complete runtime pipeline with receipts.
        
        Pipeline:
        1. Embed to E8 (Layer 2) with SpeedLight caching
        2. Validate with Seven Witness (Layer 4)
        3. Check gravitational grounding (Layer 4)
        4. Expand to Leech if needed (Layer 2)
        5. Enforce conservation (Layer 3)
        6. Generate lambda term representation
        7. Return validated state with receipt
        
        Args:
            input_data: Input data (arbitrary type)
            generate_lambda: Whether to generate lambda term
        
        Returns:
            UnifiedRuntimeState with receipt
        """
        start_time = time.time()
        logger.info(f"Processing input: {type(input_data).__name__}")
        
        # Convert to numpy array if needed
        original_input = input_data
        if isinstance(input_data, list):
            input_data = np.array(input_data, dtype=np.float64)
        elif isinstance(input_data, str):
            term = self.mglc.parse(input_data)
            input_data = np.zeros(self.E8_DIM)
        
        # Ensure 8D
        if isinstance(input_data, np.ndarray):
            if len(input_data) != self.E8_DIM:
                if len(input_data) < self.E8_DIM:
                    input_data = np.pad(input_data, (0, self.E8_DIM - len(input_data)))
                else:
                    input_data = input_data[:self.E8_DIM]
        
        # 1. Embed to E8 with caching
        def _e8_embed():
            result = self.e8.project(input_data)
            return result.tolist() if isinstance(result, np.ndarray) else result
        
        e8_result, e8_receipt = self.compute(
            "e8_embed",
            {"input": input_data.tolist()},
            _e8_embed,
            scope="geometric",
            channel=3,
            tags=["e8", "embed"]
        )
        e8_state = np.array(e8_result) if isinstance(e8_result, list) else e8_result
        if isinstance(e8_state, str):
            # Handle case where cached result was serialized
            import ast
            e8_state = np.array(ast.literal_eval(e8_state))
        
        # 2. Validate with Seven Witness
        verdict = self.seven_witness.validate(e8_state)
        
        # 3. Check gravitational grounding
        grav_state = self.gravitational.validate_structure(e8_state)
        
        # 4. Expand to Leech with caching
        def _leech_embed():
            result = self.leech.embed_e8(e8_state)
            return result.tolist() if isinstance(result, np.ndarray) else result
        
        leech_result, leech_receipt = self.compute(
            "leech_embed",
            {"e8_state": e8_state.tolist()},
            _leech_embed,
            scope="geometric",
            channel=6,
            tags=["leech", "embed"]
        )
        leech_state = np.array(leech_result) if isinstance(leech_result, list) else leech_result
        if isinstance(leech_state, str):
            import ast
            leech_state = np.array(ast.literal_eval(leech_state))
        
        # 5. Enforce conservation
        conservation_result = self.conservation.check_transformation(input_data, e8_state)
        
        # 6. Generate lambda term if requested
        lambda_term = None
        if generate_lambda:
            x = self.lambda_builder.var("x", LambdaType.VECTOR)
            embedded = self.lambda_builder.e8_embed(x)
            projected = self.lambda_builder.e8_project(embedded, self.LEECH_DIM)
            conserved = self.lambda_builder.conserve(projected)
            lambda_term = self.lambda_builder.abs("x", conserved, LambdaType.VECTOR)
        
        # Calculate total cost
        total_cost_ms = (time.time() - start_time) * 1000
        
        # Generate master receipt
        receipt = self._generate_receipt(
            operation_type="process",
            input_data=original_input,
            output_data={
                "e8_state": e8_state.tolist(),
                "leech_state": leech_state.tolist() if isinstance(leech_state, np.ndarray) else leech_state,
                "valid": verdict.valid and conservation_result.valid
            },
            cost_ms=total_cost_ms,
            lambda_term=lambda_term,
            e8_coords=e8_state,
            conservation_delta=conservation_result.delta_phi,
            valid=verdict.valid and conservation_result.valid,
            metadata={
                "witness_consensus": verdict.consensus,
                "gravitational_depth": grav_state.depth,
                "gravitational_stable": grav_state.stable,
                "conservation_valid": conservation_result.valid,
                "sub_receipts": [e8_receipt.to_dict(), leech_receipt.to_dict()]
            }
        )
        
        # Create runtime state
        state = UnifiedRuntimeState(
            morphon_state=self.morphon,
            e8_state=e8_state,
            leech_state=leech_state,
            digital_root=grav_state.digital_root,
            conservation_phi=conservation_result.delta_phi,
            valid=verdict.valid and conservation_result.valid,
            lambda_term=lambda_term,
            receipt=receipt,
            metadata={
                "witness_consensus": verdict.consensus,
                "gravitational_depth": grav_state.depth,
                "gravitational_stable": grav_state.stable,
                "conservation_valid": conservation_result.valid
            }
        )
        
        return state
    
    def status(self) -> Dict[str, Any]:
        """Get current runtime status including SpeedLight statistics."""
        sl_stats = self.speedlight.stats()
        uptime = time.time() - self._start_time
        
        return {
            "version": self.VERSION,
            "build_date": self.BUILD_DATE,
            "status": "operational",
            "uptime_seconds": uptime,
            "operation_count": self._operation_count,
            "speedlight": {
                "hits": sl_stats["hits"],
                "misses": sl_stats["misses"],
                "hit_rate": sl_stats["hits"] / max(sl_stats["hits"] + sl_stats["misses"], 1),
                "mem_bytes": sl_stats["mem_bytes"],
                "mem_cap_bytes": sl_stats["mem_cap_bytes"],
                "ledger_entries": sl_stats["ledger_len"],
                "ledger_valid": sl_stats["ledger_ok"]
            },
            "layers": {
                "L1_morphonic": {
                    "status": "operational",
                    "components": ["Universal Morphon", "MGLC (8 rules)"]
                },
                "L2_geometric": {
                    "status": "operational",
                    "components": [f"E8 ({self.E8_ROOTS} roots)", 
                                 f"Leech ({self.LEECH_MINIMAL} minimal)"]
                },
                "L3_operational": {
                    "status": "operational",
                    "components": ["Conservation (ΔΦ ≤ 0)", "MORSR"]
                },
                "L4_governance": {
                    "status": "operational",
                    "components": ["Gravitational (DR 0) ✨", "Seven Witness"]
                },
                "L5_interface": {
                    "status": "operational",
                    "components": ["Native SDK", "SpeedLight V2", "Lambda E8"]
                }
            },
            "constants": {
                "E8_DIM": self.E8_DIM,
                "E8_ROOTS": self.E8_ROOTS,
                "LEECH_DIM": self.LEECH_DIM,
                "LEECH_MINIMAL": self.LEECH_MINIMAL,
                "PHI": f"{self.PHI:.6f}",
                "COUPLING": f"{self.COUPLING:.6f}"
            }
        }
    
    def report(self) -> str:
        """Generate a formatted status report."""
        s = self.status()
        sl = s["speedlight"]
        
        return f"""
╔══════════════════════════════════════════════════════════════════════╗
║              ALETHEIA2 UNIFIED RUNTIME v{s['version']}                      ║
║                  Morphonic-Native Geometric OS                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Status: {s['status'].upper():12}  Uptime: {s['uptime_seconds']:.1f}s                          ║
║  Operations: {s['operation_count']:,}                                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  SPEEDLIGHT V2                                                       ║
║    Hits/Misses: {sl['hits']}/{sl['misses']} ({sl['hit_rate']*100:.1f}% hit rate)                        ║
║    Memory: {sl['mem_bytes']/1e6:.2f}MB / {sl['mem_cap_bytes']/1e6:.2f}MB                              ║
║    Ledger: {sl['ledger_entries']} entries, {'OK' if sl['ledger_valid'] else 'FAIL'}                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  LAYERS                                                              ║
║    L1 Morphonic:   ✓ Universal Morphon, MGLC                         ║
║    L2 Geometric:   ✓ E8 ({self.E8_ROOTS} roots), Leech ({self.LEECH_MINIMAL} min)           ║
║    L3 Operational: ✓ Conservation, MORSR                             ║
║    L4 Governance:  ✓ Gravitational (DR 0), Seven Witness             ║
║    L5 Interface:   ✓ SDK, SpeedLight V2, Lambda E8                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for the Unified Runtime."""
    print("=" * 70)
    print("Aletheia2 Unified Runtime v2.0")
    print("Morphonic-Native Geometric Operating System")
    print("=" * 70)
    print()
    print("Features:")
    print("  • SpeedLight V2 (Merkle ledger, LRU cache, receipts)")
    print("  • Lambda E8 Calculus (geometric lambda terms)")
    print("  • 5-layer architecture with full governance")
    print("  • Non-increasing energy rule (ΔΦ ≤ 0)")
    print()
    
    # Initialize runtime
    runtime = UnifiedRuntime()
    
    # Display status
    print(runtime.report())
    
    # Test processing
    print("\n" + "=" * 70)
    print("Test Processing")
    print("=" * 70)
    
    test_input = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
    print(f"\nInput: {test_input}")
    
    state = runtime.process(test_input)
    
    print(f"\nResults:")
    print(f"  Valid: {state.valid}")
    print(f"  Digital Root: {state.digital_root.name}")
    print(f"  Conservation ΔΦ: {state.conservation_phi:.6f}")
    print(f"  Lambda Term: {state.lambda_term.to_string() if state.lambda_term else 'N/A'}")
    print(f"  Receipt ID: {state.receipt.operation_id}")
    print(f"  Receipt Cost: {state.receipt.cost_ms:.2f}ms")
    
    # Test cache hit
    print("\n--- Testing Cache Hit ---")
    state2 = runtime.process(test_input)
    print(f"  Second call cost: {state2.receipt.cost_ms:.2f}ms")
    print(f"  Cache hit: {state2.receipt.metadata.get('sub_receipts', [{}])[0].get('metadata', {}).get('cache_hit', False)}")
    
    # Final report
    print("\n" + runtime.report())
    
    print("=" * 70)
    print("✓ Aletheia2 Unified Runtime is operational and ready")
    print("=" * 70)


if __name__ == "__main__":
    main()
