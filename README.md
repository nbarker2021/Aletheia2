# Aletheia2 - Morphonic Operation Platform

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/nbarker2021/Aletheia2)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

**A complete AI reasoning system using geometric embeddings, lattice structures, and constraint-first reasoning to eliminate ambiguity before computation.**

Aletheia2 is a morphonic-native geometric operating system that functions as a deployable AI assistant (like AnythingLLM). It synthesizes approximately two years of CQE research (~39 archives, ~900MB) into a unified runtime with geometric embeddings in E8, Leech, and 24 Niemeier lattice spaces.

---

## What's New in v2.0.0

- **Unified Runtime with SpeedLight V2**: Merkle-chained ledger caching with receipt generation
- **Lambda E8 Calculus**: Express geometric operations as lambda terms
- **GeoTransformer**: E8-constrained attention mechanism
- **Complete Morphonic Integration**: Full morphonic_cqe_unified package
- **Production Packaging**: Ready for deployment with setup.py

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Geometry-First Reasoning** | All operations happen in geometric space before semantics |
| **Receipt Generation** | Every action produces cryptographic receipts (governance requirement) |
| **Non-Increasing Energy** | ΔΦ ≤ 0 - the system never recomputes known answers |
| **SpeedLight V2 Caching** | Merkle-chained ledger with LRU cache for idempotent operations |
| **Lambda E8 Calculus** | Express geometric operations as lambda terms |
| **E8-Constrained Transformer** | Attention mechanism with geometric constraints |

---

## Quick Start

### Installation

```bash
git clone https://github.com/nbarker2021/Aletheia2.git
cd Aletheia2
pip install -e .
```

### Basic Usage

```python
from aletheia2 import UnifiedRuntime
import numpy as np

# Initialize runtime
runtime = UnifiedRuntime()

# Process input (8D vector for E8 embedding)
state = runtime.process(np.array([1, 2, 3, 4, 5, 6, 7, 8]))

print(f"Valid: {state.valid}")
print(f"Digital Root: {state.digital_root.name}")
print(f"Receipt: {state.receipt.operation_id}")
print(runtime.report())
```

### Using SpeedLight Caching

```python
from aletheia2 import SpeedLightV2

sl = SpeedLightV2(mem_bytes=64*1024*1024)

def expensive_computation():
    return {"result": sum(range(1000000))}

# First call computes, second call is cached (zero cost)
result, cost, key = sl.compute({"task": "sum"}, compute_fn=expensive_computation)
print(f"Cost: {cost}s")  # Second call: 0.0s
```

### Using Lambda E8 Calculus

```python
from aletheia2 import LambdaE8Builder, LambdaType

builder = LambdaE8Builder()
x = builder.var("x", LambdaType.VECTOR)
embedded = builder.e8_embed(x)
projected = builder.e8_project(embedded, 24)
conserved = builder.conserve(projected)
term = builder.abs("x", conserved, LambdaType.VECTOR)

print(term.to_string())
# (λ x: vector. (conserve (e8_project (e8_embed x) 24)))
```

### Using GeoTransformer

```python
from aletheia2 import GeoTransformer, TransformerConfig
import numpy as np

config = TransformerConfig(d_model=64, n_heads=8, n_layers=2, enforce_e8=True)
transformer = GeoTransformer(config)

x = np.random.randn(1, 8, 64)
output, lambda_term, metadata = transformer.forward(x)
print(f"Output shape: {output.shape}")
print(f"Lambda: {lambda_term.to_string()}")
```

---

## Architecture

### The 5-Layer System

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: Interface & Applications                          │
│  - Native SDK, SpeedLight V2, Lambda E8                     │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Governance & Validation                           │
│  - Gravitational Layer (DR 0), Seven Witness                │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Operational Systems                               │
│  - Conservation (ΔΦ ≤ 0), MORSR Explorer                    │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Core Geometric Engine                             │
│  - E8 Lattice (240 roots), Leech Lattice (196,560 min)      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Morphonic Foundation                              │
│  - Universal Morphon (M₀), MGLC (8 reduction rules)         │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| **Unified Runtime** | Main entry point with SpeedLight V2 | `unified_runtime.py` |
| **GeoTransformer** | E8-constrained attention | `geo_transformer.py` |
| **SpeedLight V2** | Merkle-chained caching | `morphonic_cqe_unified/sidecar/` |
| **Lambda E8** | Geometric lambda calculus | `morphonic_cqe_unified/experimental/` |
| **E8 Lattice** | 8D lattice with 240 roots | `layer2_geometric.py` |
| **Leech Lattice** | 24D lattice with 196,560 minimal vectors | `layer2_geometric.py` |

### Lattice Constants

| Lattice | Dimension | Key Property |
|---------|-----------|--------------|
| **E8** | 8D | 240 roots |
| **Leech** | 24D | 196,560 kissing number |
| **24 Niemeier** | 24D | 24 distinct lattices |

---

## The Performative Flow

How Aletheia2 processes any input:

1. **Disambiguation**: Generate ALL possible meanings using 4-24 personas mapped to Niemeier lattices
2. **Expand to 8 Views**: Map to E8 complementary views, cross-compare via witnesses
3. **Define Everything**: Lambda commands, SpeedLight tasks, GeoTransformer (ALL rotations), GeoTokenizer, MonsterMoonshineDB
4. **SpeedLight Self-Evolution**: Think Tank generates, Assembly Line validates, DTT navigates
5. **Reasoning**: Noether-Shannon-Landauer unified - conservation laws, information bounds, energy costs (ΔΦ ≤ 0)
6. **Solve Definition**: Define "solve" AFTER constraints accumulate - emerges from constraints
7. **Lattice Building**: Build actual lattices from embeddings - solution is geometric structure

---

## Critical Rules

These rules MUST be followed for correct operation:

1. **Every action produces receipts** - governance requirement
2. **Non-increasing energy (ΔΦ ≤ 0)** - never recompute known answers
3. **SpeedLight uses 3 tools minimum** - GeoTransformer, GeoTokenizer, MonsterMoonshineDB
4. **No hardcoded paths** - use relative paths only
5. **Geometry before semantics** - operate geometrically first

---

## Project Structure

```
Aletheia2/
├── unified_runtime.py          # Main entry point with SpeedLight V2
├── geo_transformer.py          # E8-constrained transformer
├── runtime.py                  # Original CQE runtime
├── layer1_morphonic.py         # L1: Morphon, MGLC
├── layer2_geometric.py         # L2: E8, Leech lattices
├── layer3_operational.py       # L3: Conservation, MORSR
├── layer4_governance.py        # L4: Gravitational, Witness
├── layer5_interface.py         # L5: SDK
├── morphonic_cqe_unified/      # Morphonic package
│   ├── experimental/           # Lambda E8, GeoTransformer
│   │   ├── lambda_e8_calculus.py
│   │   └── geometric_transformer_standalone.py
│   ├── sidecar/                # SpeedLight V2
│   │   └── speedlight_sidecar_plus.py
│   ├── core/                   # Math, governance
│   └── assistant/              # Deployable assistant
├── integration/                # Extracted systems (119 files)
├── aletheia2/                  # Package exports
├── USER_GUIDE.md               # Complete documentation
├── HANDOFF_NEXT_SESSION.md     # Session continuity
└── setup.py                    # Package configuration
```

---

## Testing

```bash
# Quick test
python3.11 unified_runtime.py

# GeoTransformer test
python3.11 geo_transformer.py

# Full integration test
python3.11 -c "
from aletheia2 import UnifiedRuntime, GeoTransformer
import numpy as np

runtime = UnifiedRuntime()
state = runtime.process(np.array([1,2,3,4,5,6,7,8]))
print(f'Valid: {state.valid}')
print(f'Receipt: {state.receipt.operation_id}')
print(runtime.report())
"
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [USER_GUIDE.md](USER_GUIDE.md) | Complete system explanation |
| [HANDOFF_NEXT_SESSION.md](HANDOFF_NEXT_SESSION.md) | Session continuity guide |
| [QUICKSTART.md](QUICKSTART.md) | Quick start guide |
| [BETA_STATUS.md](BETA_STATUS.md) | Development progress |

---

## Requirements

- Python 3.9+
- NumPy >= 1.21.0

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **v2.0.0** | Dec 16, 2025 | Morphonic integration, SpeedLight V2, Lambda E8, GeoTransformer |
| **v1.0-beta** | Dec 5, 2025 | 24 Niemeier lattices, morphonic seed generator, Weyl navigation |
| **v1.0-alpha** | Dec 5, 2025 | Initial five-layer architecture |

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work synthesizes approximately two years of CQE research and development, representing contributions from multiple sessions and implementations. The unified runtime honors the original vision while addressing critical gaps and providing a production-ready foundation for future work.

---

**Author**: Manus AI  
**Date**: December 16, 2025  
**Status**: v2.0.0 (Production Ready)

*Aletheia2 - Truth through Geometry*
