# Aletheia2 User Guide

**Version 2.0.0 | December 2025**

## Table of Contents

1. [Introduction](#1-introduction)
2. [Core Philosophy](#2-core-philosophy)
3. [System Architecture](#3-system-architecture)
4. [The Performative Flow](#4-the-performative-flow)
5. [Key Components](#5-key-components)
6. [Critical Rules](#6-critical-rules)
7. [Installation](#7-installation)
8. [Quick Start](#8-quick-start)
9. [Advanced Usage](#9-advanced-usage)
10. [Glossary](#10-glossary)

---

## 1. Introduction

Aletheia2 is a **Morphonic Operation Platform** - a complete AI reasoning system that functions as a deployable assistant (like AnythingLLM). The system uses **geometric embeddings**, **lattice structures** (E8, Leech, 24 Niemeier lattices), and a **constraint-first reasoning approach** to eliminate ambiguity before computation.

The name "Aletheia" comes from the Greek word for "truth" or "disclosure" - the system is designed to reveal truth through geometric reasoning rather than probabilistic guessing.

### What Makes Aletheia2 Different

Traditional AI systems operate on semantics first, then try to constrain outputs. Aletheia2 inverts this:

1. **Geometry First**: All reasoning happens in geometric space (E8, Leech lattices)
2. **Constraints Before Computation**: Ambiguity is eliminated before any computation
3. **Receipts for Everything**: Every operation generates a cryptographic receipt
4. **Non-Increasing Energy**: The system never recomputes known answers (ΔΦ ≤ 0)

---

## 2. Core Philosophy

### The Morphonic Paradigm

The system is built on the principle that **geometry never lies**. By operating in high-dimensional geometric spaces (8D E8, 24D Leech), the system can:

- Represent all possible meanings simultaneously
- Use geometric constraints to eliminate invalid interpretations
- Converge on truth through lattice operations

### Key Principles

| Principle | Description |
|-----------|-------------|
| **Geometry Never Lies** | Semantics removed until final steps, geometry only |
| **Recall Over Recompute** | Embeddings are equivalence classes - never recompute known answers |
| **Receipts for All** | Every action produces receipts (governance requirement) |
| **Non-Increasing Energy** | ΔΦ ≤ 0 - system always moves toward more ordered state |
| **Many Channels** | Multiple valid implementations preserved as parallel options |

---

## 3. System Architecture

### The 5-Layer Architecture

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

### Key Lattice Structures

| Lattice | Dimension | Key Property | Use |
|---------|-----------|--------------|-----|
| **E8** | 8D | 240 roots | Primary embedding space |
| **Leech** | 24D | 196,560 kissing number | High-dimensional storage |
| **24 Niemeier** | 24D | 24 distinct lattices | Persona mapping |

---

## 4. The Performative Flow

This is how the system processes any input:

### Step 1: Disambiguation (4-24 Personas)

The system generates ALL possible meanings of the input using 4-24 personas that map to the 24 Niemeier lattices. Each persona provides a different perspective through Socratic reasoning.

### Step 2: Expand to 8 Views

Each interpretation is expanded to 8 complementary views mapped to E8. These views are cross-compared by witnesses to identify consensus and divergence.

### Step 3: Define Everything

Before any computation:
- **Lambda Commands**: All operations expressed as lambda terms
- **SpeedLight Tasks**: All tasks registered with the caching system
- **GeoTransformer**: ALL rotations computed
- **GeoTokenizer**: Geometric equivalence classes established
- **MonsterMoonshineDB**: ALL embeddings saved in ALL dimensions

### Step 4: SpeedLight Self-Evolution

The system builds its own tools from the codebase:
- **Think Tank**: Generates candidate solutions
- **Assembly Line**: Validates and refines solutions
- **DTT (Decision Tree Traversal)**: Navigates solution space

### Step 5: Reasoning

Unified Noether-Shannon-Landauer reasoning:
- Conservation laws (Noether)
- Information bounds (Shannon)
- Energy costs (Landauer)

**Critical**: Non-increasing energy rule - never recompute known answers.

### Step 6: Solve Definition

Only AFTER constraint accumulation does the system define what "solve" means in this context. The definition emerges from the constraints, not imposed beforehand.

### Step 7: Lattice Building

Build actual lattices from embedding data. The solution is a geometric structure, not a text string.

---

## 5. Key Components

### SpeedLight V2

The idempotent caching system with Merkle-chained ledger:

```python
from aletheia2 import SpeedLightV2

sl = SpeedLightV2(
    mem_bytes=512*1024*1024,  # 512MB cache
    disk_dir="./cache",       # Persistent storage
    ledger_path="./ledger.jsonl"  # Receipt ledger
)

# Every computation is cached and receipted
result, cost, key = sl.compute(
    payload={"input": data},
    scope="reasoning",
    channel=3,  # 3/6/9 channels
    compute_fn=my_function
)
```

### Lambda E8 Builder

Express geometric operations as lambda terms:

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

### GeoTransformer

E8-constrained attention mechanism:

```python
from aletheia2 import GeoTransformer, TransformerConfig

config = TransformerConfig(
    d_model=64,    # Must be multiple of 8
    n_heads=8,     # Must be power of 2
    n_layers=6,
    enforce_e8=True
)

transformer = GeoTransformer(config)
output, lambda_term, metadata = transformer.forward(input_tensor)
```

### Unified Runtime

The main entry point:

```python
from aletheia2 import UnifiedRuntime

runtime = UnifiedRuntime(
    cache_dir="./cache",
    ledger_path="./ledger.jsonl"
)

state = runtime.process([1, 2, 3, 4, 5, 6, 7, 8])

print(state.valid)           # Validation result
print(state.digital_root)    # DR 0-9
print(state.lambda_term)     # Lambda representation
print(state.receipt)         # Operation receipt
```

---

## 6. Critical Rules

These rules MUST be followed for correct operation:

### Rule 1: Every Action Produces Receipts

```python
# CORRECT - receipts generated automatically
state = runtime.process(data)
print(state.receipt.operation_id)

# The receipt contains:
# - operation_id: Unique identifier
# - timestamp: When it happened
# - input_hash: Hash of input
# - output_hash: Hash of output
# - cost_ms: Computation time
# - lambda_term: Lambda representation
```

### Rule 2: Non-Increasing Energy (ΔΦ ≤ 0)

The system enforces that energy (Φ) never increases:

```python
# The conservation enforcer checks all transformations
conservation_result = runtime.conservation.check_transformation(
    input_state, output_state
)
assert conservation_result.delta_phi <= 0
```

### Rule 3: SpeedLight Uses 3 Tools Minimum

SpeedLight ALWAYS uses:
1. **GeoTransformer** - All rotations
2. **GeoTokenizer** - Geometric equivalence classes
3. **MonsterMoonshineDB** - All embeddings in all dimensions

### Rule 4: No Hardcoded Paths

All paths must be relative for deployment:

```python
# WRONG
path = "/home/ubuntu/data/file.json"

# CORRECT
import os
path = os.path.join(os.path.dirname(__file__), "data", "file.json")
```

### Rule 5: Geometry Before Semantics

Semantics are removed until final steps:

```python
# The system operates on vectors, not meanings
e8_state = runtime.e8.project(input_vector)  # Geometry
leech_state = runtime.leech.embed_e8(e8_state)  # More geometry

# Only at the end do we interpret
meaning = interpret_geometric_result(leech_state)  # Semantics last
```

---

## 7. Installation

### From Source

```bash
git clone https://github.com/nbarker2021/Aletheia2.git
cd Aletheia2
pip install -e .
```

### Requirements

- Python 3.9+
- NumPy >= 1.21.0

### Verify Installation

```bash
python -c "from aletheia2 import UnifiedRuntime; print('OK')"
```

---

## 8. Quick Start

### Basic Usage

```python
from aletheia2 import UnifiedRuntime
import numpy as np

# Initialize runtime
runtime = UnifiedRuntime()

# Process input (8D vector for E8)
input_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
state = runtime.process(input_data)

# Check results
print(f"Valid: {state.valid}")
print(f"Digital Root: {state.digital_root.name}")
print(f"Conservation ΔΦ: {state.conservation_phi}")
print(f"Receipt: {state.receipt.operation_id}")

# Get status report
print(runtime.report())
```

### Using SpeedLight Caching

```python
from aletheia2 import SpeedLightV2

sl = SpeedLightV2(mem_bytes=64*1024*1024)

def expensive_computation():
    # Some expensive operation
    return {"result": sum(range(1000000))}

# First call - computes
result1, cost1, key1 = sl.compute(
    {"task": "sum"},
    compute_fn=expensive_computation
)
print(f"First call: {cost1:.4f}s")

# Second call - cached (zero cost)
result2, cost2, key2 = sl.compute(
    {"task": "sum"},
    compute_fn=expensive_computation
)
print(f"Second call: {cost2:.4f}s")  # 0.0000s
```

### Using Lambda E8

```python
from aletheia2 import LambdaE8Builder, LambdaType

builder = LambdaE8Builder()

# Build a geometric operation as lambda term
x = builder.var("x", LambdaType.VECTOR)
y = builder.var("y", LambdaType.VECTOR)

# Compose operations
embedded_x = builder.e8_embed(x)
embedded_y = builder.e8_embed(y)
composed = builder.path_compose(embedded_x, embedded_y)
result = builder.conserve(composed)

# Create abstraction
term = builder.abs("x", builder.abs("y", result))
print(term.to_string())
```

---

## 9. Advanced Usage

### Custom Transformer Configuration

```python
from aletheia2 import GeoTransformer, TransformerConfig

config = TransformerConfig(
    d_model=128,      # Embedding dimension (multiple of 8)
    n_heads=16,       # Attention heads (power of 2)
    d_ff=512,         # Feedforward dimension
    n_layers=12,      # Number of layers
    max_seq_len=256,  # Maximum sequence length
    enforce_e8=True   # E8 constraint enforcement
)

transformer = GeoTransformer(config)
```

### Persistent Caching

```python
from aletheia2 import UnifiedRuntime

runtime = UnifiedRuntime(
    cache_dir="./aletheia_cache",      # Disk cache
    ledger_path="./aletheia_ledger.jsonl",  # Receipt ledger
    mem_bytes=1024*1024*1024  # 1GB memory cache
)

# All operations are now persisted
state = runtime.process(data)

# Restart the runtime - cache persists
runtime2 = UnifiedRuntime(
    cache_dir="./aletheia_cache",
    ledger_path="./aletheia_ledger.jsonl"
)

# This will be a cache hit (zero cost)
state2 = runtime2.process(data)
```

### Accessing Core Layers

```python
from aletheia2 import UnifiedRuntime

runtime = UnifiedRuntime()

# Layer 1: Morphonic Foundation
morphon = runtime.morphon
mglc = runtime.mglc

# Layer 2: Geometric Engine
e8 = runtime.e8
leech = runtime.leech

# Layer 3: Operational
conservation = runtime.conservation
morsr = runtime.morsr

# Layer 4: Governance
gravitational = runtime.gravitational
witness = runtime.seven_witness

# Layer 5: Interface
sdk = runtime.sdk
speedlight = runtime.speedlight
```

---

## 10. Glossary

| Term | Definition |
|------|------------|
| **E8 Lattice** | 8-dimensional exceptional Lie group lattice with 240 roots |
| **Leech Lattice** | 24-dimensional lattice with kissing number 196,560 |
| **Niemeier Lattices** | 24 distinct even unimodular lattices in 24 dimensions |
| **Morphon** | Fundamental unit of the morphonic system |
| **MGLC** | Morphonic Generalized Lambda Calculus |
| **SpeedLight** | Idempotent caching system with receipt generation |
| **Digital Root (DR)** | Single digit (0-9) representing state classification |
| **ΔΦ** | Change in energy/potential - must be ≤ 0 |
| **Seven Witness** | Multi-perspective validation system |
| **Gravitational Layer** | Foundational grounding (DR 0) |
| **MORSR** | Observe-Reflect-Synthesize-Recurse exploration |
| **Conservation** | Energy conservation enforcement |
| **Receipt** | Cryptographic proof of operation |
| **Lambda Term** | Geometric operation expressed in lambda calculus |

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/nbarker2021/Aletheia2/issues
- Documentation: See `/docs` directory

---

*Aletheia2 - Truth through Geometry*
