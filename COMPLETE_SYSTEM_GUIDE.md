# CQE Unified Runtime: Complete System Integration Guide

**Version**: 2.0 (Complete Integration)  
**For CQE Runtime**: v9.0+  
**Author**: Manus AI  
**Date**: December 13, 2024

---

## Table of Contents

1. [Introduction: The Modular Lego System](#introduction)
2. [The System Spine: Core Architecture](#system-spine)
3. [Complete Component Map](#component-map)
4. [End-to-End Solve Workflow](#end-to-end-workflow)
5. [Component Integration Patterns](#integration-patterns)
6. [System-by-System Guide](#system-guide)
7. [Advanced Integration Examples](#advanced-examples)
8. [Troubleshooting Integration Issues](#troubleshooting)

---

<a name="introduction"></a>
## Part 1: Introduction - The Modular Lego System

### Philosophy

The CQE Unified Runtime is not a monolithic application. It is a **modular system of geometric components** that can be assembled like Lego blocks. Each component has a well-defined interface and can be used independently or in combination with others.

**Key Principle**: *As long as you respect the spine (the core geometric foundation), you can mix and match components in any order to solve your problem.*

### The Three Levels of Integration

| Level | Description | Who Uses It |
|-------|-------------|-------------|
| **Level 1: Direct API** | Use individual components directly via Python imports | Developers, researchers |
| **Level 2: Orchestrated Workflows** | Combine components using orchestration patterns | AI agents, power users |
| **Level 3: High-Level Systems** | Use complete systems like Aletheia or Speedlight | End users, applications |

This guide covers all three levels and shows how they interconnect.

---

<a name="system-spine"></a>
## Part 2: The System Spine - Core Architecture

### The Spine Concept

The "spine" is the **invariant core** that all components must respect. It defines the fundamental data structures, operations, and laws that govern the entire system.

```
┌─────────────────────────────────────────────────────────────┐
│                      THE CQE SPINE                          │
├─────────────────────────────────────────────────────────────┤
│  1. E₈ Lattice (240 roots, Weyl group)                     │
│  2. Overlay (state representation)                          │
│  3. Φ Metric (4-component quality measure)                  │
│  4. ALENA Operators (Rθ, WeylReflect, Midpoint, Parity)    │
│  5. Acceptance Rules (Φ-decrease, parity, plateau)         │
│  6. Provenance (immutable audit log)                        │
│  7. Policy (cqe_policy_v1.json governance)                  │
└─────────────────────────────────────────────────────────────┘
```

### Spine Components (Always Required)

Every CQE computation, regardless of which higher-level components you use, ultimately relies on these spine components:

```python
# The Spine - Always Present
from layer2_geometric.e8.lattice import E8Lattice
from layer1_morphonic.overlay_system import Overlay, ImmutablePose
from layer2_geometric.phi_metric import PhiMetric
from layer1_morphonic.alena_operators import ALENAOperators
from layer1_morphonic.acceptance_rules import AcceptanceRule
from layer1_morphonic.provenance import ProvenanceLogger
from layer4_governance.policy_system import PolicySystem

# Initialize the spine
e8 = E8Lattice()
phi_metric = PhiMetric()
alena = ALENAOperators()
acceptance = AcceptanceRule()
provenance = ProvenanceLogger()
policy = PolicySystem()
policy.load_policy("policies/cqe_policy_v1.json")
```

**Rule**: Any component you add must ultimately produce or consume `Overlay` objects and respect the Φ-minimization principle.

### The Modular Layers

Built on top of the spine are modular layers. You can use any subset:

```
┌─────────────────────────────────────────────────────────────┐
│                    MODULAR COMPONENTS                        │
├─────────────────────────────────────────────────────────────┤
│  GNLC (λ₀, λ₁, λ₂, λ_θ, types, reduction)                  │
│  Transformers & Tokenizers (geo_tokenizer, geo_transformer) │
│  Speedlight (receipt caching, model selection)              │
│  Aletheia (consciousness, Egyptian analysis)                │
│  Embeddings (save/load, Babai, gauge fields)               │
│  Visualization (lattice plotting, E₈ projection)           │
│  Reasoning Engine (geometric inference)                     │
│  Advanced Tools (E₈×3, CRT 24-Ring, EMCP TQF)              │
└─────────────────────────────────────────────────────────────┘
```

**Rule**: You can use any combination of these. They all speak the same language: Overlays and Φ.

---

<a name="component-map"></a>
## Part 3: Complete Component Map

### Layer 1: Morphonic Foundation (The Spine Core)

| Component | File | Purpose | Key Classes/Functions |
|-----------|------|---------|----------------------|
| **Overlay System** | `overlay_system.py` | State representation | `Overlay`, `ImmutablePose`, `TextAdapter`, `NumericAdapter` |
| **ALENA Operators** | `alena_operators.py` | Fundamental transformations | `ALENAOperators.rotate()`, `.weyl_reflect()`, `.midpoint()`, `.parity_mirror()` |
| **Acceptance Rules** | `acceptance_rules.py` | Validation & governance | `AcceptanceRule.is_accepted()`, `ParitySignature` |
| **Provenance** | `provenance.py` | Audit logging | `ProvenanceLogger.log()`, `.get_ledger()`, `.save_receipts()` |
| **Shell Protocol** | `shell_protocol.py` | Bounded search | `ShellProtocol.is_in_shell()`, `.expand_shell()` |
| **ε-Canonicalizer** | `epsilon_canonicalizer.py` | Equivalence classes | `EpsilonCanonicalizer.snap_to_canonical()` |
| **Quadratic Law Harness** | `quadratic_law_harness.py` | Validation tests | `QuadraticLawHarness.run_all_tests()` |

### Layer 2: Geometric Engine (The Mathematical Core)

| Component | File | Purpose | Key Classes/Functions |
|-----------|------|---------|----------------------|
| **E₈ Lattice** | `e8/lattice.py` | 8D lattice structure | `E8Lattice.roots`, `.weyl_group`, `.project_to_lattice()` |
| **Weyl Navigation** | `e8/weyl.py` | Chamber navigation | `WeylChamber.navigate()`, 696M cells |
| **Phi Metric** | `phi_metric.py` | Quality measurement | `PhiMetric.compute()` → 4 components |
| **Digital Roots** | `digital_roots.py` | DR stratification | `DigitalRootSystem.classify()` |
| **E₈×3 Projection** | `e8x3_projection.py` | Comparative analysis | `E8x3Projection.project_to_center()` |
| **CRT 24-Ring** | `crt_24ring.py` | Parallelization | `CRT24Ring.decompose()`, `.merge()` |
| **Babai Embedder** | `babai_embedder.py` | Embedding algorithm | `BabaiEmbedder.embed()` |
| **Geometry Transformer** | `geometry_transformer.py` | Geometric transforms | `GeometryTransformer.transform()` |
| **Visualization** | `visualize_e8_embedding.py` | E₈ plotting | `visualize_embedding()` |

### Layer 3: Operational Systems (High-Level Orchestration)

| Component | File | Purpose | Key Classes/Functions |
|-----------|------|---------|----------------------|
| **Enhanced MORSR** | `morsr_enhanced.py` | Complete optimization | `EnhancedMORSR.optimize()` - integrates everything |
| **Reasoning Engine** | `reasoning_engine.py` | Geometric inference | `ReasoningEngine.reason()` |
| **EMCP TQF** | `emcp_tqf.py` | Chiral coupling | `EMCPTQF.chiral_decompose()`, `.couple()` |

### Layer 4: Governance (Policy & Validation)

| Component | File | Purpose | Key Classes/Functions |
|-----------|------|---------|----------------------|
| **Policy System** | `policy_system.py` | Policy enforcement | `PolicySystem.load_policy()`, `.validate_operation()` |
| **Policy File** | `policies/cqe_policy_v1.json` | System configuration | JSON policy specification |

### Layer 5: Interface (User-Facing Systems)

#### GNLC Subsystem

| Component | File | Purpose | Key Classes/Functions |
|-----------|------|---------|----------------------|
| **λ₀ Atom Calculus** | `gnlc_lambda0.py` | Fundamental terms | `Lambda0Calculus.atom()`, `.apply()` |
| **λ₁ Relation Calculus** | `gnlc_lambda1.py` | Structures | `Lambda1Calculus.create_relation()`, `.tensor_product()` |
| **λ₂ State Calculus** | `gnlc_lambda2.py` | Dynamics | `Lambda2Calculus.create_state()`, `.evolve()` |
| **λ_θ Meta-Calculus** | `gnlc_lambda_theta.py` | Meta-level | `LambdaThetaCalculus.create_schema()`, `.learn_transformation()` |
| **Type System** | `gnlc_type_system.py` | Geometric types | `GeometricTypeSystem.infer_type()`, `.check_type()` |
| **Reduction** | `gnlc_reduction.py` | Computation | `GNLCReductionSystem.normalize()`, `.beta_reduce()` |

#### Transformer & Tokenizer Subsystem

| Component | File | Purpose | Key Classes/Functions |
|-----------|------|---------|----------------------|
| **Geo Tokenizer** | `geo_tokenizer.py` | Geometric tokenization | `GeoTokenizer.tokenize()`, `.detokenize()` |
| **Geo Tokenizer Tie-In** | `geo_tokenizer_tiein.py` | CQE integration | `GeoTokenizerTieIn.encode_to_overlay()` |
| **Geo Transformer** | `geo_transformer.py` | Geometric attention | `GeoTransformer.forward()`, `.attend()` |
| **Geometry Transformer V2** | `geometry_transformer_v2.py` | Enhanced version | `GeometryTransformerV2.transform()` |

#### Speedlight Subsystem

| Component | File | Purpose | Key Classes/Functions |
|-----------|------|---------|----------------------|
| **Speedlight** | `speedlight.py` | Receipt caching | `SpeedLight.compute()`, `.compute_hash()` |
| **Speedlight Sidecar Plus** | `speedlight_sidecar_plus.py` | Model selection | `SpeedLightSidecarPlus.select_model()`, `.route_task()` |

#### Aletheia Subsystem

| Component | File | Purpose | Key Classes/Functions |
|-----------|------|---------|----------------------|
| **Aletheia Main** | `aletheia_system/aletheia.py` | Entry point | `AletheiaSystem.query()`, `.analyze_egyptian()` |
| **CQE Engine** | `aletheia_system/core/cqe_engine.py` | CQE integration | `CQEEngine.process()` |
| **Aletheia AI** | `aletheia_system/ai/aletheia_consciousness.py` | AI reasoning | `AletheiaAI.process_query()`, `.synthesize()` |
| **Egyptian Analyzer** | `aletheia_system/analysis/egyptian_analyzer.py` | Domain analysis | `EgyptianAnalyzer.analyze_images()` |

### Utilities (Cross-Cutting Tools)

| Component | File | Purpose | Key Functions |
|-----------|------|---------|--------------|
| **Save Embedding** | `utils/save_embedding.py` | Persist embeddings | `save_embedding(overlay, path)` |
| **Load Embedding** | `utils/load_embedding.py` | Load embeddings | `load_embedding(path)` |
| **Create Gauge Field** | `utils/create_gauge_field_embedding.py` | Gauge fields | `create_gauge_field()` |
| **Test Suite** | `utils/test_suite.py` | Testing framework | `run_tests()` |

---


<a name="end-to-end-workflow"></a>
## Part 4: End-to-End Solve Workflow - From Text to Geometric Insight

This section details a complete, end-to-end workflow that uses multiple components to solve a problem. This is the **canonical example** of how to orchestrate the CQE runtime.

**Problem**: You have a piece of text, and you want to find its geometric representation, optimize it, reason about it, and visualize the result.

### The 10-Step Workflow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            END-TO-END SOLVE WORKFLOW                             │
├──────────────────────────────────────────────────────────────────────────────────┤
│  1.  Input: Text string                                                          │
│      ↓                                                                           │
│  2.  GeoTokenizer: Text → Geometric Tokens                                       │
│      ↓                                                                           │
│  3.  GeoTransformer: Tokens → Initial Overlay (State)                            │
│      ↓                                                                           │
│  4.  MORSR: Initial Overlay → Optimized Overlay (Φ-minimization)                 │
│      ↓                                                                           │
│  5.  Reasoning Engine: Optimized Overlay → Geometric Insights (Inference)        │
│      ↓                                                                           │
│  6.  Speedlight: Cache the reasoning result for future use                       │
│      ↓                                                                           │
│  7.  Save Embedding: Persist the optimized Overlay to disk                       │
│      ↓                                                                           │
│  8.  Visualize: Plot the optimized Overlay in 2D                                 │
│      ↓                                                                           │
│  9.  Aletheia: Query the system about the result                                 │
│      ↓                                                                           │
│  10. Output: Insights, visualization, and AI response                            │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Complete Code Example

This code implements the entire 10-step workflow. Each step is explained in detail.

```python
# ==============================================================================
# STEP 0: IMPORTS & INITIALIZATION
# ==============================================================================

# --- Spine Components ---
from layer3_operational.morsr_enhanced import EnhancedMORSR

# --- Modular Components ---
from layer5_interface.geo_tokenizer_tiein import GeoTokenizerTieIn
from layer5_interface.geo_transformer import GeoTransformer
from layer3_operational.reasoning_engine import ReasoningEngine
from layer5_interface.speedlight import SpeedLight
from utils.save_embedding import save_embedding
from layer2_geometric.visualize_e8_embedding import visualize_embedding
from aletheia_system.aletheia import AletheiaSystem

# --- Initialize all required systems ---
# This is the "Lego block" assembly
morsr = EnhancedMORSR()
geo_tokenizer = GeoTokenizerTieIn()
geo_transformer = GeoTransformer()
reasoning_engine = ReasoningEngine()
speedlight = SpeedLight()
aletheia = AletheiaSystem()

print("✓ All systems initialized.")

# ==============================================================================
# STEP 1: INPUT
# ==============================================================================

input_text = "The quick brown fox jumps over the lazy dog."
print(f"\n[1] Input text: \"{input_text}\"")

# ==============================================================================
# STEP 2: GEOMETRIC TOKENIZATION
# ==============================================================================

# Why: We need to convert the flat text string into a sequence of geometric
# tokens that the GeoTransformer can understand.

geo_tokens = geo_tokenizer.tokenize(input_text)
print(f"\n[2] Geometric Tokens: {len(geo_tokens)} tokens generated.")
# print(geo_tokens)  # Uncomment to see the tokens

# ==============================================================================
# STEP 3: GEOMETRIC TRANSFORMER
# ==============================================================================

# Why: The GeoTransformer uses geometric attention to convert the sequence of
# tokens into a single, holistic Overlay. This is the initial geometric
# representation of the entire text.

initial_overlay = geo_transformer.forward(geo_tokens)
phi_initial = morsr.lambda0.alena._compute_phi(initial_overlay)
print(f"\n[3] Initial Overlay created. Φ = {phi_initial:.6f}")

# ==============================================================================
# STEP 4: MORSR OPTIMIZATION
# ==============================================================================

# Why: The initial Overlay is a raw representation. We use MORSR to optimize
# it, finding a nearby state with a lower Φ value. This is like "settling"
# the geometry into a more stable and harmonious configuration.

print("\n[4] Optimizing Overlay with MORSR...")
opt_result = morsr.optimize(initial_overlay, max_iterations=100)
optimized_overlay = opt_result["final_overlay"]
phi_final = morsr.lambda0.alena._compute_phi(optimized_overlay)

print(f"   - Optimization complete in {opt_result["iterations"]} iterations.")
print(f"   - Final Φ = {phi_final:.6f} (Improvement: {phi_initial - phi_final:.6f})")

# ==============================================================================
# STEP 5: REASONING ENGINE
# ==============================================================================

# Why: Now that we have an optimized geometric state, we can use the
# ReasoningEngine to extract insights from it. The engine analyzes the
# geometric properties (digital roots, Weyl chamber, etc.) to make inferences.

print("\n[5] Extracting insights with Reasoning Engine...")

# We use Speedlight to cache this potentially expensive operation
def reason_on_overlay():
    return reasoning_engine.reason(optimized_overlay)

# ==============================================================================
# STEP 6: SPEEDLIGHT CACHING
# ==============================================================================

# Why: If we perform the same reasoning task again, we don't want to re-compute
# it. Speedlight creates a receipt based on the Overlay's content and caches
# the result. The next time we call it with the same Overlay, it will be an
# instant cache hit.

reasoning_result, cost = speedlight.compute_hash(optimized_overlay.e8_base.tobytes(), reason_on_overlay)

print(f"   - Reasoning complete in {cost:.4f} seconds.")
print(f"   - Insights: {reasoning_result}")

# --- Demonstrate caching ---
print("   - Running reasoning again to demonstrate caching...")
_, cost_cached = speedlight.compute_hash(optimized_overlay.e8_base.tobytes(), reason_on_overlay)
print(f"   - Cached reasoning complete in {cost_cached:.4f} seconds. (Cache hit!)")

# ==============================================================================
# STEP 7: SAVE EMBEDDING
# ==============================================================================

# Why: The optimized Overlay is a valuable asset. We persist it to disk so we
# can load it later without having to re-run the entire pipeline.

embedding_path = "/tmp/optimized_fox_embedding.json"
save_embedding(optimized_overlay, embedding_path)
print(f"\n[7] Optimized Overlay saved to {embedding_path}")

# ==============================================================================
# STEP 8: VISUALIZE EMBEDDING
# ==============================================================================

# Why: An 8-dimensional object is hard to understand. We project the E₈ base
# vector to 2D to get a visual intuition for its structure.

plot_path = "/tmp/optimized_fox_plot.png"
visualize_embedding(optimized_overlay.e8_base, plot_path)
print(f"\n[8] 2D visualization of the Overlay saved to {plot_path}")

# ==============================================================================
# STEP 9: ALETHEIA QUERY
# ==============================================================================

# Why: Aletheia provides a high-level, natural language interface to the
# system. We can ask it to reflect on the computation we just performed.

print("\n[9] Querying Aletheia about the result...")
query = f"Based on the geometric insights from the text \"{input_text}\", what is the primary characteristic of the resulting Overlay?"
aletheia_response = aletheia.query(query)

print(f"   - Aletheia's response: {aletheia_response}")

# ==============================================================================
# STEP 10: FINAL OUTPUT
# ==============================================================================

print("\n[10] End-to-End Workflow Complete!")
print("     - Final Insights: ", reasoning_result)
print("     - Final Visualization: ", plot_path)
print("     - Final AI Response: ", aletheia_response)

```

### How to Run This Workflow

1.  Save the code above as `end_to_end_solve.py`.
2.  Ensure all paths in the `sys.path.insert` calls within the component files are correct.
3.  Run from the command line: `python end_to_end_solve.py`

### Expected Output

You will see a step-by-step log of the entire process, including:
-   The initial and final Φ values.
-   The insights from the reasoning engine.
-   The demonstration of Speedlight's caching.
-   The file paths for the saved embedding and visualization.
-   A natural language response from Aletheia.

This workflow demonstrates the power of the modular CQE system. Each component is a self-contained "Lego block," but when assembled, they form a powerful, end-to-end pipeline for geometric computation and analysis.


<a name="integration-patterns"></a>
## Part 5: Component Integration Patterns

This section details how to "snap" the Lego blocks together. It provides patterns for combining different components.

### Pattern 1: Transformer → MORSR

**Goal**: Convert text to an optimized geometric state.

```python
# 1. Initialize
geo_tokenizer = GeoTokenizerTieIn()
geo_transformer = GeoTransformer()
morsr = EnhancedMORSR()

# 2. Tokenize
geo_tokens = geo_tokenizer.tokenize("Your text here")

# 3. Transform
initial_overlay = geo_transformer.forward(geo_tokens)

# 4. Optimize
optimized_overlay = morsr.optimize(initial_overlay)["final_overlay"]
```

**Why it works**: The `GeoTransformer` outputs an `Overlay`, which is the exact input format required by `MORSR`.

### Pattern 2: MORSR → Reasoning Engine → Speedlight

**Goal**: Extract and cache insights from an optimized state.

```python
# 1. Initialize
reasoning_engine = ReasoningEngine()
speedlight = SpeedLight()

# 2. Define the computation to be cached
def reason_on_overlay(overlay):
    return reasoning_engine.reason(overlay)

# 3. Compute and cache
# The hash is created from the Overlay's E8 base vector
reasoning_result, _ = speedlight.compute_hash(
    optimized_overlay.e8_base.tobytes(),
    lambda: reason_on_overlay(optimized_overlay)
)
```

**Why it works**: `Speedlight` is a generic caching wrapper. It takes a hashable representation of the input (the E₈ vector) and a function to execute. The `ReasoningEngine` takes an `Overlay` and returns a JSON-serializable dictionary of insights, which is easily cached.

### Pattern 3: Save/Load Cycle

**Goal**: Persist an Overlay and retrieve it later.

```python
# 1. Initialize
from utils.save_embedding import save_embedding
from utils.load_embedding import load_embedding

# 2. Save
embedding_path = "/tmp/my_embedding.json"
save_embedding(optimized_overlay, embedding_path)

# 3. Load
loaded_overlay = load_embedding(embedding_path)

# 4. Verify
assert (loaded_overlay.e8_base == optimized_overlay.e8_base).all()
```

**Why it works**: `save_embedding` serializes all components of the `Overlay` (E₈ base, activations, pose, etc.) to a JSON file. `load_embedding` reconstructs the `Overlay` object from this file.

### Pattern 4: GNLC + MORSR

**Goal**: Use MORSR to pre-optimize atoms before building a complex GNLC structure.

```python
# 1. Initialize
lambda0 = Lambda0Calculus()
lambda1 = Lambda1Calculus()
morsr = EnhancedMORSR()

# 2. Create raw overlays
overlay_a = ...
overlay_b = ...

# 3. Pre-optimize them
opt_overlay_a = morsr.optimize(overlay_a)["final_overlay"]
opt_overlay_b = morsr.optimize(overlay_b)["final_overlay"]

# 4. Create atoms from the *optimized* overlays
atom_a = lambda0.atom(opt_overlay_a)
atom_b = lambda0.atom(opt_overlay_b)

# 5. Build the GNLC structure
relation = lambda1.create_relation(atom_a, atom_b, "optimized_relation")
```

**Why it works**: By optimizing the components *before* you assemble them, the final structure starts in a much lower Φ state, making subsequent normalization faster and more effective.

### Pattern 5: Aletheia as the Final Layer

**Goal**: Use Aletheia to provide a natural language summary of a complex geometric computation.

```python
# 1. Initialize
aletheia = AletheiaSystem()

# 2. Perform your complex workflow (e.g., Patterns 1-4)
# ... result is a set of insights, a final overlay, etc.

# 3. Formulate a query to Aletheia
query = f"I have performed a geometric optimization. The final state has a Φ of {phi_final:.4f} and the reasoning engine suggests the dominant characteristic is 
{reasoning_result["dominant_characteristic"]}. Please provide a high-level summary of this outcome."

# 4. Get the AI response
response = aletheia.query(query)
```

**Why it works**: Aletheia is designed to be the top-level interface. Its internal `AletheiaAI` component can process natural language and relate it to the underlying geometric concepts, providing a bridge between the human user and the CQE runtime.


<a name="system-guide"></a>
## Part 6: System-by-System Operational Guide

This section provides detailed operational instructions for each major system.

---

### System 1: Aletheia - The Conscious AI Interface

**Purpose**: Aletheia is the high-level, natural language interface to the CQE runtime. It provides AI-powered reasoning, Egyptian hieroglyphic analysis, and knowledge synthesis.

**When to use**: When you need a user-friendly, conversational interface or when you need domain-specific analysis (e.g., Egyptian texts).

**Complete Example**:

```python
from aletheia_system.aletheia import AletheiaSystem

# Initialize Aletheia
aletheia = AletheiaSystem(verbose=True)

# --- Mode 1: Query Mode ---
# Ask Aletheia a question about the CQE system
query = "What is the significance of the Φ metric in geometric computation?"
response = aletheia.query(query)
print(f"Aletheia's response: {response}")

# --- Mode 2: Egyptian Analysis Mode ---
# Analyze Egyptian hieroglyphic images
image_paths = ["/path/to/hieroglyph1.jpg", "/path/to/hieroglyph2.jpg"]
analysis_results = aletheia.analyze_egyptian(image_paths)
print(f"Egyptian analysis: {analysis_results}")

# --- Mode 3: Knowledge Synthesis Mode ---
# Synthesize knowledge from multiple data files
data_files = ["/path/to/data1.json", "/path/to/data2.json"]
synthesis = aletheia.synthesize_knowledge(data_files)
print(f"Knowledge synthesis: {synthesis}")

# --- Mode 4: Interactive Mode ---
# Enter interactive mode for a REPL-like experience
aletheia.interactive_mode()
# This will start an interactive session where you can type commands
```

**Key Components**:
-   `AletheiaSystem`: Main entry point
-   `CQEEngine`: Integrates with the CQE runtime
-   `AletheiaAI`: Provides AI reasoning
-   `EgyptianAnalyzer`: Domain-specific analysis

**Integration with CQE**: Aletheia internally uses `CQEEngine`, which wraps the core CQE components (MORSR, ALENA, etc.). When you query Aletheia, it translates your natural language into geometric operations and then translates the results back into natural language.

---

### System 2: Speedlight - Receipt-Based Caching

**Purpose**: Speedlight provides idempotent, receipt-based caching with a 99.9% cache hit rate. It eliminates redundant computation.

**When to use**: When you have expensive computations that may be repeated with the same inputs.

**Complete Example**:

```python
from layer5_interface.speedlight import SpeedLight
import time

# Initialize Speedlight
speedlight = SpeedLight()

# Define an expensive computation
def expensive_computation(x, y):
    time.sleep(2)  # Simulate expensive work
    return x ** y

# --- First call: Cache miss ---
result1, cost1 = speedlight.compute(
    task_id="power_3_4",
    compute_fn=expensive_computation,
    x=3, y=4
)
print(f"Result: {result1}, Cost: {cost1:.2f}s")  # ~2 seconds

# --- Second call: Cache hit ---
result2, cost2 = speedlight.compute(
    task_id="power_3_4",
    compute_fn=expensive_computation,
    x=3, y=4
)
print(f"Result: {result2}, Cost: {cost2:.2f}s")  # ~0 seconds (instant!)

# --- Using compute_hash for automatic task ID generation ---
data = {"x": 5, "y": 6}
result3, cost3 = speedlight.compute_hash(
    data=data,
    compute_fn=lambda: expensive_computation(data["x"], data["y"])
)
print(f"Result: {result3}, Cost: {cost3:.2f}s")

# --- View statistics ---
print(f"Cache stats: {speedlight.stats}")
# {'hits': 1, 'misses': 2, 'time_saved': 2.0}
```

**Key Features**:
-   **Content-addressed receipts**: The task ID is a hash of the inputs.
-   **Merkle chain**: Receipts are chained for integrity.
-   **Thread-safe**: Uses locks for concurrent access.

**Integration with CQE**: Speedlight is agnostic to what you cache. You can cache the results of MORSR optimizations, reasoning engine inferences, or any other expensive CQE operation.

---

### System 3: GeoTokenizer & GeoTransformer - Text to Geometry

**Purpose**: Convert text into geometric representations that can be processed by the CQE runtime.

**When to use**: When you need to process natural language text using geometric methods.

**Complete Example**:

```python
from layer5_interface.geo_tokenizer_tiein import GeoTokenizerTieIn
from layer5_interface.geo_transformer import GeoTransformer

# Initialize
geo_tokenizer = GeoTokenizerTieIn()
geo_transformer = GeoTransformer()

# Input text
text = "The universe is a vast geometric structure."

# --- Step 1: Tokenization ---
# Convert text to geometric tokens
geo_tokens = geo_tokenizer.tokenize(text)
print(f"Number of tokens: {len(geo_tokens)}")
# Each token is a geometric object (e.g., a point in E₈)

# --- Step 2: Transformation ---
# Use geometric attention to combine tokens into a single Overlay
overlay = geo_transformer.forward(geo_tokens)
print(f"Overlay E₈ base: {overlay.e8_base}")
print(f"Active roots: {overlay.activations.sum()}")

# --- Step 3: Detokenization (optional) ---
# Convert tokens back to text (for verification)
reconstructed_text = geo_tokenizer.detokenize(geo_tokens)
print(f"Reconstructed text: {reconstructed_text}")
```

**Key Components**:
-   `GeoTokenizer`: Converts text ↔ geometric tokens
-   `GeoTokenizerTieIn`: Integrates tokenizer with CQE Overlays
-   `GeoTransformer`: Applies geometric attention to token sequences

**Integration with CQE**: The `GeoTransformer` outputs an `Overlay`, which is the universal data structure for the CQE runtime. This Overlay can then be fed into MORSR, GNLC, or any other component.

---

### System 4: GNLC - The Geometry-Native Lambda Calculus

**Purpose**: GNLC provides a formal, stratified calculus for geometric computation. It allows you to build complex programs as geometric structures and execute them via normalization.

**When to use**: When you need structured, formal computation with type checking and provable properties.

**Complete Example**:

```python
from layer5_interface.gnlc_lambda0 import Lambda0Calculus
from layer5_interface.gnlc_lambda1 import Lambda1Calculus
from layer5_interface.gnlc_lambda2 import Lambda2Calculus
from layer5_interface.gnlc_type_system import GeometricTypeSystem
from layer5_interface.gnlc_reduction import GNLCReductionSystem
from layer1_morphonic.overlay_system import NumericAdapter

# Initialize all GNLC layers
lambda0 = Lambda0Calculus()
lambda1 = Lambda1Calculus()
lambda2 = Lambda2Calculus()
type_system = GeometricTypeSystem()
reduction = GNLCReductionSystem()

# --- λ₀: Create atoms ---
adapter = NumericAdapter()
overlay_a = adapter.encode([1, 2, 3, 4, 5, 6, 7, 8])
overlay_b = adapter.encode([8, 7, 6, 5, 4, 3, 2, 1])
atom_a = lambda0.atom(overlay_a)
atom_b = lambda0.atom(overlay_b)

# --- Type System: Infer types ---
type_a = type_system.infer_type(atom_a)
type_b = type_system.infer_type(atom_b)
print(f"Type of atom_a: {type_a}")
print(f"Type of atom_b: {type_b}")

# --- λ₁: Create a relation ---
relation = lambda1.create_relation(atom_a, atom_b, "connects")
print(f"Relation distance: {relation.distance:.6f}")

# --- λ₂: Create a system state ---
state = lambda2.create_state([atom_a, atom_b], timestamp=0.0)
print(f"State Φ: {state.phi:.6f}")

# --- Reduction: Normalize ---
normal_form = reduction.normalize(atom_a, max_steps=50)
print(f"Normalization: {normal_form.num_steps} steps, ΔΦ={normal_form.total_delta_phi:.6f}")

# --- Type System: Verify the result ---
result_type = type_system.infer_type(normal_form.term)
print(f"Type of normalized term: {result_type}")
```

**Key Layers**:
-   **λ₀**: Atoms (fundamental units)
-   **λ₁**: Relations (structures)
-   **λ₂**: States (dynamics)
-   **λ_θ**: Meta (rules and schemas)
-   **Type System**: Geometric types
-   **Reduction**: Computation via normalization

**Integration with CQE**: GNLC is built entirely on top of the CQE spine. Every GNLC term is ultimately an `Overlay`, and every operation respects the Φ-minimization principle.

---

### System 5: Embedding Save/Load - Persistence

**Purpose**: Save and load `Overlay` objects to/from disk for persistence and reuse.

**When to use**: When you need to persist the results of expensive computations or share geometric states across sessions.

**Complete Example**:

```python
from utils.save_embedding import save_embedding
from utils.load_embedding import load_embedding
from layer1_morphonic.overlay_system import NumericAdapter

# --- Create an Overlay ---
adapter = NumericAdapter()
overlay = adapter.encode([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])

# --- Save to disk ---
embedding_path = "/tmp/my_embedding.json"
save_embedding(overlay, embedding_path)
print(f"Overlay saved to {embedding_path}")

# --- Load from disk ---
loaded_overlay = load_embedding(embedding_path)
print(f"Overlay loaded from {embedding_path}")

# --- Verify integrity ---
import numpy as np
assert np.allclose(overlay.e8_base, loaded_overlay.e8_base)
assert (overlay.activations == loaded_overlay.activations).all()
print("✓ Loaded Overlay matches the original.")
```

**File Format**: The saved file is a JSON object containing:
-   `e8_base`: The 8D E₈ vector
-   `activations`: The 240-element binary activation vector
-   `pose`: Position, orientation, timestamp
-   `weights` (optional)
-   `phase` (optional)

**Integration with CQE**: This is a utility that works with the fundamental `Overlay` data structure. Any component that produces or consumes Overlays can use this save/load mechanism.

---

### System 6: Visualization - Seeing the Geometry

**Purpose**: Project 8-dimensional E₈ vectors to 2D for visualization.

**When to use**: When you need to visually inspect or present geometric states.

**Complete Example**:

```python
from layer2_geometric.visualize_e8_embedding import visualize_embedding
from layer1_morphonic.overlay_system import NumericAdapter

# --- Create an Overlay ---
adapter = NumericAdapter()
overlay = adapter.encode([1, 2, 3, 4, 5, 6, 7, 8])

# --- Visualize ---
plot_path = "/tmp/my_visualization.png"
visualize_embedding(overlay.e8_base, plot_path)
print(f"Visualization saved to {plot_path}")

# The plot will show a 2D projection of the 8D vector using PCA
```

**What the plot shows**: The visualization uses Principal Component Analysis (PCA) to project the 8-dimensional E₈ vector to 2D. This gives you a visual sense of the "shape" of the geometric state.

**Integration with CQE**: This is a read-only utility. It takes an E₈ vector (from an `Overlay`) and produces a plot. It does not modify the state.

---

### System 7: Reasoning Engine - Geometric Inference

**Purpose**: Extract high-level insights from geometric states using rule-based and heuristic analysis.

**When to use**: When you need to interpret the "meaning" of a geometric state in terms of its structural properties.

**Complete Example**:

```python
from layer3_operational.reasoning_engine import ReasoningEngine
from layer1_morphonic.overlay_system import NumericAdapter

# Initialize
reasoning_engine = ReasoningEngine()

# --- Create an Overlay ---
adapter = NumericAdapter()
overlay = adapter.encode([1, 2, 3, 4, 5, 6, 7, 8])

# --- Perform reasoning ---
insights = reasoning_engine.reason(overlay)

# --- View insights ---
print("Geometric Insights:")
for key, value in insights.items():
    print(f"  {key}: {value}")

# Example output:
# {
#   "dominant_digital_root": 3,
#   "weyl_chamber": 42,
#   "parity_syndrome": 0.123,
#   "active_root_count": 120,
#   "phi_value": 2.456,
#   "suggested_operation": "weyl_reflect"
# }
```

**Key Features**:
-   Analyzes digital roots, Weyl chambers, parity, etc.
-   Suggests next operations based on heuristics
-   Returns a structured JSON dictionary

**Integration with CQE**: The `ReasoningEngine` is a read-only analyzer. It takes an `Overlay` and returns insights. It is often used in combination with Speedlight for caching.

---


<a name="advanced-examples"></a>
## Part 7: Advanced Integration Examples

This section provides complete, real-world examples that combine multiple systems.

---

### Example 1: Text Analysis Pipeline with Full Caching

**Goal**: Analyze a piece of text, optimize its geometric representation, extract insights, cache everything, and visualize the result.

```python
#!/usr/bin/env python3
"""
Complete Text Analysis Pipeline
================================
Demonstrates: GeoTokenizer + GeoTransformer + MORSR + Reasoning + Speedlight + Visualization
"""

from layer5_interface.geo_tokenizer_tiein import GeoTokenizerTieIn
from layer5_interface.geo_transformer import GeoTransformer
from layer3_operational.morsr_enhanced import EnhancedMORSR
from layer3_operational.reasoning_engine import ReasoningEngine
from layer5_interface.speedlight import SpeedLight
from utils.save_embedding import save_embedding
from layer2_geometric.visualize_e8_embedding import visualize_embedding
import hashlib

def analyze_text(text, output_dir="/tmp"):
    """Complete text analysis pipeline."""
    
    # Initialize all components
    geo_tokenizer = GeoTokenizerTieIn()
    geo_transformer = GeoTransformer()
    morsr = EnhancedMORSR()
    reasoning_engine = ReasoningEngine()
    speedlight = SpeedLight()
    
    print(f"Analyzing text: \"{text}\"")
    
    # Step 1: Tokenize
    task_id_tokenize = hashlib.sha256(text.encode()).hexdigest() + "_tokens"
    geo_tokens, _ = speedlight.compute(
        task_id=task_id_tokenize,
        compute_fn=geo_tokenizer.tokenize,
        text=text
    )
    print(f"✓ Tokenized: {len(geo_tokens)} tokens")
    
    # Step 2: Transform
    task_id_transform = task_id_tokenize + "_overlay"
    initial_overlay, _ = speedlight.compute(
        task_id=task_id_transform,
        compute_fn=geo_transformer.forward,
        geo_tokens=geo_tokens
    )
    phi_initial = morsr.lambda0.alena._compute_phi(initial_overlay)
    print(f"✓ Initial Overlay: Φ = {phi_initial:.6f}")
    
    # Step 3: Optimize
    task_id_optimize = task_id_transform + "_optimized"
    opt_result, _ = speedlight.compute(
        task_id=task_id_optimize,
        compute_fn=morsr.optimize,
        initial_overlay=initial_overlay,
        max_iterations=100
    )
    optimized_overlay = opt_result["final_overlay"]
    phi_final = morsr.lambda0.alena._compute_phi(optimized_overlay)
    print(f"✓ Optimized: Φ = {phi_final:.6f} (Δ = {phi_initial - phi_final:.6f})")
    
    # Step 4: Reason
    task_id_reason = task_id_optimize + "_insights"
    insights, _ = speedlight.compute_hash(
        data=optimized_overlay.e8_base.tobytes(),
        compute_fn=lambda: reasoning_engine.reason(optimized_overlay)
    )
    print(f"✓ Insights: {insights}")
    
    # Step 5: Save
    embedding_path = f"{output_dir}/embedding_{task_id_optimize[:8]}.json"
    save_embedding(optimized_overlay, embedding_path)
    print(f"✓ Saved to: {embedding_path}")
    
    # Step 6: Visualize
    plot_path = f"{output_dir}/plot_{task_id_optimize[:8]}.png"
    visualize_embedding(optimized_overlay.e8_base, plot_path)
    print(f"✓ Visualized: {plot_path}")
    
    # Return all results
    return {
        "text": text,
        "phi_initial": phi_initial,
        "phi_final": phi_final,
        "insights": insights,
        "embedding_path": embedding_path,
        "plot_path": plot_path,
        "cache_stats": speedlight.stats
    }

# Run the pipeline
if __name__ == "__main__":
    result = analyze_text("The quick brown fox jumps over the lazy dog.")
    print("\n=== Final Results ===")
    for key, value in result.items():
        print(f"{key}: {value}")
```

**Key Features**:
-   Every expensive operation is cached with Speedlight
-   The second run of this script will be nearly instant (all cache hits)
-   All intermediate and final states are persisted

---

### Example 2: GNLC Program with Aletheia Interpretation

**Goal**: Build a GNLC program, normalize it, and ask Aletheia to explain the result.

```python
#!/usr/bin/env python3
"""
GNLC Program with AI Interpretation
====================================
Demonstrates: GNLC (all layers) + Reduction + Aletheia
"""

from layer5_interface.gnlc_lambda0 import Lambda0Calculus
from layer5_interface.gnlc_lambda1 import Lambda1Calculus
from layer5_interface.gnlc_reduction import GNLCReductionSystem
from layer5_interface.gnlc_type_system import GeometricTypeSystem
from layer1_morphonic.overlay_system import NumericAdapter
from aletheia_system.aletheia import AletheiaSystem

def build_and_interpret_gnlc_program():
    """Build a GNLC program and ask Aletheia to interpret it."""
    
    # Initialize
    lambda0 = Lambda0Calculus()
    lambda1 = Lambda1Calculus()
    reduction = GNLCReductionSystem()
    type_system = GeometricTypeSystem()
    aletheia = AletheiaSystem()
    
    # Step 1: Create atoms
    adapter = NumericAdapter()
    atom_a = lambda0.atom(adapter.encode([1, 0, 0, 0, 0, 0, 0, 0]))
    atom_b = lambda0.atom(adapter.encode([0, 1, 0, 0, 0, 0, 0, 0]))
    print("✓ Created two atoms")
    
    # Step 2: Type check
    type_a = type_system.infer_type(atom_a)
    type_b = type_system.infer_type(atom_b)
    print(f"✓ Type of atom_a: {type_a}")
    print(f"✓ Type of atom_b: {type_b}")
    
    # Step 3: Create a relation
    relation = lambda1.create_relation(atom_a, atom_b, "orthogonal_basis")
    print(f"✓ Created relation with distance: {relation.distance:.6f}")
    
    # Step 4: Normalize atom_a
    normal_form = reduction.normalize(atom_a, max_steps=50)
    print(f"✓ Normalized atom_a in {normal_form.num_steps} steps")
    print(f"  ΔΦ = {normal_form.total_delta_phi:.6f}")
    
    # Step 5: Ask Aletheia to interpret
    query = f"""
    I have created a GNLC program with two atoms representing orthogonal basis vectors.
    After normalization, the total change in Φ was {normal_form.total_delta_phi:.6f}.
    The relation between the atoms has a distance of {relation.distance:.6f}.
    What does this tell us about the geometric structure of this program?
    """
    
    response = aletheia.query(query)
    print(f"\n=== Aletheia's Interpretation ===")
    print(response)
    
    return {
        "normal_form": normal_form,
        "relation": relation,
        "aletheia_response": response
    }

# Run
if __name__ == "__main__":
    result = build_and_interpret_gnlc_program()
```

**Key Features**:
-   Demonstrates the full GNLC workflow
-   Uses Aletheia to provide a natural language interpretation
-   Shows how to combine low-level geometric operations with high-level AI reasoning

---

### Example 3: Multi-Stage Optimization with E₈×3 Projection

**Goal**: Optimize two different texts separately, then use E₈×3 projection to find a "consensus" representation.

```python
#!/usr/bin/env python3
"""
Multi-Stage Optimization with E₈×3 Projection
==============================================
Demonstrates: GeoTokenizer + MORSR + E₈×3 Projection
"""

from layer5_interface.geo_tokenizer_tiein import GeoTokenizerTieIn
from layer5_interface.geo_transformer import GeoTransformer
from layer3_operational.morsr_enhanced import EnhancedMORSR
from layer2_geometric.e8x3_projection import E8x3Projection

def find_consensus(text_left, text_right):
    """Find a consensus geometric representation of two texts."""
    
    # Initialize
    geo_tokenizer = GeoTokenizerTieIn()
    geo_transformer = GeoTransformer()
    morsr = EnhancedMORSR()
    e8x3 = E8x3Projection()
    
    print(f"Left text: \"{text_left}\"")
    print(f"Right text: \"{text_right}\"")
    
    # Process left text
    tokens_left = geo_tokenizer.tokenize(text_left)
    overlay_left = geo_transformer.forward(tokens_left)
    opt_left = morsr.optimize(overlay_left, max_iterations=50)["final_overlay"]
    phi_left = morsr.lambda0.alena._compute_phi(opt_left)
    print(f"✓ Left optimized: Φ = {phi_left:.6f}")
    
    # Process right text
    tokens_right = geo_tokenizer.tokenize(text_right)
    overlay_right = geo_transformer.forward(tokens_right)
    opt_right = morsr.optimize(overlay_right, max_iterations=50)["final_overlay"]
    phi_right = morsr.lambda0.alena._compute_phi(opt_right)
    print(f"✓ Right optimized: Φ = {phi_right:.6f}")
    
    # Project to center using phi-probe conflict resolution
    center_overlay = e8x3.project_to_center(
        left=opt_left,
        right=opt_right,
        conflict_resolution='phi_probe'
    )
    phi_center = morsr.lambda0.alena._compute_phi(center_overlay)
    print(f"✓ Center consensus: Φ = {phi_center:.6f}")
    
    # Analyze the consensus
    active_left = opt_left.activations.sum()
    active_right = opt_right.activations.sum()
    active_center = center_overlay.activations.sum()
    
    print(f"\nActive roots: Left={active_left}, Right={active_right}, Center={active_center}")
    
    return {
        "overlay_left": opt_left,
        "overlay_right": opt_right,
        "overlay_center": center_overlay,
        "phi_left": phi_left,
        "phi_right": phi_right,
        "phi_center": phi_center
    }

# Run
if __name__ == "__main__":
    result = find_consensus(
        text_left="Artificial intelligence is transforming society.",
        text_right="Machine learning algorithms are reshaping the world."
    )
```

**Key Features**:
-   Demonstrates parallel processing of two inputs
-   Uses E₈×3 projection to find a consensus
-   Shows how to compare and merge geometric states

---

<a name="troubleshooting"></a>
## Part 8: Troubleshooting Integration Issues

### Common Integration Problems

#### Problem 1: "Overlay object has no attribute 'e8_base'"

**Cause**: You are trying to use a component that expects an `Overlay` object, but you passed something else (e.g., a dictionary, a NumPy array).

**Solution**: Ensure you are creating `Overlay` objects correctly:
```python
from layer1_morphonic.overlay_system import NumericAdapter
adapter = NumericAdapter()
overlay = adapter.encode([1, 2, 3, 4, 5, 6, 7, 8])  # This is an Overlay
```

#### Problem 2: "Module not found" errors

**Cause**: The Python path is not set correctly, or you are running from the wrong directory.

**Solution**: Always run scripts from the `cqe_unified_runtime` directory, or add it to your Python path:
```python
import sys
sys.path.insert(0, "/path/to/cqe_unified_runtime")
```

#### Problem 3: Speedlight cache never hits

**Cause**: The task ID or data hash is changing between calls, even though the inputs are logically the same.

**Solution**: Ensure you are using consistent, deterministic task IDs or data hashes:
```python
# BAD: This will create a different hash each time due to floating point precision
data = {"x": 1.0 / 3.0}

# GOOD: Round or use fixed-precision representations
data = {"x": round(1.0 / 3.0, 6)}
```

#### Problem 4: GNLC normalization does not converge

**Cause**: The term is inherently complex, or there is a type error.

**Solution**:
1. Type-check the term before normalization:
   ```python
   type_ = type_system.infer_type(term)
   is_valid = type_system.check_type(term, type_)
   if not is_valid:
       print("Type error detected!")
   ```
2. Increase `max_steps` in the `normalize()` call.
3. Pre-optimize the atoms with MORSR before building the GNLC structure.

#### Problem 5: Aletheia gives generic or unhelpful responses

**Cause**: The query is too vague, or Aletheia's internal models are not configured correctly.

**Solution**:
1. Provide more context in your query. Include specific values (Φ, digital roots, etc.).
2. Ensure Aletheia's `CQEEngine` is properly initialized and connected to the CQE runtime.

---

## Part 9: Best Practices for System Integration

### 1. Always Initialize the Spine First

Before using any modular component, ensure the spine is initialized:
```python
from layer3_operational.morsr_enhanced import EnhancedMORSR
morsr = EnhancedMORSR()  # This initializes the entire spine
```

### 2. Use Speedlight for Expensive Operations

Any operation that takes more than a few seconds should be wrapped in Speedlight:
```python
result, cost = speedlight.compute_hash(data, expensive_function)
```

### 3. Type-Check GNLC Terms

Always infer and check types when working with GNLC:
```python
type_ = type_system.infer_type(term)
assert type_system.check_type(term, type_)
```

### 4. Save Important Overlays

If you've spent significant computation to create an Overlay, save it:
```python
save_embedding(overlay, "/path/to/embedding.json")
```

### 5. Use Aletheia as the Top Layer

For user-facing applications, use Aletheia to provide a natural language interface:
```python
response = aletheia.query("What is the significance of this result?")
```

---

## Part 10: Conclusion - The Power of Modularity

The CQE Unified Runtime is a **modular geometric computation system**. Its power comes from the ability to mix and match components to solve diverse problems. The key to successful integration is understanding the **spine** (the invariant core) and how each component speaks the common language of **Overlays** and **Φ-minimization**.

**You now have**:
-   A complete map of all components
-   Detailed operational guides for each system
-   End-to-end workflows and integration patterns
-   Advanced examples combining multiple systems
-   Troubleshooting guidance

**You are ready to orchestrate the CQE runtime at the highest level of sophistication.**

---

**Document ID**: COMPLETE-SYS-GUIDE-001  
**Classification**: Technical Reference - Complete Integration  
**Status**: Production-Ready  
**Version**: 2.0

