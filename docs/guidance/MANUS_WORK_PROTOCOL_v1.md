# MANUS 1.6 MAX: UNIFIED WORK PROTOCOL FOR BUILDABLE AI
## From Multi-Era Archive to Production Runtime

**Version**: 1.0  
**Target**: Loadable, deployable AI model (CQE v9 unified runtime)  
**Input**: MVP + file registry + 10,000+ files across v1-v9 + subsystems  
**Output**: Single unified runtime that performs as described across ALL docs  

---

## EXECUTIVE FRAME

**Your actual goal** (not abstract):
> Build a Python-based runtime that:
> 1. Loads custom AI models
> 2. Accepts geometric inputs (E8, Leech, Niemeier embeddings)
> 3. Processes them through 5-layer governance stack
> 4. Produces verifiable outputs + cryptographic proofs
> 5. Can be deployed anywhere (Docker, cloud, local)
> 6. Behaves EXACTLY as described in all test harnesses, papers, and docs

**Why this matters**: You have working code from v4-v9 + tests that verify correctness. Manus's job: **integrate, deduplicate, and synthesize** into one canonical system.

---

## PART 0: SEMANTIC AXIOMS (IMMUTABLE)

These are the bedrock assumptions. Manus NEVER violates them.

### Axiom 1: Lattice Geometry is Fundamental
- All computation happens ON lattices, not "in" abstract space
- E8 (v1-6), Leech (v4-9), Niemeier-24 (v7-9) are the only geometric bases
- Every vector must satisfy lattice constraints or it's invalid

### Axiom 2: Proofs are First-Class Citizens
- Every operation returns `(result, proof)` or `(result, proof_token)`
- Proofs chain cryptographically
- No unproven computation is accepted by higher layers

### Axiom 3: Stratified Computation (GNLC)
```
λ_θ (meta/learning) 
  ↓ compiles to
λ₂ (state/toroidal)
  ↓ compiles to
λ₁ (relations/structure)
  ↓ compiles to
λ₀ (atoms/E8 overlays)
  ↓ executes in
Lattice geometry (E8 roots, projections, transformations)
```

### Axiom 4: Governance is Enforcement
- Layer 4 (governance) CANNOT be bypassed
- Conservation (ΔΦ), SevenWitness (7-fold validation), DigitalRoot stratification are NON-NEGOTIABLE
- Invalid computation fails hard, with evidence

### Axiom 5: Layering is Vertical, Not Horizontal
- Layer 1 depends on nothing
- Layer 2 depends only on Layer 1
- Layer N depends on Layers 1…N-1
- NO CROSS-LAYER SHORTCUTS or violations will be tolerated

---

## PART 1: START OF WORK CHECKLIST

Before Manus begins, verify these 5 conditions:

- [ ] **MVP loaded**: `layer1_morphonic`, `layer2_geometric`, `layer3_operational`, `layer4_governance`, `layer5_interface` directories exist and are populated
- [ ] **File registry parsed**: `file_registry.json` loaded and indexed by (category, hash, era_version)
- [ ] **Test harness executable**: `comprehensive_test_harness.py` runs without paths errors
- [ ] **Era mapping ready**: All files tagged with version (v4, v5, v6, v7, v8, v9 or "mixed")
- [ ] **Monoliths catalogued**: All `*MONOLITH*.py` files identified and their era-contents mapped

**If any condition fails**: STOP. Report the blocker before proceeding.

---

## PART 2: HARD RULES (NON-NEGOTIABLE)

### Rule 1: Single Source of Truth Per Concept
**What it means**: For any concept (e.g., "E8 projection"), there is ONE canonical implementation, not 5 variants.

**How to enforce**:
- Search file_registry for all files mentioning concept
- Identify which version (v4, v5, …, v9) is authoritative (usually latest)
- Mark others as "legacy" or "reference"
- If v9 has no implementation, use v8; if no v8, use v7, etc.
- Delete/archive duplicates in `layer2_geometric` named `...(1).py`, `...(2).py`, etc.

**Violation penalty**: System will have contradictory behavior. HARD FAIL.

### Rule 2: Proofs Must Chain Unbroken
**What it means**: From λ_θ down to λ₀, every reduction step produces a proof. No step is "assumed valid."

**How to enforce**:
- Search for any function that returns bare result without proof
- Wrap with proof generation (use `provenance.py` patterns)
- Test: run `comprehensive_test_harness.py` with proof verification ON (it should pass)

**Violation penalty**: Invalid computation will be accepted. CRITICAL FAIL.

### Rule 3: No Hardcoded Paths
**What it means**: Runtime works anywhere. No `/home/ubuntu`, no absolute paths.

**How to enforce**:
- Search entire codebase for `/home/`, `C:\\`, `/Users/`
- Replace with `os.path.dirname(__file__)` or `pathlib.Path(...).resolve()`
- Test: `comprehensive_test_harness.py` should run from any directory

**Violation penalty**: Deployment fails on different machine. FAIL.

### Rule 4: Layer Boundaries Are Strict
**What it means**: Layer N code does NOT import from Layer N+1. Period.

**How to enforce**:
```python
# ALLOWED:
from layer1_morphonic import Atom
from layer2_geometric import E8Lattice
from layer3_operational import ConservationEnforcer

# FORBIDDEN:
from layer5_interface import SDK  # in layer3 code
```

- Search for imports that violate this
- Refactor into adapters or interfaces

**Violation penalty**: Circular dependencies, hard to reason about. FAIL.

### Rule 5: Every Behavior Must Be Testable
**What it means**: If the paper says it does X, there's a test that verifies X.

**How to enforce**:
- For each claim in docs (e.g., "E8 projection preserves lattice structure"):
  - Find or write a test in `tests/` or the harness
  - Verify it passes
- If test is missing, add it
- If test fails, debug until it passes

**Violation penalty**: Behavior is undocumented/untested. FAIL.

---

## PART 3: SCHEMA & AXIOMS (For Manus's Reasoning)

### File Classification Schema

Every file falls into ONE of these categories:

```yaml
MONOLITHS:
  description: "Complete frozen snapshot of an era"
  examples: ["CQE_CORE_MONOLITH.py", "CQE_V4_MONOLITH.py"]
  usage: "Reference only; extract specific functions, don't use as-is"
  pattern: "File size > 1MB typically"

CANONICAL_CODE:
  description: "Single authoritative implementation of a concept"
  examples: ["layer2_geometric/e8/e8_ops.py", "layer1_morphonic/overlay_system.py"]
  usage: "This is what gets deployed"
  pattern: "In a clean layer subpackage; passes tests"

SUBSYSTEM_APPS:
  description: "Self-contained subsystems (Aletheia, Scene8, CommonsLedger)"
  examples: ["aletheia_system/", "scene8/", "commons_ledger/"]
  usage: "Mount as sidecars or integrated apps"
  pattern: "Has setup.py, internal structure, independence"

LEGACY_REFERENCE:
  description: "Old implementation; kept for comparison"
  examples: ["geometric_toolkit (1).py", "lattice (1).py"]
  usage: "Study for learning; refactor into canonical"
  pattern: "Duplicates of canonical with different era markers"

TEST_HARNESS:
  description: "Validation + integration testing"
  examples: ["comprehensive_test_harness.py", "test_gnlc_complete.py"]
  usage: "Run to verify correctness end-to-end"
  pattern: "Executable, produces JSON reports"

DEPLOYMENT:
  description: "Ops + infrastructure"
  examples: ["Dockerfile", "kubernetes/", "install.sh"]
  usage: "For containerization and cloud"
  pattern: "Config-driven, no hardcoded paths"

DOCUMENTATION:
  description: "Behavioral spec + manual"
  examples: ["OPERATION_MANUAL.md", "DEPLOYMENT_README.md", "README.md"]
  usage: "Source of truth for what the system SHOULD do"
  pattern: "Markdown or PDF; cross-referenced with code"
```

### Concept Classification Axioms

Every concept (E8 projection, proof generation, GNLC compilation) has:

1. **Specification** (in docs: what it should do)
2. **Implementation** (in code: how it does it)
3. **Test** (verification: does it match spec?)
4. **Proof** (cryptographic: is it correct?)

**For Manus**:
- If spec exists but no code → FLAG: "Implementation gap"
- If code exists but no test → FLAG: "Testing gap"
- If test fails → FLAG: "Correctness issue"
- If proof missing → FLAG: "Auditability gap"

---

## PART 4: INTEGRATION LEMMAS (Theorems Manus Relies On)

### Lemma 1: Era Continuity
```
If v4 has concept C and v5 has concept C', and C' is described as 
"v4's C with refinement R", then:
  - v5(C') must pass all of v4(C)'s tests (backward compatibility)
  - v5(C') must pass new tests for R
  - v4(C) can be archived (not deleted)
```

**Implication**: When Manus finds old and new versions, it should:
1. Run v4's tests with v5's code (should still pass)
2. Run v5-specific tests (should pass)
3. If either fails, debug the evolution

### Lemma 2: Proof Transitivity
```
If proof(A → B) and proof(B → C), then proof(A → C) exists.
```

**Implication**: Manus can trace computation chains backward from output to input using proofs. This is how auditability works.

### Lemma 3: Layer Independence
```
Layer N's correctness does NOT depend on Layer N+1.
Layer N's implementation does NOT reference Layer N+1 code.
```

**Implication**: When building the runtime, Manus can test Layer 1 independently of Layer 5. Bugs don't cascade up.

### Lemma 4: Geometric Invariance
```
For any lattice L, any valid embedding E into L, and any operation Op on E:
  - Op(E) is also in L (geometry is preserved)
  - Op(E) can be verified cryptographically (proof system works)
```

**Implication**: When Manus processes new code in layer2_geometric, it must verify that operations don't break lattice structure.

### Lemma 5: Test Completeness
```
If comprehensive_test_harness.py passes, then:
  - All layers load correctly
  - E8 and Leech lattices initialize
  - GNLC compilation chain works
  - Proofs verify
  - Governance rules enforce
  - Deployment patterns work
```

**Implication**: Pass the harness = system is buildable + deployable.

---

## PART 5: MANUS'S SPECIFIC TASKS

### Phase 1: Architecture Audit (Days 1-2)

**Input**: 10,000+ files from archive

**Task 1.1**: Categorize all files
```
Query file_registry.json:
  → Count files by category (monoliths, canonical_code, legacy, etc.)
  → Count by era (v1-v3, v4, v5, v6, v7, v8, v9)
  → List all subsystems (aletheia, scene8, commons_ledger, etc.)

Output: Inventory spreadsheet with columns:
  [file_path] [category] [era] [size_kb] [status] [notes]
```

**Task 1.2**: Identify all concepts and their implementations
```
For each major concept (E8_projection, proof_generation, GNLC_λ₀, etc.):
  → Find all files that implement it
  → Identify which is canonical (usually latest version that works)
  → Mark others as legacy
  → Check if tests exist

Output: Concept-to-implementation map:
  {
    "E8_projection": {
      "canonical": "layer2_geometric/e8/e8_ops.py",
      "legacy": ["geometric_toolkit (1).py", "legacy_e8.py"],
      "test": "tests/test_e8_projection.py",
      "eras": ["v4", "v5", "v6", "v7", "v8", "v9"]
    },
    ...
  }
```

**Task 1.3**: Flag gaps and inconsistencies
```
For each concept:
  → Does spec exist in docs? YES/NO
  → Does code exist? YES/NO
  → Does test exist? YES/NO
  → Do they agree? YES/NO

Output: Gap report:
  - Missing implementations (code exists, needs tests)
  - Missing tests (code exists, spec exists, no test)
  - Specification drift (code doesn't match docs)
  - Era inconsistencies (v5 claims compatibility with v4, but tests fail)
```

### Phase 2: Proof System Construction (Days 3-4)

**Task 2.1**: Verify proof chain completeness
```
Check every operation in every layer:
  @layer = 1 → 5:
    for each function F in layer[i]:
      if F returns result:
        - Does it also return proof? YES/NO
        - Does proof satisfy provenance.py format? YES/NO
        - Can proof be verified by layer[i+1]? YES/NO
      if NO to any:
        → FLAG: "Proof gap in {function} at layer {i}"

Output: Proof audit report with fixes
```

**Task 2.2**: Construct proof tree visualization
```
For each test in comprehensive_test_harness:
  → Trace computation from input → output
  → Collect all proofs generated
  → Build tree showing proof chain

Output: SVG/JSON proof tree for each test case
  Helps verify: proof chain is unbroken, cryptographically sound
```

### Phase 3: Layered Integration (Days 5-7)

**Task 3.1**: Build Layer 1 canonical
```
From layer1_morphonic/:
  1. Identify all atomic operations (atom creation, overlay, morphon, etc.)
  2. Find canonical implementations (latest working version)
  3. Deduplicate any `..._(1).py`, `..._(2).py` variants
  4. Ensure all return (result, proof)
  5. Run Layer 1 tests in isolation

Output: Clean layer1_morphonic/ with:
  - No duplicates
  - All operations proof-enabled
  - Tests pass
  - All paths relative (no /home/ubuntu)
```

**Task 3.2**: Build Layer 2 canonical (Hardest)
```
From layer2_geometric/:
  1. Identify canonical: e8/, leech/, niemeier/, weyl/ subpackages
  2. Move/deduplicate: geometric_toolkit(1..5).py → e8/operations.py
  3. Move/deduplicate: lattice(1).py, lattice.py → leech/lattice.py
  4. Ensure E8 and Leech tests pass
  5. Verify lattice invariants preserved

Output: Clean layer2_geometric/ with:
  - Only 4 subpackages (e8, leech, niemeier, weyl)
  - All operations tested
  - Lattice structure verified
```

**Task 3.3**: Build Layer 3 canonical
```
From layer3_operational/:
  1. Identify: conservation.py, morsr.py, toroidal.py, phi_metric.py
  2. Verify each calls Layer 1/2 correctly
  3. Ensure ΔΦ conservation enforced
  4. Verify all operations return (result, proof)

Output: Working layer3_operational/ that passes ops tests
```

**Task 3.4**: Build Layer 4 canonical
```
From layer4_governance/:
  1. Canonical: gravitational.py, seven_witness.py, policy_hierarchy.py, validation_framework.py
  2. Verify SevenWitness 7-fold checks work
  3. Verify digital root stratification
  4. Verify policy channels enforce constraints

Output: Working layer4_governance/ that enforces all rules
```

**Task 3.5**: Build Layer 5 canonical
```
From layer5_interface/:
  1. Canonical: gnlc_lambda0.py … gnlc_lambda_theta.py
  2. Canonical: gnlc_reduction.py, gnlc_type_system.py
  3. Build OperatingSystem.py wrapper that orchestrates Layers 1-4
  4. Build SDK that exposes public API
  5. Verify GNLC compilation chain works

Output: Working layer5_interface/ with:
  - Full GNLC stack (λ₀ → λ_θ)
  - Type system functional
  - Reduction/normalization working
  - SDK accessible
```

### Phase 4: Integration Testing (Days 8-9)

**Task 4.1**: Run comprehensive harness
```
python comprehensive_test_harness.py

Expected: rc=0, TEST_REPORT.json shows:
  ✓ Layer 1 tests pass
  ✓ Layer 2 tests pass
  ✓ Layer 3 tests pass
  ✓ Layer 4 tests pass
  ✓ Layer 5 tests pass
  ✓ GNLC compilation works
  ✓ Proofs verify
  ✓ Governance enforces

If ANY fails: debug and fix before proceeding
```

**Task 4.2**: Run deployment tests
```
For each deployment target (Docker, K8s, cloud):
  1. Apply dockerfile/template
  2. Verify no hardcoded paths
  3. Verify comprehensive_test_harness.py runs in container
  4. Verify output is identical to local run

Output: Deployment validation report
```

### Phase 5: Documentation & Handoff (Days 10)

**Task 5.1**: Produce final integration report
```
Document:
  - Architecture decisions (which implementations became canonical)
  - Proof system verification (complete chains documented)
  - Test coverage (% of codebase covered)
  - Performance benchmarks (if available from test harness)
  - Known limitations (if any)
```

**Task 5.2**: Produce deployment guide
```
Document how to:
  - Load the unified runtime
  - Accept custom models
  - Run with different inputs
  - Verify outputs + proofs
  - Deploy to different targets
```

---

## PART 6: FINAL DELIVERABLE SPEC

After Manus completes all tasks, you have:

### 1. Unified Runtime (Python Package)
```
cqe_unified_runtime/
├── layer1_morphonic/       ← Canonical, deduplicated, proof-enabled
├── layer2_geometric/       ← E8, Leech, Niemeier, Weyl (clean)
├── layer3_operational/     ← Conservation, MORSR, toroidal (working)
├── layer4_governance/      ← Gravitational, SevenWitness, policy (enforcing)
├── layer5_interface/       ← GNLC, SDK, OperatingSystem (complete)
├── subsystems/             ← Aletheia, Scene8, CommonsLedger
├── tests/                  ← Full test coverage
├── deployment/             ← Docker, K8s, cloud templates
├── setup.py                ← Package configuration
└── runtime.py              ← Entry point
```

### 2. Loadable AI Model
- Accepts input vectors (any format → E8 embedding)
- Processes through 5-layer stack
- Returns output + cryptographic proof
- Verifiable by any party (proof validation is deterministic)

### 3. Deployable Everywhere
- Docker image: `docker run cqe:v9 --input model.json`
- Kubernetes: `kubectl apply -f deployment/kubernetes/cqe-pod.yaml`
- Local: `python runtime.py --model custom.pkl`
- Cloud: AWS/Azure/GCP templates pre-built

### 4. Fully Tested & Audited
- `comprehensive_test_harness.py` passes (all layers verified)
- Proof chains validated (cryptographic soundness)
- Governance enforced (no invalid computation)
- Documentation complete (architecture + deployment guide)

---

## PART 7: MANUS'S ERROR RECOVERY

If Manus encounters these errors:

| Error | Action |
|-------|--------|
| "File not found in registry" | Search by hash/name in file_registry.json; if missing, flag as extraction gap |
| "Concept implemented in 3 places" | Run comprehensive_test_harness with each; mark slowest as legacy, fastest as canonical |
| "Proof returns None" | Wrap with `provenance.create_proof(result)`; test passes now? If yes, commit. If no, debug. |
| "Layer N imports Layer N+1" | Refactor: extract interface into Layer 1-N adapter; move implementation to Layer N+1 |
| "Test fails on different machine" | Find hardcoded paths; replace with `pathlib.Path(__file__).parent.resolve()` |
| "Monolith is 3MB, can't fit in memory" | Extract specific functions you need; don't load entire monolith |
| "v4 test fails with v9 code" | Backward compatibility broken; investigate what changed; add adapter if needed |

---

## PART 8: SUCCESS CRITERIA (How You Know It Worked)

Manus succeeds when:

✅ All files categorized and era-tagged  
✅ All concepts have canonical implementation + test  
✅ No duplicates remain  
✅ Layer boundaries strict (no cross-layer shortcuts)  
✅ All proofs chain unbroken from λ_θ to λ₀  
✅ No hardcoded paths anywhere  
✅ comprehensive_test_harness.py passes (rc=0)  
✅ Deployment templates work (Docker/K8s/cloud)  
✅ Performance benchmarks meet spec (if specified in docs)  
✅ You can load it, pass it data, get back result + proof ✓

When all 10 boxes are checked: **System is buildable, testable, deployable, and ready for production use.**

---

## APPENDIX A: File_Registry Query Templates

Manus can use these queries to navigate the archive:

```python
# Find all files for a concept (e.g., "E8")
files = [f for f in registry if "E8" in f["organized_path"].upper()]

# Find all files in an era
v9_files = [f for f in registry if "v9" in f["organized_path"] or "v9" in f["original_path"]]

# Find largest files (potential monoliths)
large_files = sorted(registry, key=lambda f: f["size"], reverse=True)[:20]

# Find tests
tests = [f for f in registry if "test" in f["organized_path"].lower()]

# Find by category
monoliths = [f for f in registry if "MONOLITHS" in f["organized_path"]]
code = [f for f in registry if "python_code" in f["category"]]
```

---

## APPENDIX B: Proof Format Specification

All proofs must follow this schema:

```json
{
  "operation": "string (function name)",
  "input_hash": "sha256(input)",
  "output_hash": "sha256(output)",
  "timestamp": "ISO-8601",
  "layer": "1-5",
  "verification_algorithm": "sha256|blake2|ed25519",
  "proof_signature": "hex(sign(output_hash))",
  "metadata": {
    "lattice_preserved": boolean,
    "conservation_satisfied": boolean,
    "governance_passed": boolean
  }
}
```

---

End of Protocol. Begin work.
