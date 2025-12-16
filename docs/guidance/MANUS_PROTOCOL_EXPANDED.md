# MANUS PROTOCOL - EXPANDED
## Detailed Integration Workflow with Ordering & Decision Rules

**For**: Manus  
**Scope**: Complete workflow from inventory through deployment  
**Length**: ~80 pages of detailed guidance  
**Outcome**: Functional system with all 5 layers working

---

## PART 1: PRE-INTEGRATION SETUP (Phase 0)

### Step 1: Load and Index File Registry

```python
import json
registry = json.load(open("file_registry.json"))

# Index by various keys
by_category = {}
for file in registry:
    cat = file.get("category", "other")
    if cat not in by_category:
        by_category[cat] = []
    by_category[cat].append(file)

# Print inventory
for cat, files in sorted(by_category.items()):
    print(f"{cat}: {len(files)} files")
```

**Output**: Understand what file types exist and quantity.

### Step 2: Pre-Integration Checklist

- [ ] File registry loads without errors
- [ ] At least 10 Python code files exist
- [ ] Test files exist
- [ ] Documentation files exist
- [ ] MVP is identified and runs

**If any fail**: Escalate blocker before proceeding.

---

## PART 2: PHASE 1 - INVENTORY & CATEGORIZATION (2 days)

### Goal
Map every file to a category + layer + concept + completeness level.

### Step 1: Categorize by File Type

```
File types to identify:

MONOLITHS (>500KB single Python files)
  └─ Action: Reference only; extract functions, don't deploy

CANONICAL_CODE (clean implementations in organized directories)
  └─ Action: These become deployed runtime

SUBSYSTEMS (self-contained apps like Aletheia, Scene8)
  └─ Action: Mount as integrated apps

TEST_HARNESS (comprehensive test runners)
  └─ Action: Run after each integration phase

UNIT_TESTS (single-concept tests)
  └─ Action: Run layer-by-layer

DOCUMENTATION (spec + manual)
  └─ Action: Read to understand behavior

DEPLOYMENT (Docker/K8s/cloud templates)
  └─ Action: Test at end

DATA_FILES (JSON, CSV, YAML configs)
  └─ Action: Extract configs, merge duplicates

LEGACY (old implementations, v1-v4 era)
  └─ Action: Archive for reference
```

**Task**: For each file, assign primary category.

### Step 2: Categorize by Layer

```
Layer 1: atoms, lambda, overlay, provenance, proof, acceptance
Layer 2: e8, leech, niemeier, weyl, geometry, lattice, projection
Layer 3: conservation, morsr, toroidal, phi, language, state, operation
Layer 4: gravitational, seven_witness, policy, validation, governance
Layer 5: gnlc, sdk, interface, api, cli, ui, operating_system
Subsystem: aletheia, scene8, commons_ledger, speedlight
```

**Task**: For each code file, assign primary layer.

### Step 3: Categorize by Concept

```
Major concepts:

LAYER 1:
  - Atom creation + manipulation
  - Lambda term evaluation
  - Proof generation + verification
  - Overlay system operations

LAYER 2:
  - E8 lattice operations (240 roots)
  - Leech lattice operations (196,560 roots)
  - Niemeier-24 lattice specifications
  - Projection algorithms
  - Lattice validators

LAYER 3:
  - State transitions
  - Conservation enforcement (ΔΦ)
  - MORSR exploration
  - Toroidal evolution
  - Phi metric scoring

LAYER 4:
  - Digital root stratification (DR0-DR9)
  - Seven witness validation
  - Policy channels
  - Formal validation framework

LAYER 5:
  - Lambda calculus compilation (λ₀→λ₁→λ₂→λ_θ)
  - SDK (public API)
  - CLI interface
  - Web API
```

**Task**: For each code file, assign primary concept.

### Step 4: Identify Duplicates & Variants

```bash
# Find similar filenames
ls -la layer2_geometric/*.py | awk '{print $NF}' | sort | uniq -d

# Find similar content
for file1 in *.py; do
  for file2 in *.py; do
    if [ "$file1" != "$file2" ]; then
      diff -q "$file1" "$file2" && echo "DUPLICATE: $file1 vs $file2"
    fi
  done
done
```

**Output**: List of duplicate/variant files:
```
Concept: E8_Projection
  - Implementation A: geometric_toolkit.py (1000 lines, 10 tests)
  - Implementation B: e8_ops.py (300 lines, 15 tests)
  - Implementation C: legacy_e8.py (archived, 200 lines, 0 tests)
```

---

## PART 3: PHASE 2 - LAYER 1 ASSEMBLY (2 days)

### Goal
Build atomic foundation. Layer 1 has NO imports from other layers.

### Step 1: Identify All Layer 1 Implementations

Search file registry:
```python
layer1_files = [f for f in registry if "layer1" in f["organized_path"].lower() or 
                                        any(x in f["organized_path"] for x in 
                                        ["atom", "lambda", "proof", "overlay", "provenance"])]
```

**Output**: List all Layer 1 candidates.

### Step 2: Concept-by-Concept Selection (Layer 1)

For EACH concept in Layer 1, apply selection algorithm:

#### Concept: Atom Creation

**Candidates**:
```
Candidate A: layer1_morphonic/cqe_atom.py
  - Completeness: Creates atoms + operations (90%)
  - Testing: 8 tests, 7 pass (87%)
  - Simplicity: Clear, 200 lines
  - Score: 0.9*0.40 + 0.87*0.40 + 0.85*0.20 = 0.876

Candidate B: atom_legacy.py (archived)
  - Completeness: Basic atom only (60%)
  - Testing: 3 tests, 2 pass (67%)
  - Simplicity: 150 lines but dense
  - Score: 0.6*0.40 + 0.67*0.40 + 0.70*0.20 = 0.658

Candidate C: morphonic_atom.py
  - Completeness: Full + edge cases (95%)
  - Testing: 12 tests, 10 pass (83%)
  - Simplicity: 400 lines, complex
  - Score: 0.95*0.40 + 0.83*0.40 + 0.60*0.20 = 0.832
```

**Decision**: Candidate A scores highest (0.876).  
**Action**: Use Candidate A. Archive B and C for reference.

**Repeat for**:
- Lambda term evaluation
- Proof generation
- Overlay system
- Acceptance rules
- Other Layer 1 concepts

### Step 3: Build Layer 1 Directory

```
layer1_morphonic/
├── __init__.py
├── cqe_atom.py                 ← chosen implementation
├── lambda_term.py              ← chosen implementation
├── overlay_system.py           ← chosen implementation
├── provenance.py               ← chosen implementation
├── acceptance_rules.py         ← chosen implementation
├── shell_protocol.py           ← chosen or merged
├── tests/
│   ├── test_atom.py
│   ├── test_lambda.py
│   ├── test_overlay.py
│   ├── test_provenance.py
│   └── test_acceptance.py
└── archive/
    ├── atom_legacy.py          ← not used; kept for reference
    └── morphonic_atom_v2.py    ← not used; kept for reference
```

### Step 4: Verify Layer 1 Independence

```bash
# Check: no imports from layers 2-5
grep -r "from layer[2-5]" layer1_morphonic/ --include="*.py" | grep -v "^#"
# Should return: empty

# Check: all files have __all__ or clear exports
grep -r "__all__" layer1_morphonic/ --include="*.py"
# Should show what's exported
```

### Step 5: Test Layer 1

```bash
python -m pytest layer1_morphonic/tests/ -v --tb=short

# Expected: All tests pass
# If any fail: debug and fix before proceeding
```

**Checkpoint 1 Checklist**:
- [ ] All Layer 1 concepts mapped to implementations
- [ ] Chosen implementations selected via scoring
- [ ] Layer 1 code organized in clean directory
- [ ] No imports from layers 2-5
- [ ] All Layer 1 tests pass

---

## PART 4: PHASE 3 - LAYER 2 ASSEMBLY (3 days)

### Goal
Build geometric engine. Most complex layer. Layer 2 imports only from Layer 1.

### Step 1: Identify Layer 2 Concepts & Sub-Dependencies

Layer 2 has 4 sub-concepts with ordering:

```
Priority 1 (Foundation):
  ├─ E8 lattice (240 roots) - all other geometries reference this
  └─ Leech lattice (196,560 roots) - depends on E8

Priority 2 (Extensions):
  ├─ Niemeier-24 lattices - depends on Leech
  └─ Weyl chamber operations - depends on E8

Priority 3 (Validation):
  └─ Geometry validators - depends on all above
```

**Important**: You MUST integrate in this order. Do not skip around.

### Step 2: Integrate E8 First

```python
# Find all E8 implementations
e8_files = [f for f in registry if "e8" in f["organized_path"].lower() or 
                                   "e8" in f["original_path"].lower()]

# Score each implementation
# Completeness: Does it have roots? projections? validators?
# Testing: How many tests? Pass rate?
# Simplicity: Readability?
```

**Selection criteria for E8**:
- Must have: 240 root definitions
- Must have: projection algorithm
- Must have: lattice validation
- Should have: tests that verify roots are valid

**Action**: Choose best E8 implementation, integrate into layer2_geometric/e8/

### Step 3: Integrate Leech Second

```python
# Find Leech implementations
leech_files = [f for f in registry if "leech" in f["organized_path"].lower()]

# Scoring INCLUDES: Does it depend on E8? Can it import E8?
# If Leech implementation imports non-existent E8 code → reject
```

**Selection criteria for Leech**:
- Must have: 24D structure definition
- Must have: E8→Leech embedding algorithm
- Should reference: E8 implementation for embedding basis
- Must pass: tests verifying Leech properties

**Action**: Choose best Leech, integrate into layer2_geometric/leech/

### Step 4: Integrate Niemeier-24 Third

**Only after E8 + Leech work.**

```
layer2_geometric/
├── e8/
│   ├── roots.py              (240 E8 roots)
│   ├── operations.py         (projection, etc.)
│   ├── weyl.py               (Weyl chamber ops)
│   └── tests/
├── leech/
│   ├── lattice.py            (24D Leech)
│   ├── embeddings.py         (E8→Leech)
│   └── tests/
├── niemeier/
│   ├── specs.py              (24 Niemeier specs)
│   ├── generator.py          (generation algorithm)
│   └── tests/
├── validators/
│   ├── lattice_invariant.py
│   └── tests/
└── tests/
    └── test_integration.py   (E8+Leech+Niemeier together)
```

### Step 5: Cross-Layer Testing

```bash
# Layer 1 should still pass
python -m pytest layer1_morphonic/tests/ -v

# Layer 2 should pass
python -m pytest layer2_geometric/tests/ -v

# Layer 2 should import from Layer 1
python -c "from layer2_geometric import e8; print(e8.__name__)"

# Layer 2 should NOT import from Layer 3+
grep -r "from layer[3-5]" layer2_geometric/ --include="*.py"
# Should return: empty
```

**Checkpoint 2 Checklist**:
- [ ] E8 integrated and tests pass
- [ ] Leech integrated and tests pass
- [ ] Niemeier integrated and tests pass
- [ ] Validators integrated and tests pass
- [ ] All Layer 2 tests pass
- [ ] Layer 2 only imports from Layer 1
- [ ] Layer 2 exports clean API

---

## PART 5: PHASE 4 - LAYER 3 ASSEMBLY (2 days)

### Goal
Operational semantics. Layer 3 handles state transitions + constraints.

### Sub-Concept Ordering for Layer 3

```
Priority 1 (State):
  └─ State definition + transitions

Priority 2 (Validation):
  ├─ Conservation enforcer (ΔΦ checking)
  └─ Constraint validators

Priority 3 (Exploration):
  ├─ MORSR algorithm
  └─ Toroidal evolution

Priority 4 (Metrics):
  └─ Phi metric scoring
```

### Integration Process (Same as Layer 2)

1. **Identify candidates** for each sub-concept
2. **Score** each (completeness + testing + simplicity)
3. **Integrate in priority order** (state first, metrics last)
4. **Test after each** (don't wait until end)
5. **Verify imports** (Layer 3 → Layer 1-2 only)

### Layer 3 Output

```
layer3_operational/
├── state.py                  (state definition)
├── transitions.py            (transition logic)
├── conservation.py           (ΔΦ enforcement)
├── morsr.py                  (MORSR explorer)
├── toroidal.py               (cyclic evolution)
├── phi_metric.py             (scoring)
├── language_engine.py        (semantic integration)
└── tests/
    ├── test_state.py
    ├── test_conservation.py
    ├── test_morsr.py
    ├── test_toroidal.py
    └── test_integration.py
```

**Checkpoint 3**: All Layer 3 tests pass; Layer 1-2 tests still pass.

---

## PART 6: PHASE 5 - LAYER 4 ASSEMBLY (2 days)

### Goal
Governance + validation. Layer 4 enforces rules + performs multi-witness checks.

### Sub-Concept Ordering

```
Priority 1 (Foundation):
  └─ Digital root stratification (DR0-DR9 levels)

Priority 2 (Validation):
  ├─ Seven witness framework (independent validators)
  └─ Validation aggregation

Priority 3 (Governance):
  ├─ Policy hierarchy (channels, bounds)
  └─ Formal validation framework
```

### Key Requirement

Layer 4 receives states from Layer 3 and either:
- ✓ Validates + passes through (returns same state + validation proof)
- ✗ Rejects + returns error proof (does NOT modify state)

Layer 4 is READ-ONLY on state.

### Integration Process

Same as Layer 3, but with special attention to:
- Seven witnesses must be independent implementations
- Each witness must have its own test
- All seven must reach agreement

---

## PART 7: PHASE 6 - LAYER 5 ASSEMBLY (2 days)

### Goal
User interface. Lambda calculus compilation. Public API.

### Sub-Concept Ordering

```
Priority 1 (Compilation):
  └─ Lambda 0 (atomic, direct E8 execution)

Priority 2 (Calculi):
  ├─ Lambda 1 (relations, structure)
  ├─ Lambda 2 (state, toroidal)
  └─ Lambda theta (meta, learning)

Priority 3 (Interface):
  ├─ SDK (public API)
  ├─ CLI (command-line)
  └─ Web API (HTTP)

Priority 4 (Subsystems):
  ├─ Aletheia integration
  ├─ Scene8 integration
  └─ CommonsLedger integration
```

---

## PART 8: PHASE 7 - INTEGRATION TESTING (1 day)

### Goal
Verify all 5 layers work together.

### Comprehensive Test Harness

Run the master test:
```bash
python comprehensive_test_harness.py
```

Expected output:
```
Layer 1 tests: PASS (12/12)
Layer 2 tests: PASS (18/18)
Layer 3 tests: PASS (15/15)
Layer 4 tests: PASS (20/20)
Layer 5 tests: PASS (14/14)
Integration tests: PASS (25/25)
Proof verification: PASS (all chains valid)
Governance checks: PASS (invalid states rejected)

OVERALL: ✓ ALL TESTS PASS
```

If ANY test fails: debug and fix before proceeding.

---

## PART 9: PHASE 8 - DEPLOYMENT (1 day)

Test deployment to 3 targets:

### Docker
```bash
docker build -t cqe:latest .
docker run cqe:latest python runtime.py --info
# Should work identically to local
```

### Kubernetes
```bash
kubectl apply -f deployment/kubernetes/
kubectl exec cqe-pod -- python runtime.py --test
# Should work
```

### Cloud (pick one: AWS/Azure/GCP)
```bash
# Deploy using template
# Test endpoint
# Verify results match local
```

---

## PART 10: DECISION MATRIX FOR SCORING IMPLEMENTATIONS

When you have multiple implementations of same concept:

```python
def score_implementation(candidate):
    completeness = assess_completeness(candidate)  # 0-1
    testing = assess_testing(candidate)            # 0-1
    simplicity = assess_simplicity(candidate)      # 0-1
    
    score = (0.40 * completeness + 
             0.40 * testing + 
             0.20 * simplicity)
    return score

def assess_completeness(candidate):
    """Does it cover all required functionality?"""
    has_core_functions = all(func in candidate.functions for func in REQUIRED_FUNCS)
    has_edge_cases = count_edge_case_handling(candidate) > 5
    has_documentation = candidate.has_docstrings
    
    # Return 0-1 score
    score = 0
    if has_core_functions: score += 0.5
    if has_edge_cases: score += 0.3
    if has_documentation: score += 0.2
    return min(score, 1.0)

def assess_testing(candidate):
    """Is it well-tested?"""
    test_count = len(candidate.tests)
    pass_rate = candidate.tests_passed / test_count if test_count > 0 else 0
    coverage = analyze_code_coverage(candidate)
    
    score = 0.3 * min(test_count / 10, 1.0)  # 0-10 tests max
    score += 0.4 * pass_rate                   # 100% pass is 0.4
    score += 0.3 * min(coverage / 0.8, 1.0)  # 80% coverage is max
    return score

def assess_simplicity(candidate):
    """Is it readable and maintainable?"""
    lines = count_lines(candidate)
    complexity = cyclomatic_complexity(candidate)
    dependencies = count_external_imports(candidate)
    
    score = 1.0
    if lines > 500: score -= 0.2
    if complexity > 10: score -= 0.3
    if dependencies > 5: score -= 0.15
    return max(score, 0)
```

**Tiebreaker (if two score the same)**:
1. Which has more recent last-modified date?
2. Which is more readable (lower complexity)?
3. Which is in official layer directory (not archived)?

---

## PART 11: CHECKPOINT SYSTEM & ROLLBACK

After each phase, create checkpoint:

```bash
# After Phase N complete, save state
git add .
git commit -m "Phase N complete: Layer X integration finished"
git tag "checkpoint-phase-$N"
```

If Phase N+1 breaks everything:
```bash
# Rollback to last known good
git checkout checkpoint-phase-N
# Fix the issue
# Re-attempt Phase N+1
```

---

## PART 12: CONFLICT RESOLUTION

If two implementations are incompatible:

```
Incompatibility: Implementation A generates proof_v1, Implementation B expects proof_v2

Option 1: Use A + update B to understand proof_v1
Option 2: Use B + update A to generate proof_v2
Option 3: Build adapter layer that converts between proof versions
Option 4: Choose third implementation that handles both

Decision: Based on:
- Which requires least code changes?
- Which is most robust?
- Which is most tested?
```

---

End of expanded protocol. You now have:
- Clear ordering for each layer
- Scoring algorithm with examples
- Checkpoint system
- Conflict resolution
- Concrete steps for each phase

Proceed with confidence.
