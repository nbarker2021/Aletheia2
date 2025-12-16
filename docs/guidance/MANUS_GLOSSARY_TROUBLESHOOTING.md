# MANUS TECHNICAL GLOSSARY & TROUBLESHOOTING
## Concepts, Definitions, and Problem-Solving Trees

---

## PART 1: TECHNICAL GLOSSARY

### Geometric Concepts

**Lattice**
- Definition: A discrete, regularly-spaced geometric structure
- In CQE: Mathematical foundation for all computation
- E8 lattice: 240-dimensional structure with 240 root vectors
- Leech lattice: 24-dimensional structure with 196,560 roots
- Why it matters: All valid states must be lattice points; invalid states violate geometry
- Example: "The projection must preserve lattice structure" = output must be a valid lattice point

**Root Vector**
- Definition: A fundamental direction in lattice space
- E8 has 240 roots; Leech has 196,560 roots
- Why it matters: Operations move between roots; not arbitrary positions
- How it works: A + B might not be valid unless (A+B) lands on a lattice root

**Projection**
- Definition: Mapping from higher-dimensional space to lower-dimensional lattice
- Example: 248D ambient space → 240D E8 lattice
- Why it matters: Ensures data fits into valid geometric structure
- How it fails: Projection that doesn't preserve lattice structure is invalid

**Embedding**
- Definition: Mapping from one lattice to another (not necessarily lower dimension)
- Example: E8 → Leech (8D into 24D)
- Why it matters: Connects different geometric contexts
- How it works: Each E8 root maps to a specific Leech root or region

---

### Proof System Concepts

**Proof**
- Definition: Cryptographic evidence that an operation is correct
- Format: JSON with operation, input_hash, output_hash, layer, signature
- Why it matters: Allows verification without re-running computation
- How it works: Verifier checks hash chain and signature

**Proof Chain**
- Definition: Linked sequence of proofs from Layer 1 through Layer N
- Layer 1 proof → Layer 2 proof (references L1) → Layer 3 proof (references L2) → ...
- Why it matters: Complete audit trail from input to output
- How it fails: Broken link means verification fails

**Hash**
- Definition: Deterministic fixed-length encoding of data
- SHA256: Most common in this system (256-bit output)
- Why it matters: Same input always produces same hash; different input almost never produces same hash
- How it's used: Proof verification relies on hash chain being unbroken

**Signature**
- Definition: Cryptographic proof that a specific actor created this proof
- Ed25519: Elliptic curve signature algorithm used here
- Why it matters: Proves proof hasn't been tampered with
- How it fails: If signature is invalid, proof is forged

**Parent Proof Hash**
- Definition: Hash of the previous layer's proof
- Example: Layer 2 proof contains hash of Layer 1 proof
- Why it matters: Links proofs together; breaking this link breaks the chain
- How to check: `verify(current_proof.parent_hash == hash(previous_proof))`

---

### Operational Concepts

**State**
- Definition: Current configuration of the system
- What it contains: Current values, constraints, governance status
- Why it matters: Operations transform one state into another
- How it moves: Layer 1 → Layer 2 → Layer 3 → ... (each layer processes and passes it forward)

**State Transition**
- Definition: Legal move from one state to another
- What makes it legal: Governed by Layer 3 rules (conservation, constraints)
- Why it matters: Not all moves are valid; only transitions that preserve invariants
- How it's checked: Layer 3 validates transition, Layer 4 validates final state

**Conservation**
- Definition: Principle that certain quantities cannot be created or destroyed
- ΔΦ metric: "Phi change" must be bounded
- Why it matters: Prevents infinite loops or unbounded growth
- Example: "You can't increase Phi by more than 0.5 per operation"

**DigitalRoot (DR)**
- Definition: Hierarchical anchoring system (DR0 = ground, DR9 = highest)
- Purpose: Stratification of validity levels
- Why it matters: Some operations only valid at certain DR levels
- How it works: Every state has a DR level; transitions must respect level rules

---

### System Architecture Concepts

**Layer**
- Definition: One of 5 horizontal strata in the system
- Layer 1: Atoms and proofs (foundation)
- Layer 2: Geometry (lattices, embeddings)
- Layer 3: Operations (movement, constraints)
- Layer 4: Governance (rules, validation)
- Layer 5: Interface (user-facing, API)
- Why layering: Each layer depends only on lower layers; enables isolation and testing

**Atom**
- Definition: Smallest indivisible unit of computation
- What it contains: Value + metadata
- Why it matters: Layer 1 works with atoms; can't go smaller
- How it's used: Atoms are wrapped in overlays, processed through layers

**Overlay**
- Definition: Container that wraps an atom with additional metadata
- What it adds: Proof, state information, constraints
- Why it matters: Atoms alone aren't enough; overlays track computation
- How it works: Atom moves through system inside overlay

**Proof Object**
- Definition: Structured data containing proof information
- Fields: operation, input_hash, output_hash, layer, signature, metadata
- Why it matters: Standardized format allows chaining and verification
- How it's created: Each layer generates one when completing an operation

---

### Lambda Calculus Concepts

**Lambda 0 (λ₀)**
- Definition: Atomic calculus—direct execution in E8 overlays
- What it compiles to: Operations on single atoms
- Why it's useful: Base case for reduction
- Example: `λx. project_to_e8(x)` → directly executable

**Lambda 1 (λ₁)**
- Definition: Relation calculus—operations on multiple atoms
- What it compiles to: λ₀ operations composed together
- Why it's useful: Can express relationships and compositions
- Example: `map(λ₀_op, list_of_atoms)` → chains λ₀ operations

**Lambda 2 (λ₂)**
- Definition: State calculus—operations over time/state
- What it compiles to: Sequences of λ₁ operations
- Why it's useful: Can express stateful transformations
- Example: `loop: λ₁_op; transition; λ₁_op` → stateful execution

**Lambda Theta (λ_θ)**
- Definition: Meta-calculus—self-modifying operations
- What it compiles to: Operations that can rewrite themselves
- Why it's useful: Learning, adaptation, meta-programming
- Example: `op.rewrite_if(condition)` → modifies itself based on state

**Compilation**
- Definition: Process of reducing higher-level lambda to lower-level
- Flow: λ_θ → λ₂ → λ₁ → λ₀ → E8 execution
- Why it matters: Type checking and verification happen at each step
- How to verify: Each step should produce a proof that compilation succeeded

---

## PART 2: COMMON PROBLEMS & SOLUTION TREES

### Problem: "Tests Pass But Proofs Don't Verify"

```
Test passes (no assertion error)
├─ But proof verification fails
│  ├─ Is the proof hash correct?
│  │  ├─ Manual check: compute sha256(proof_json) === reported_hash?
│  │  ├─ NO → Proof was modified after generation
│  │  │    └─ Solution: Check that proof is not being modified
│  │  └─ YES → Proceed to next check
│  ├─ Is the parent proof hash linked?
│  │  ├─ Manual check: previous_proof_hash === current_proof.parent_hash?
│  │  ├─ NO → Proof chain is broken
│  │  │    └─ Solution: Ensure each layer passes proof to next layer
│  │  └─ YES → Proceed to next check
│  └─ Is the signature valid?
│     ├─ Try: verify_proof_signature(proof, public_key)
│     ├─ INVALID → Signature was forged or key is wrong
│     │    └─ Solution: Check if public key matches proof signer
│     └─ VALID → Unexpected; escalate
└─ Resolution: Fix the broken step identified above, re-run test
```

---

### Problem: "Layer 2 Tests Pass But Integration Tests Fail"

```
Layer 2 alone: PASS
Layer 1+2 together: FAIL
├─ Is it a proof chaining issue?
│  ├─ Check: Does L2 accept proof from L1?
│  ├─ NO → L2 expects different proof format
│  │    └─ Solution: Update L2 to accept L1 proof format
│  └─ YES → Proceed to next check
├─ Is it a state format issue?
│  ├─ Check: What does L1 output? What does L2 expect?
│  ├─ Different → L2 can't parse L1 output
│  │    └─ Solution: Build adapter or fix interface
│  └─ Same → Proceed to next check
├─ Is it a lattice invariant issue?
│  ├─ Check: Does L2 validate lattice structure?
│  ├─ NO → L2 should validate
│  │    └─ Solution: Add lattice validation to L2
│  └─ YES → Does L1 output violate lattice?
│     ├─ YES → L1 is generating invalid lattice points
│     │    └─ Solution: Fix L1 projection algorithm
│     └─ NO → Proceed to next check
└─ Resolution: Implement the fix identified above
```

---

### Problem: "Hardcoded Path Error in Different Directory"

```
Works in ~/project/cqe/:
  python runtime.py ✓

Fails in /tmp/:
  cd /tmp; python ~/project/cqe/runtime.py ✗
  
├─ Check what the error is
│  ├─ "FileNotFoundError: data/file.json" → relative path issue
│  ├─ "ModuleNotFoundError: layer1" → sys.path issue
│  └─ Other → different problem
│
├─ If relative path issue:
│  ├─ Find where relative path is used
│  │  └─ grep -n "data/" runtime.py
│  ├─ Replace with: Path(__file__).parent / "data"
│  └─ Re-test from /tmp
│
├─ If sys.path issue:
│  ├─ Check: Does __init__.py exist in layer directories?
│  ├─ NO → Add empty __init__.py to each layer
│  ├─ YES → Check PYTHONPATH
│  │  └─ Add: sys.path.insert(0, Path(__file__).parent)
│  └─ Re-test
│
└─ Resolution: Once it works from any directory, commit
```

---

### Problem: "Circular Import Error"

```
ImportError: circular import detected
├─ Which modules are importing each other?
│  ├─ Read the full error trace
│  ├─ Identify chain: module_a → module_b → module_c → module_a
│  └─ Circular dependency confirmed
│
├─ Is it a layer boundary violation?
│  ├─ Example: layer3 imports layer4, layer4 imports layer3
│  ├─ YES → This violates Rule 1
│  │    └─ Solution: Layer 4 should be called by Layer 5, not by Layer 3
│  └─ NO → Proceed to next check
│
├─ Is it a same-layer dependency?
│  ├─ Example: module_a (in layer2) imports module_b (in layer2), which imports module_a
│  ├─ YES → Refactor to remove cycle
│  │    ├─ Option A: Move shared code to separate module
│  │    ├─ Option B: Use lazy import (import inside function)
│  │    └─ Option C: Restructure modules
│  └─ NO → Proceed to next check
│
└─ Resolution: Refactor to eliminate cycle identified above
```

---

### Problem: "Test Has No Proof"

```
Test runs and returns result, but no proof in output
├─ Is the function supposed to return proof?
│  ├─ Check documentation: "Returns (result, proof)" vs "Returns result"
│  ├─ Supposed to have proof but doesn't
│  │    └─ Solution: Wrap operation with proof generation
│  └─ Not supposed to have proof (utility function)
│     └─ No fix needed; expected behavior
│
├─ If supposed to have proof:
│  ├─ How is proof generated?
│  │  └─ Check: generate_proof() function exists?
│  ├─ NO → Add proof generation
│  │    └─ result, proof = Layer1.create_atom(value)
│  ├─ YES → Is it being called?
│  │  ├─ Add: proof = generate_proof(result)
│  │  ├─ Add: return result, proof
│  │  └─ Re-run test
│  └─ If still no proof → might be nested function not generating proof
│     └─ Check caller function also wraps with proof
│
└─ Resolution: Add proof generation, re-run, verify
```

---

### Problem: "Operations Score the Same"

```
Implementation A scores: 0.84
Implementation B scores: 0.84
├─ Apply tiebreaker 1: Most recent modification
│  ├─ Check: git log --oneline file_a.py file_b.py
│  ├─ A is newer → Choose A
│  └─ B is newer → Choose B
│
├─ If same date, apply tiebreaker 2: Code complexity
│  ├─ A has cyclomatic complexity 8 → simpler
│  ├─ B has cyclomatic complexity 12 → more complex
│  └─ Choose A (lower complexity)
│
├─ If still tied, apply tiebreaker 3: Location
│  ├─ A is in layer2_geometric/ (canonical location)
│  ├─ B is in legacy/ (archive location)
│  └─ Choose A (in canonical location)
│
└─ If still tied → Both are genuinely equivalent
   └─ Choose either; document why
```

---

## PART 3: QUICK REFERENCE FOR RUNNING CHECKS

```bash
# Before each phase
bash enforce_rules.sh

# After integrating each layer
python -m pytest layer${N}/tests/ -v --tb=short

# Check proof chain
python test_proof_chain.py

# Run comprehensive harness
python comprehensive_test_harness.py

# Check all constraints
check_layer_imports && \
check_proof_completeness && \
check_hardcoded_paths && \
check_layer_modification && \
echo "✓ All constraints satisfied"
```

---

## PART 4: WHEN TO ESCALATE

Escalate as blocker if:

1. **MVP won't run** (even after trying troubleshooting)
2. **File registry is corrupted** (invalid JSON, missing files)
3. **All implementations of critical concept fail tests** (no valid candidate)
4. **Circular dependency can't be broken** (architectural issue)
5. **Tests pass but system obviously doesn't work** (test suite is broken)
6. **Can't find proof of concept existing** (might be missing from archive)
7. **Deployment infrastructure is incompatible** (Docker/K8s/cloud all fail)

**How to escalate**:
```
1. Document what you tried
2. Include error logs
3. Note which phase you're in
4. Ask specific question
5. Wait for clarification before proceeding
```

---

## PART 5: EXPECTED MESSAGES THROUGHOUT PROCESS

### Phase 1-2 (Inventory & Layer 1)

```
✓ File registry loaded (4231 files)
✓ MVP identified: runtime.py
✓ MVP runs: python runtime.py --info → [output]
✓ Categorized 847 files (265 code, 412 tests, 89 docs, 271 other)
✓ Layer 1: 23 implementations found
✓ Layer 1: Selected cqe_atom.py (score 0.91)
✓ Layer 1 tests pass: 23/23
```

### Phase 3-4 (Layers 2-3)

```
✓ Layer 2 E8: Selected e8_ops.py (score 0.88)
✓ Layer 2 E8 tests pass: 18/18
✓ Layer 2 Leech: Selected leech_lattice.py (score 0.85)
✓ Layer 2 Leech tests pass: 15/15
⚠ Layer 2 integration test: 2 failures (investigating)
✓ Layer 2 failures resolved
✓ All Layer 2 tests pass: 63/63
✓ Layer 3 Conservation: Selected conservation.py (score 0.90)
✓ Layer 3 tests pass: 28/28
```

### Phase 5-7 (Layers 4-5 & Deployment)

```
✓ Layer 4 tests pass: 31/31
✓ Layer 5 GNLC tests pass: 25/25
✓ Proof chain verified: 100/100 proofs valid
✓ Docker build succeeds
✓ Docker container runs: identical output to local
✓ Kubernetes deployment: pods healthy
✓ Cloud deployment: endpoints responding
✓ ALL TESTS PASS: 412/412 ✓
✓ SYSTEM READY FOR PRODUCTION
```

---

End of glossary and troubleshooting guide.
