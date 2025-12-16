# MANUS QUICK REFERENCE - EXPANDED
## Complete Working Guide with Enforcement & Tools

---

## READING ORDER (START HERE)

1. **MANUS_MVP_GUIDE_EXPANDED.md** ← Validate baseline (2-4 hours)
2. **MANUS_PROTOCOL_EXPANDED.md** ← Integration workflow (ongoing)
3. **This document** ← Quick lookup while working

---

## THE 5 UNBREAKABLE RULES + ENFORCEMENT

### Rule 1: Geometric Layering

**What**: Layer N imports ONLY from Layers 1…N-1. NO exceptions.

**Enforcement - Automated Check**:
```bash
# Check for violations
check_layer_imports() {
  for layer in layer3_operational layer4_governance layer5_interface; do
    layer_num=$(echo $layer | grep -o "[0-9]")
    next_layer=$((layer_num + 1))
    
    violations=$(grep -r "from layer[${next_layer}-9]" $layer --include="*.py" 2>/dev/null)
    if [ ! -z "$violations" ]; then
      echo "❌ VIOLATION: $layer imports from later layers"
      echo "$violations"
      return 1
    fi
  done
  echo "✓ All layers properly isolated"
  return 0
}

check_layer_imports
```

**When to Run**: Before integrating each layer, after completing it.

**What Violates It**:
```python
# Layer 3 code:
from layer5_interface import SDK  # ❌ VIOLATION

# Layer 2 code:
from layer4_governance import Validator  # ❌ VIOLATION
```

**What's OK**:
```python
# Layer 3 code:
from layer1_morphonic import Atom  # ✓ OK
from layer2_geometric import E8Lattice  # ✓ OK
```

---

### Rule 2: Proofs Are First-Class

**What**: Every operation returns `(result, proof)`. Never bare result.

**Enforcement - Automated Check**:
```bash
# Find operations that don't return proofs
check_proof_completeness() {
  for layer in layer1 layer2 layer3 layer4 layer5; do
    echo "Checking $layer for proof-less functions..."
    
    # Find functions that return but aren't proof-related
    violations=$(grep -n "^\s*return [^(]" "$layer"*.py 2>/dev/null | \
                 grep -v "return None" | \
                 grep -v "return True" | \
                 grep -v "return False" | \
                 grep -v "return \[" | \
                 grep -v "# proof" | \
                 head -10)
    
    if [ ! -z "$violations" ]; then
      echo "⚠️ WARNING: Possible proof-less returns:"
      echo "$violations"
    fi
  done
}

check_proof_completeness
```

**When to Run**: During code review of each layer.

**What Violates It**:
```python
def project_to_e8(vector):
    result = compute_projection(vector)
    return result  # ❌ VIOLATION: no proof

def atom_creation(value):
    atom = Atom(value)
    return atom  # ❌ VIOLATION: no proof
```

**What's OK**:
```python
def project_to_e8(vector):
    result = compute_projection(vector)
    proof = generate_proof(result)
    return result, proof  # ✓ OK

def atom_creation(value):
    atom = Atom(value)
    proof = Proof(operation="atom_creation", output=atom)
    return atom, proof  # ✓ OK
```

---

### Rule 3: Relative Paths Only

**What**: No `/home/`, no `C:\`, no absolute paths.

**Enforcement - Automated Check**:
```bash
# Find hardcoded paths
check_hardcoded_paths() {
  echo "Checking for hardcoded paths..."
  
  violations=$(grep -r "/home\|/Users\|/root\|C:\\\\\\\\Users\|D:\\\\\\\\Users" \
               layer*/*.py tests/*.py 2>/dev/null)
  
  if [ ! -z "$violations" ]; then
    echo "❌ HARDCODED PATHS FOUND:"
    echo "$violations"
    return 1
  fi
  
  echo "✓ No hardcoded absolute paths"
  return 0
}

check_hardcoded_paths
```

**When to Run**: Before integration of each file.

**Fix Template**:
```python
# ❌ WRONG
DATA_DIR = "/home/ubuntu/project/data"

# ✅ RIGHT
import pathlib
DATA_DIR = pathlib.Path(__file__).parent / "data"

# ✅ ALSO RIGHT
from pathlib import Path
CONFIG_FILE = Path(__file__).parent.parent / "config" / "settings.json"
```

---

### Rule 4: All Behaviors Are Testable

**What**: If docs say it happens, a test verifies it.

**Enforcement - Mapping Check**:
```python
# Build map of documented behaviors vs tests
documented_behaviors = parse_documentation("docs/")
existing_tests = parse_test_files("tests/")

missing_tests = documented_behaviors - existing_tests

if missing_tests:
    print(f"⚠️ MISSING TESTS for {len(missing_tests)} behaviors:")
    for behavior in missing_tests:
        print(f"  - {behavior}")
else:
    print("✓ All documented behaviors have tests")
```

**When to Run**: During integration of documentation.

**What Violates It**:
```
Documented: "E8 projection preserves lattice structure"
Tests: None that verify this
Status: ❌ VIOLATION
```

**What's OK**:
```
Documented: "E8 projection preserves lattice structure"
Tests: test_e8_projection_preserves_structure() exists and passes
Status: ✓ OK
```

---

### Rule 5: Layer Boundaries Are Strict

**What**: Layer N cannot modify Layer N+1 code. Ever.

**Enforcement - Scope Check**:
```bash
# Check if layer code tries to modify parent layer
check_layer_modification() {
  for layer in layer1 layer2 layer3 layer4; do
    layer_num=$(echo $layer | grep -o "[0-9]")
    next_layer="layer$((layer_num + 1))"
    
    modifications=$(grep -r "open.*$next_layer.*w\|write.*$next_layer\|patch.*$next_layer" \
                   $layer --include="*.py" 2>/dev/null)
    
    if [ ! -z "$modifications" ]; then
      echo "❌ VIOLATION: $layer tries to modify $next_layer"
      return 1
    fi
  done
  echo "✓ No cross-layer modifications"
  return 0
}

check_layer_modification
```

---

## SCORING RUBRIC (Detailed)

When choosing between implementations:

```
Total Score = (0.40 × Completeness) + (0.40 × Testing) + (0.20 × Simplicity)
```

### Completeness (0-1 scale)

| Score | Criteria |
|-------|----------|
| 1.0 | All required functions, all edge cases, full documentation |
| 0.9 | All functions, most edge cases, good documentation |
| 0.8 | All functions, some edge cases, partial documentation |
| 0.7 | Core functions, few edge cases, minimal documentation |
| 0.6 | Core functions only, no edge cases, no documentation |
| 0.5 | Partial functions, incomplete |
| <0.5 | Stub implementation |

**How to measure**:
```python
def assess_completeness(file_path):
    required_functions = ["create", "validate", "execute", "verify"]
    actual_functions = extract_functions(file_path)
    
    func_coverage = len([f for f in required_functions if f in actual_functions]) / len(required_functions)
    edge_case_count = count_edge_cases(file_path)
    doc_coverage = count_documented_functions(file_path) / len(actual_functions)
    
    score = (0.5 * func_coverage + 
             0.3 * min(edge_case_count / 5, 1.0) + 
             0.2 * doc_coverage)
    return score
```

### Testing (0-1 scale)

| Score | Criteria |
|-------|----------|
| 1.0 | >20 tests, 100% pass rate, >90% code coverage |
| 0.9 | 15-20 tests, 95%+ pass rate, 80-90% coverage |
| 0.8 | 10-14 tests, 90%+ pass rate, 70-80% coverage |
| 0.7 | 5-9 tests, 85%+ pass rate, 50-70% coverage |
| 0.6 | 3-4 tests, 80%+ pass rate, 30-50% coverage |
| 0.5 | 1-2 tests, <80% pass rate, <30% coverage |
| <0.5 | 0 tests or <50% pass rate |

**How to measure**:
```bash
# Run tests and collect metrics
pytest $file_path --cov --cov-report=json
python -m pytest $file_path -v | grep -c "PASSED"
```

### Simplicity (0-1 scale)

| Score | Criteria |
|-------|----------|
| 1.0 | <200 lines, complexity ≤5, ≤3 external imports |
| 0.9 | 200-300 lines, complexity 5-7, 3-4 imports |
| 0.8 | 300-400 lines, complexity 7-10, 4-5 imports |
| 0.7 | 400-500 lines, complexity 10-15, 5-7 imports |
| 0.6 | 500-700 lines, complexity 15-20, 7-10 imports |
| <0.6 | >700 lines or complexity >20 |

**How to measure**:
```python
def assess_simplicity(file_path):
    lines = count_lines(file_path)
    complexity = calculate_cyclomatic_complexity(file_path)
    imports = count_non_stdlib_imports(file_path)
    
    line_score = max(0, 1.0 - (lines - 200) / 500)
    complex_score = max(0, 1.0 - (complexity - 5) / 15)
    import_score = max(0, 1.0 - (imports - 3) / 7)
    
    return (0.5 * line_score + 0.3 * complex_score + 0.2 * import_score)
```

---

## PROOF VERIFICATION PROCEDURE

### Step 1: Check Proof Format

```python
def verify_proof_format(proof):
    required_fields = [
        "operation",
        "input_hash",
        "output_hash",
        "layer",
        "timestamp",
        "parent_proof_hash",
        "verification_algorithm",
        "signature"
    ]
    
    for field in required_fields:
        if field not in proof:
            raise ValueError(f"Missing field: {field}")
    
    return True
```

### Step 2: Verify Hash Chain

```python
def verify_proof_chain(current_proof, previous_proof):
    # Current proof must reference previous proof's hash
    current_parent_hash = current_proof["parent_proof_hash"]
    previous_hash = hash_proof(previous_proof)
    
    if current_parent_hash != previous_hash:
        raise ValueError("Proof chain broken: parent hash doesn't match")
    
    # Verify current proof's hash is correct
    computed_hash = sha256(proof_to_canonical_json(current_proof)).hexdigest()
    if computed_hash != hash_of(current_proof):
        raise ValueError("Proof hash mismatch")
    
    return True
```

### Step 3: Verify Signature

```python
def verify_proof_signature(proof, public_key):
    signature = bytes.fromhex(proof["signature"])
    message = proof_to_canonical_json(proof)
    
    try:
        ed25519_verify(public_key, signature, message.encode())
        return True
    except Exception as e:
        raise ValueError(f"Signature verification failed: {e}")
```

### Full Verification Procedure

```bash
verify_proof_system() {
  echo "Verifying proof system..."
  
  # 1. Check format
  python -c "from proof_validator import verify_proof_format; verify_proof_format(load_proof())"
  
  # 2. Check chain
  python -c "from proof_validator import verify_proof_chain; verify_proof_chain(...)"
  
  # 3. Check signature
  python -c "from proof_validator import verify_proof_signature; verify_proof_signature(...)"
  
  echo "✓ All proofs valid"
}
```

---

## DEPLOYMENT VERIFICATION CHECKLIST

### Docker

```bash
# 1. Build
docker build -t cqe:test .
# Expected: Build succeeds, no layer errors

# 2. Run locally
docker run cqe:test python runtime.py --info
# Expected: Same output as local run

# 3. Run test harness inside container
docker run cqe:test python comprehensive_test_harness.py
# Expected: All tests pass (or same failures as local)

# 4. Check paths
docker run cqe:test find . -name "*.py" -exec grep -l "/home" {} \;
# Expected: Empty (no hardcoded paths)
```

### Kubernetes

```bash
# 1. Apply manifests
kubectl apply -f deployment/kubernetes/

# 2. Wait for pod ready
kubectl wait --for=condition=ready pod -l app=cqe --timeout=300s

# 3. Run test
kubectl exec $(kubectl get pod -l app=cqe -o name) -- python runtime.py --test

# 4. Check logs
kubectl logs $(kubectl get pod -l app=cqe -o name) | grep -i error

# 5. Cleanup
kubectl delete -f deployment/kubernetes/
```

### Cloud (AWS Example)

```bash
# 1. Deploy
aws cloudformation create-stack --template-body file://deployment/aws/template.yml

# 2. Wait
aws cloudformation wait stack-create-complete

# 3. Get endpoint
ENDPOINT=$(aws cloudformation describe-stacks --query 'Stacks[0].Outputs[?OutputKey==`Endpoint`].OutputValue' --output text)

# 4. Test
curl $ENDPOINT/api/test

# 5. Compare with local
diff <(python runtime.py --test | jq) <(curl $ENDPOINT/api/test | jq)
```

---

## ERROR QUICK-FIX WITH ROOT CAUSE ANALYSIS

| Error | Likely Cause | Fix | Verify |
|-------|--------------|-----|--------|
| `ImportError: No module layer2` | File not in PYTHONPATH | Add `sys.path.insert(0, ...)` or check `__init__.py` | `python -c "from layer2 import ..."` |
| `TypeError: expecting (result, proof)` | Operation returns bare result | Wrap with `generate_proof()` | Run test; should pass |
| `FileNotFoundError: /home/ubuntu` | Hardcoded path | Replace with `pathlib.Path(__file__).parent / ...` | Run from different directory |
| `AssertionError: proof verification failed` | Proof hash wrong OR signature invalid | Check hash computation; verify key matches | Recompute hash manually |
| `Circular ImportError` | Layer N imports Layer N+1 which imports Layer N | Refactor: move shared code to Layer N | Check imports with `grep` |
| `TestFailed: assertion X == Y` | Logic error or state corruption | Debug with print statements or debugger | Add print; rerun |
| `Timeout: test takes >30s` | Infinite loop OR N² algorithm | Add timeout to test; profile code | Run with `time` command |

---

## SUCCESS METRICS (With Measurement Method)

| Metric | Target | How to Measure |
|--------|--------|---|
| File deduplication | 0 `...(1).py` files | `find . -name '*(1).py'` → should be empty |
| Layer boundaries | 0 cross-layer imports | `check_layer_imports()` → should pass |
| Path hardcoding | 0 absolute paths | `check_hardcoded_paths()` → should pass |
| Proof completeness | 100% return (result, proof) | `check_proof_completeness()` → should pass |
| Test coverage | ≥80% | `pytest --cov . --cov-report=term-missing` |
| Harness pass rate | rc=0 | `python comprehensive_test_harness.py; echo $?` |
| Deployment success | All 3 targets work | `check_docker() && check_k8s() && check_cloud()` |
| Proof verification | 100% proofs verify | `verify_proof_system()` → all pass |
| Governance enforcement | Invalid states rejected | Try to create invalid state; should be caught |
| Performance | <1s per layer 1-2 op | `time python -c "layer1.create_atom()"` |

---

## AUTOMATED ENFORCEMENT SCRIPT

```bash
#!/bin/bash
# Run all checks before proceeding

set -e  # Exit on first error

echo "=== MANUS ENFORCEMENT CHECK ==="

# Rule 1: Layering
echo "Checking Rule 1: Layer isolation..."
check_layer_imports || exit 1

# Rule 2: Proofs
echo "Checking Rule 2: Proof completeness..."
check_proof_completeness || exit 1

# Rule 3: Paths
echo "Checking Rule 3: No hardcoded paths..."
check_hardcoded_paths || exit 1

# Rule 4: Testing
echo "Checking Rule 4: Test coverage..."
pytest --cov=. --cov-report=json
COVERAGE=$(python -c "import json; print(json.load(open('.coverage.json'))['totals']['percent_covered'])")
if (( $(echo "$COVERAGE < 80" | bc -l) )); then
  echo "❌ Coverage too low: $COVERAGE%"
  exit 1
fi

# Rule 5: Boundaries
echo "Checking Rule 5: Layer modification..."
check_layer_modification || exit 1

echo ""
echo "✓ ALL ENFORCEMENT CHECKS PASSED"
```

Save as: `enforce_rules.sh`  
Run: `bash enforce_rules.sh`

---

## WHEN YOU GET STUCK: DEBUGGING TREE

```
Symptom: Tests fail
├─ Is it a syntax error?
│  └─ python -m py_compile file.py
├─ Is it an import error?
│  └─ python -c "from module import *"
├─ Is it a logic error?
│  ├─ Add print statements
│  ├─ Run with debugger: python -m pdb
│  └─ Compare with reference implementation
├─ Is it a proof error?
│  ├─ Check hash computation
│  ├─ Verify signature
│  └─ Trace proof chain back to L1
└─ Is it a path error?
   └─ echo $PWD ; python -c "print(os.getcwd())"
```

---

## CONCEPT GLOSSARY

| Term | Meaning | Example |
|------|---------|---------|
| **Atom** | Primitive unit, indivisible | Single geometric point |
| **Lattice** | Discrete geometric structure | E8 has 240 nodes |
| **Embedding** | Map from one space to another | E8 → Leech projection |
| **Proof** | Cryptographic evidence of correctness | SHA256 hash of operation result |
| **Proof chain** | Sequential proofs linked by hashes | L1 proof → L2 proof → L3 proof |
| **Governance** | Enforcement of constraints and rules | Layer 4: validates states |
| **Witness** | Independent validator | 7 witnesses must agree |
| **Stratification** | Hierarchical organization | DigitalRoot levels DR0-DR9 |
| **Invariant** | Property that must always hold | "Lattice structure preserved" |
| **Canonical** | The official/authoritative version | Best-scoring implementation chosen |

---

## FINAL GOAL

You're done when:

**You can load the runtime → pass it data → get result + proof → verify proof → all without errors.**

That's production-ready.

---

Keep this card visible while integrating.
