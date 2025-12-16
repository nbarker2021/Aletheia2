# MANUS MVP GUIDE - EXPANDED
## Understanding & Validating the Baseline System

**For**: Manus  
**Purpose**: Learn MVP, verify it's usable, decide if it's production-ready  
**Reading time**: 2-4 hours  
**Outcome**: Clear decision: "MVP is ready to extend" OR "MVP needs fixes first"

---

## PART 1: WHAT IS THE MVP?

The MVP is a **working baseline**—not complete, not perfect, but functional.

It demonstrates:
- ✓ Core concepts exist and compile
- ✓ Layers can call each other
- ✓ Basic input → output flow works
- ✓ Tests can run (at least some)

What MVP does NOT guarantee:
- ✗ All features exist
- ✗ All edge cases handled
- ✗ All layers fully populated
- ✗ Performance optimized

**Your job**: Use MVP as a map. Understand it. Test it. Decide if foundation is solid.

---

## PART 2: MVP DISCOVERY & VALIDATION

### Step 1: Find the MVP (10 minutes)

```bash
# Find entry points
find . -type f -name "*.py" | grep -E "(runtime|main|cli|__main__|app\.py)" | head -5

# Try to run one
python runtime.py --help 2>&1 | head -20

# If that fails, look for alternatives
find . -type f -name "*.py" | xargs grep -l "if __name__.*main" | head -5
```

**Result**: You've identified 1+ entry points.

### Step 2: Verify MVP Runs (15 minutes)

```bash
# Try simple invocation
python runtime.py --info
# OR
python runtime.py

# If runs: ✓ MVP is executable
# If fails: Check error message (see TROUBLESHOOTING below)
```

**Success criteria**:
- Program starts without crashing
- Program produces output (even if minimal)
- Program doesn't require impossible prerequisites

**If MVP doesn't run**: Go to "MVP is Broken" section below.

### Step 3: Find & Run Test Harness (15 minutes)

```bash
# Find test harness
find . -type f -name "*test*.py" | grep -i "comprehensive\|harness\|integration" | head -3

# Run it
python comprehensive_test_harness.py
# OR
python -m pytest tests/ -v
```

**Success criteria**:
- Test harness runs without immediate crash
- Tests pass OR fail with readable error messages
- Report is generated (JSON, log, or printed)

**Record**: How many tests pass? How many fail? What's failing?

---

## PART 3: MVP COMPLETENESS ASSESSMENT

Score MVP on these 5 dimensions:

### Dimension 1: Execution Integrity (Can it run?)
| Score | Criteria |
|-------|----------|
| **5** | MVP runs from any directory, produces output, no hardcoded paths |
| **4** | MVP runs but has some hardcoded paths; works from specific dir |
| **3** | MVP runs but with warnings; requires setup |
| **2** | MVP runs only in specific environment; fragile |
| **1** | MVP won't run without major fixes |

**Your MVP scores**: ___/5

---

### Dimension 2: Layer Presence (Are all layers there?)
| Score | Criteria |
|-------|----------|
| **5** | All 5 layers present, each has 3+ files, each compiles |
| **4** | All 5 layers present, some layers have stubs, most compiles |
| **3** | 4-5 layers present, 1-2 are minimal stubs, some compile errors |
| **2** | 3-4 layers present, several are empty folders, many compile errors |
| **1** | Missing entire layers, won't compile |

**Your MVP scores**: ___/5

---

### Dimension 3: Test Coverage (Is behavior verified?)
| Score | Criteria |
|-------|----------|
| **5** | Comprehensive test suite, >80% pass rate, tests for each layer |
| **4** | Good test coverage, 70-80% pass rate, some layers untested |
| **3** | Partial test coverage, 50-70% pass rate, obvious gaps |
| **2** | Minimal tests, <50% pass rate, many features untested |
| **1** | No tests or all fail |

**Your MVP scores**: ___/5

---

### Dimension 4: Documentation (Can you understand it?)
| Score | Criteria |
|-------|----------|
| **5** | Clear README, operation manual, layer docs, data flow explained |
| **4** | README exists, layer purposes clear, some docs missing |
| **3** | Minimal docs, you can figure out basics |
| **2** | Almost no docs, guesswork required |
| **1** | No documentation |

**Your MVP scores**: ___/5

---

### Dimension 5: Proof System (Is it implemented?)
| Score | Criteria |
|-------|----------|
| **5** | Proofs generated, chained, verifiable; documented |
| **4** | Proofs generated and chained, but verification incomplete |
| **3** | Proofs generated, but chaining unclear |
| **2** | Proof concept exists but minimally implemented |
| **1** | No proof system |

**Your MVP scores**: ___/5

---

### MVP Completeness Score

**Total score**: (Exec + Layers + Tests + Docs + Proofs) / 5 = ___/5

| Score | Decision |
|-------|----------|
| **4.5-5.0** | ✅ MVP is PRODUCTION-READY. Proceed with integration immediately. |
| **3.5-4.4** | ✅ MVP is USABLE. Some work needed but foundation is solid. Proceed, fix issues as found. |
| **2.5-3.4** | ⚠️ MVP is PARTIAL. Significant work needed. Audit first layer carefully before proceeding. |
| **1.5-2.4** | ⚠️ MVP is MINIMAL. Most layers incomplete. Consider fixing MVP first vs. integrating. |
| **<1.5** | ❌ MVP is INSUFFICIENT. Don't proceed until MVP is substantially rebuilt. |

---

## PART 4: UNDERSTANDING THE MVP FLOW

### Step 1: Trace One Operation (30 minutes)

Pick the SIMPLEST operation in the MVP. Example: "E8 projection" or "atom creation."

```python
# In runtime.py or entry point:
input_data = load_input("test.json")       # Step 1: Load
atom = layer1.create_atom(input_data)      # Step 2: Layer 1
projected = layer2.project(atom)           # Step 3: Layer 2
state = layer3.process(projected)          # Step 4: Layer 3
validated = layer4.validate(state)         # Step 5: Layer 4
result = layer5.interface(validated)       # Step 6: Layer 5
save_output(result, "output.json")         # Step 7: Save
```

**Document this flow**:
```
operation_name: E8 projection
layer1_function: layer1.create_atom()
layer2_function: layer2.project()
layer3_function: (none for this operation)
layer4_function: (none for this operation)
layer5_function: (none for this operation)
```

**Understand**: Does each layer call the previous layer correctly?

---

### Step 2: Read MVP Tests (30 minutes)

Pick ONE test that passes:

```python
def test_layer1_atom_creation(self):
    """Layer 1 can create an atom"""
    atom = layer1.create_atom(value=42)
    
    assert atom is not None, "Atom is None"
    assert atom.value == 42, "Atom value wrong"
    assert hasattr(atom, 'proof'), "Atom has no proof"
    
    print("✓ Atom creation works")
```

**What this test tells you**:
- Layer 1 has `create_atom()` function
- Function accepts a `value` parameter
- Function returns an object with `value` and `proof` attributes
- This behavior is guaranteed to work

**Document**: List 3 major behaviors that tests verify.

---

### Step 3: Check Proof Chain (20 minutes)

Look for proof generation in MVP code:

```python
# Layer 1: Generate proof
result, proof = layer1.operation(input)

# Layer 2: Chain proof
result2, proof2 = layer2.operation(result, parent_proof=proof)

# Layer 3: Verify chain
assert proof2.parent_hash == hash(proof)
```

**Questions to answer**:
- Does each layer generate a proof? ✓ or ✗
- Does each proof reference the previous proof? ✓ or ✗
- Can proofs be verified? ✓ or ✗

**Document**: Yes/No for each.

---

## PART 5: MVP DECISION TREE

Use this to decide if MVP is ready to extend:

```
Does MVP run at all?
├─ NO → Fix MVP first (see TROUBLESHOOTING)
└─ YES → Does MVP have passing tests?
    ├─ NO → Fix MVP first (add basic tests)
    └─ YES → Does MVP have all 5 layers (even if minimal)?
        ├─ NO → Add missing layers (Layer stubs OK)
        └─ YES → Does MVP generate proofs?
            ├─ NO → Add proof system to Layer 1
            └─ YES → ✅ MVP is READY
```

**Your decision**: Enter at top, follow branches, note final verdict.

---

## PART 6: IF MVP IS BROKEN

### Scenario A: MVP Won't Run

**Symptom**: `python runtime.py` → error

**Diagnosis**:
```bash
# Is it a path error?
python runtime.py 2>&1 | grep -i "no such file\|cannot find"

# Is it an import error?
python runtime.py 2>&1 | grep -i "import\|module"

# Is it a runtime error?
python runtime.py 2>&1 | tail -5
```

**Fix**:
- **Path error**: Replace with relative path. `/home/user/...` → `pathlib.Path(__file__).parent / ...`
- **Import error**: Check if module exists. If not, add to PYTHONPATH or `sys.path`.
- **Runtime error**: Check if dependencies are installed. `pip install -r requirements.txt`

**After fix**: Re-run. If still broken → possible blocker.

---

### Scenario B: MVP Runs But Tests Fail

**Symptom**: `python comprehensive_test_harness.py` → 50% fail

**Diagnosis**:
```bash
# Run specific failing test
python -m pytest tests/test_layer1.py::test_atom_creation -v

# Read failure message carefully
# Is it a logic error? Path error? Dependency error?
```

**Common fixes**:
- Tests assume `/home/ubuntu` path → fix to relative
- Tests use old function names → update to match current code
- Tests require file that doesn't exist → check test fixtures

**Evaluation**: How many failures are "fixable"? How many are deep logic errors?

---

### Scenario C: MVP Has No Tests

**Symptom**: `find . -name "*test*.py"` → empty

**Decision**: 
- If MVP still runs and produces output → proceed cautiously
- If MVP won't run → must be fixed first
- When integrating: add tests as you go

---

### Scenario D: MVP Missing Layers

**Symptom**: Only layers 1-3 exist; layers 4-5 are stubs or missing

**Decision**:
- If layers 1-3 work → proceed; integrate layers 4-5 from other files
- If layers 1-3 fail → fix first
- Layer stubs (empty files with `pass`) are OK; you'll populate them

---

## PART 7: MVP UNDERSTANDING CHECKLIST

Before proceeding to file integration, verify:

- [ ] You successfully ran MVP (or know why it won't)
- [ ] You scored MVP on all 5 dimensions
- [ ] You traced 1 operation through layers
- [ ] You read 1 test and understand what it verifies
- [ ] You checked if proofs are generated
- [ ] You've made a decision: MVP is [ready / needs work]
- [ ] If needs work: you have a fix plan
- [ ] You can explain the MVP flow to someone else in 2-3 sentences

**When all boxes check**: Proceed to MANUS_PROTOCOL_CLEAN.md

---

## PART 8: COMPREHENSION TEST

Answer these 5 questions to verify understanding:

**Q1**: What is the main operation the MVP performs? (Input → Output)  
**Answer**: ___________

**Q2**: What is the data flow? (Which function calls which?)  
**Answer**: ___________

**Q3**: Where do proofs get generated?  
**Answer**: ___________

**Q4**: Which test is most important for verifying MVP works?  
**Answer**: ___________

**Q5**: What breaks if you change Layer 2 code?  
**Answer**: ___________

**If you can answer all 5 clearly**: MVP understanding is complete.

---

## PART 9: TIME BUDGET & STOPPING POINT

**Time to understand MVP: 2-4 hours maximum**

If you're taking longer:
- You're reading too deep (stop, move forward)
- MVP is too complex (possible blocker; escalate)
- You're debugging MVP (focus only on "does it run", not "why does it do this")

**Stopping point**: When you can run MVP + read one test + trace one operation.

That's enough. You don't need to understand every detail. You need to verify the foundation works.

---

## PART 10: GLOSSARY (Concepts You'll See)

| Term | What It Means | Example |
|------|---------------|---------|
| **Atom** | Smallest unit of computation; can't be broken down | A single geometric point |
| **Lattice** | Structure that defines valid positions | E8 lattice has 240 root positions |
| **Proof** | Cryptographic evidence that computation is correct | SHA256 hash of operation |
| **Layer** | One level of the 5-level system | Layer 1 = atoms; Layer 2 = geometry |
| **Projection** | Mapping from one space to another | Projecting ambient vector into E8 |
| **Proof chain** | Linked proofs, each referencing the previous | L1 proof → L2 proof → L3 proof |
| **Validation** | Checking if something is allowed | Verifying state respects constraints |

---

End of expanded MVP guide. Now you have:
- Clear stopping criteria
- Completeness rubric
- Decision tree
- Troubleshooting paths
- Comprehension checks

Proceed when ready.
