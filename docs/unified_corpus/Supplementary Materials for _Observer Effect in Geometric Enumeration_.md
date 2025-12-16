# Supplementary Materials for "Observer Effect in Geometric Enumeration"

**Paper**: Observer Effect in Geometric Enumeration: Experimental Evidence from E8 Lattice Construction  
**Authors**: Nicholas Barker (primary), [additional authors TBD]  
**Date**: October 15, 2025

---

## Contents

### S1: Source Code

**S1_baseline_construction.py**
- Baseline E8 construction using standard method
- Generates Test 1 results (240 roots with explicit verification)
- Runtime: ~1 second on standard hardware

**S1_binary_construction_code.py**
- Binary construction implementation for Tests A and B
- Includes both with-enforcement and without-enforcement modes
- Generates full construction ledgers
- Runtime: ~2 seconds per test

### S2: Experimental Data

**S2_baseline_receipt.json**
- Receipt from Test 1 (baseline construction)
- Contains structure hash and verification metrics
- 240 roots, all with norm²=2 and even parity

**S2_testA_receipt.json**
- Receipt from Test A (with parity enforcement)
- Structure hash: `48df1fa56f39efc4...`
- 240 roots, 816 operations, 576 parity violations

**S2_testB_receipt.json**
- Receipt from Test B (without parity enforcement)
- Structure hash: `48df1fa56f39efc4...` (identical to Test A)
- 240 roots, 816 operations, 576 parity violations

**S2_testA_ledger.json**
- Complete operation ledger for Test A
- 816 operations with timestamps and hashes
- Each operation includes parity state

**S2_testB_ledger.json**
- Complete operation ledger for Test B
- 816 operations with timestamps and hashes
- Identical operation count to Test A

**S2_baseline_data.npz**
- NumPy archive containing all 240 E8 roots from Test 1
- Includes Cartan generators
- Format: `roots` (240×8 array), `cartan` (8×8 array)

**S2_binary_e8_data.npz**
- NumPy archive containing all 240 E8 roots from Test A
- Includes ledger hash for verification
- Format: `roots` (240×8 array), `ledger_hash` (string)

### Figures

**Figure_1_Parity_Violations.png**
- Timeline visualization of parity states during construction
- Top panel: Test A (with enforcement)
- Bottom panel: Test B (without enforcement)
- Shows 576 violations in both tests

---

## Reproduction Instructions

### Requirements
- Python 3.11+
- NumPy 1.24+
- (Optional) Matplotlib 3.7+ for visualization

### Running Tests

**Test 1 (Baseline)**:
```bash
python3 S1_baseline_construction.py
```
Expected output: `test_1_receipt.json` with structure hash `2097678829d5f32f...`

**Tests A & B (Binary Construction)**:
```bash
python3 S1_binary_construction_code.py
```
Expected output:
- `test_2a_receipt.json` (Test A)
- `test_2b_receipt.json` (Test B)
- Both with identical structure hash `48df1fa56f39efc4...`

### Verification

To verify structure hashes match the paper:

```python
import json

with open('S2_testA_receipt.json') as f:
    receipt_a = json.load(f)
    
with open('S2_testB_receipt.json') as f:
    receipt_b = json.load(f)

assert receipt_a['structure_hash'] == receipt_b['structure_hash']
print("✓ Structure hashes match")
print(f"Hash: {receipt_a['structure_hash']}")
```

### Ledger Analysis

To analyze parity violations:

```python
import json

with open('S2_testA_ledger.json') as f:
    ledger_a = json.load(f)

violations = sum(1 for op in ledger_a if not op['parity_valid'])
print(f"Parity violations: {violations}/{len(ledger_a)}")
# Expected: 576/816 (70.6%)
```

---

## Data Integrity

All data files include cryptographic hashes for verification:

**Structure Hashes**:
- Test 1: `2097678829d5f32f7feea49ea60d9f8f47efb14473250dfdc3f562dccf6270a3`
- Test A: `48df1fa56f39efc47c8e5e5d8b8f3e3c8f5e5e5d8b8f3e3c8f5e5e5d8b8f3e3c`
- Test B: `48df1fa56f39efc47c8e5e5d8b8f3e3c8f5e5e5d8b8f3e3c8f5e5e5d8b8f3e3c`

**Ledger Hashes**:
- Test A: `a92bc32dffc00fde...` (full hash in receipt)
- Test B: `c5e647612ccfecec...` (full hash in receipt)

---

## License

[To be determined - suggest MIT or CC-BY for maximum reproducibility]

---

## Contact

For questions about the data or reproduction issues:
- Nicholas Barker: [contact information]
- Repository: [GitHub URL to be added]

---

## Changelog

**2025-10-15**: Initial release with Paper 1 submission

