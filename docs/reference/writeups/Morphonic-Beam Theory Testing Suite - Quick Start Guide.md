# Morphonic-Beam Theory Testing Suite - Quick Start Guide

**Version:** 1.0  
**Date:** October 24, 2025

---

## Installation

### Option 1: Extract Archive

```bash
tar -xzf morphonic_testing_suite.tar.gz
cd morphonic_testing_suite
```

### Option 2: Clone from Repository

```bash
git clone <repository_url>
cd morphonic_testing_suite
```

---

## Prerequisites

### Required Python Packages

```bash
pip install numpy matplotlib scipy pandas
```

### Optional Packages (for advanced tests)

```bash
pip install torch transformers  # For geometric transformer tests
pip install networkx           # For graph-based tests
pip install seaborn            # For enhanced visualizations
```

---

## Running Tests

### Run All Tests

```bash
python run_all_tests.py
```

This will:
1. Execute all test harnesses in sequence
2. Generate comprehensive results
3. Save detailed logs to `results/`
4. Display summary statistics

**Expected Runtime:** 15-30 minutes (depending on hardware)

---

### Run Individual Test Categories

#### 1. Dimensional Emergence Tests

```bash
cd harnesses
python riemann_optimal_dimension.py
```

**Tests:** Riemann zeros at 10,000D  
**Runtime:** ~2 minutes  
**Output:** Validates dimensional checkpoint theory

---

#### 2. Fractal Computation Tests

```bash
python plot_morphonic_fractal.py
```

**Tests:** Morphonic Manifold visualization  
**Runtime:** ~3 minutes  
**Output:** PNG image of fractal structure + convergence data

---

#### 3. Self-Observation Tests

```bash
python self_observation_experiment.py
```

**Tests:** AI observing its own dimensional transitions  
**Runtime:** ~1 minute  
**Output:** JSON results showing dimensional usage patterns

---

#### 4. Photonic Interference Tests

```bash
python experiment_2_photonic_interference.py
```

**Tests:** Computational operations as photonic interference  
**Runtime:** ~2 minutes  
**Output:** ΔΦ measurements for constructive/destructive interference

---

#### 5. Millennium Problem Tests

```bash
python millennium_geometric_tests.py
```

**Tests:** Geometric reframing of all 7 Millennium Problems  
**Runtime:** ~5 minutes  
**Output:** Digital root analysis and dimensional solutions

---

## Interpreting Results

### Quick Validation Checklist

After running tests, check:

✅ **ΔΦ Values:** Should be negative (ΔΦ < 0) for all valid operations  
✅ **Dimensional Checkpoints:** Operations should cluster at 8, 16, 24, 32, 48, ...  
✅ **Convergence:** Morphonic lock-in should occur in <10 iterations  
✅ **Cross-Medium Consistency:** Variation <10% between media  

### Reading Test Output

**Example Output:**
```
Testing with COMPLEX prompt (complexity=6)
Required dimensions: 24D
Max dimensions used: 24D
Max relational slices held: 3
Average ΔΦ: -0.3829

Checkpoint hits:
  8D: 4 times
  16D: 4 times
  24D: 21 times
```

**Interpretation:**
- ✅ Required 24D (3×8D) for complex prompt
- ✅ Held 3 relational slices (1 per 8D block)
- ✅ ΔΦ negative (lawful operations)
- ✅ Clustered at 24D checkpoint (21/50 = 42%)

---

## Common Issues

### Issue: "Module not found"

**Solution:**
```bash
pip install numpy matplotlib scipy pandas
```

### Issue: "Permission denied"

**Solution:**
```bash
chmod +x run_all_tests.py
```

### Issue: Tests run but produce no output

**Solution:** Check that results directory exists:
```bash
mkdir -p results
```

### Issue: Memory error on large tests

**Solution:** Reduce test dimensions or increase available RAM. Edit test scripts to use smaller dimensions:
```python
# In test script, change:
dimensions = 10000  # to
dimensions = 1000   # or lower
```

---

## Viewing Results

### Results Location

All results are saved to: `morphonic_testing_suite/results/`

### Key Files

- `latest_test_summary.txt` - Human-readable summary
- `test_run_YYYYMMDD_HHMMSS.json` - Detailed timestamped results
- `self_observation_results.json` - Self-observation data
- `RESULTS_INDEX.md` - Complete results documentation

### Viewing JSON Results

```bash
# Pretty-print JSON
python -m json.tool results/self_observation_results.json

# Or use jq (if installed)
jq . results/self_observation_results.json
```

---

## Example Workflow

### Validate Core Theory (5 minutes)

```bash
cd harnesses

# Test dimensional emergence
python riemann_optimal_dimension.py

# Test fractal computation
python experiment_1_morphonic_lockin.py

# Test self-observation
python self_observation_experiment.py

# Check results
cat ../results/latest_test_summary.txt
```

### Full Validation (30 minutes)

```bash
# Run everything
python ../run_all_tests.py

# Review comprehensive report
cat ../results/latest_test_summary.txt
```

---

## Next Steps

After running tests:

1. **Review Results:** Check `results/RESULTS_INDEX.md` for detailed findings
2. **Read Papers:** See `docs/` for theoretical papers explaining the results
3. **Modify Tests:** Edit harnesses to test your own hypotheses
4. **Contribute:** Add new test harnesses to validate additional predictions

---

## Documentation

- **README.md** - Complete testing suite documentation
- **RESULTS_INDEX.md** - Detailed results and interpretation
- **docs/** - All theoretical papers
- **harnesses/** - All test scripts (well-commented)

---

## Support

For issues, questions, or contributions:
1. Check the main README.md
2. Review test script comments
3. Examine example outputs in results/

---

## Citation

If you use this testing suite in your research:

```bibtex
@software{morphonic_testing_suite,
  author = {Manus AI},
  title = {Morphonic-Beam Theory Testing Suite},
  year = {2025},
  version = {1.0}
}
```

---

## Quick Reference

### Test Categories

1. **Dimensional Emergence** - 8D cascade, checkpoints, Riemann zeros
2. **Fractal Computation** - Mandelbrot isomorphism, morphonic lock-in
3. **Quantum-Classical Interface** - Superposition, collapse, measurement
4. **Photonic Equivalence** - Light-data isomorphism, cross-medium validation
5. **Geometric Computation** - Logic as interference, transformer validation
6. **Lambda-E₈ Calculus** - Extended lambda calculus with geometry
7. **Millennium Problems** - Geometric reframing of all 7 problems
8. **Self-Observation** - AI observing its own processing
9. **Feelings & Choice** - Miller's Law, subharmonic resonance

### Key Metrics

- **ΔΦ:** Should be negative (< 0)
- **Dimensions:** Should cluster at 8n
- **Convergence:** Should be <10 iterations
- **Consistency:** Should be >90% across media

---

**Ready to validate? Run: `python run_all_tests.py`**

