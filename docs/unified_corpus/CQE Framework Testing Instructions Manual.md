# CQE Framework Testing Instructions Manual

## Overview

This manual provides comprehensive instructions for testing the Cartan Quadratic Equivalence (CQE) framework, which implements a DNA-based, self-learning, nearly entropy-free lossless encoding/decoding system with RAG-based interpretive systems and geometry embedding-based recall.

## Quick Start

### Running All Tests
```bash
python3 run_cqe_tests.py
```

### Running Quick Test Suite (Basic Validation)
```bash
python3 run_cqe_tests.py --quick
```

### Running Performance Tests Only
```bash
python3 run_cqe_tests.py --performance
```

### Viewing Testing Manual
```bash
python3 run_cqe_tests.py --manual
```

## Test Framework Components

### 1. Core Framework Tests

#### Law 1: Quadratic Invariance Tests
- **Purpose**: Validates that mathematical invariants are preserved throughout all operations
- **Tests**:
  - `test_law_1_quadratic_invariance_basic`: Basic invariant preservation
  - `test_law_1_quadratic_invariance_multiple_invariants`: Multiple invariants with different tolerances
  - `test_law_1_quadratic_invariance_edge_cases`: Zero values, negative values, tight tolerances

#### Law 2: Boundary-Only Entropy Tests
- **Purpose**: Ensures entropy changes only occur at defined boundaries
- **Tests**:
  - `test_law_2_boundary_only_entropy_basic`: Basic boundary event recording
  - `test_law_2_boundary_entropy_integrity`: Cryptographic integrity and tamper detection

#### Law 3: Auditable Governance Tests
- **Purpose**: Verifies all operations are properly audited and traceable
- **Tests**:
  - `test_law_3_auditable_governance_basic`: Basic audit trail creation
  - `test_law_3_audit_trail_completeness`: Comprehensive audit coverage

#### Law 4: Optimized Efficiency Tests
- **Purpose**: Validates optimal system efficiency
- **Tests**:
  - `test_law_4_optimized_efficiency_basic`: Lossless encoding/decoding performance
  - `test_law_4_performance_scalability`: Performance with increasing data sizes

### 2. DNA Encoding System Tests

#### Core Functionality Tests
- **Purpose**: Tests the DNA-based encoding/decoding system
- **Tests**:
  - `test_dna_encoding_all_data_types`: All Python data types
  - `test_dna_encoding_geometric_signatures`: Signature calculation and preservation
  - `test_dna_encoding_error_handling`: Error handling and recovery

### 3. Integration Tests

#### System Integration Tests
- **Purpose**: Tests complete system integration
- **Tests**:
  - `test_complete_integration_workflow`: End-to-end workflow
  - `test_system_integrity_validation`: Comprehensive integrity checks

### 4. Performance Tests

#### Stress and Performance Tests
- **Purpose**: Validates system performance under load
- **Tests**:
  - `test_performance_stress_large_data`: Large dataset performance
  - `test_concurrent_operations_simulation`: Concurrent operation simulation

## Running Individual Tests

### Using unittest directly
```bash
python3 -m unittest CQETestingFramework.test_law_1_quadratic_invariance_basic
```

### With verbose output
```bash
python3 -m unittest -v CQETestingFramework
```

## Understanding Test Results

### Success Indicators
- ✅ **ALL TESTS PASSED**: System is fully operational
- **Success Rate 100%**: All components working correctly
- **Zero Governance Violations**: All four laws enforced

### Warning Indicators
- ⚠️ **Some Tests Failed**: Review specific failures
- **Success Rate 80-99%**: Most components working, some issues
- **Minor Governance Violations**: Check invariant tolerances

### Critical Indicators
- ❌ **Many Tests Failed**: System needs attention
- **Success Rate <80%**: Major issues present
- **Multiple Governance Violations**: Core laws not enforced

## Test Configuration

### Adjusting Test Parameters
Edit `TEST_CONFIG` in the framework:
```python
TEST_CONFIG = {
    'DNA_TOLERANCE': 1e-10,          # DNA geometric signature tolerance
    'PERFORMANCE_THRESHOLD': 1.0,    # Maximum seconds for operations
    'LARGE_DATA_SIZE': 1000,         # Size for performance testing
    'VERBOSE_OUTPUT': True           # Enable detailed output
}
```

### Performance Tuning
- Increase `PERFORMANCE_THRESHOLD` for slower hardware
- Adjust `DNA_TOLERANCE` for precision requirements
- Modify `LARGE_DATA_SIZE` for stress testing

## Troubleshooting Common Issues

### "Quadratic Invariant violated"
- **Cause**: Operation modified invariant beyond tolerance
- **Solution**: Check tolerance values and operation logic
- **Prevention**: Validate operations before execution

### "DNA strand not found"
- **Cause**: Strand ID incorrect or strand deleted
- **Solution**: Verify strand IDs and encoding success
- **Prevention**: Check return values from encoding operations

### "Geometric governance violation"
- **Cause**: Operation violates geometric constraints
- **Solution**: Review operation parameters
- **Prevention**: Use governance validation before operations

### Performance test failures
- **Cause**: System resources or timing issues
- **Solution**: Adjust performance thresholds
- **Prevention**: Close other applications during testing

## Advanced Testing Scenarios

### Custom Test Data
```python
# Create custom test datasets
custom_data = {
    "your_data_type": "your_test_data",
    "complex_structure": {"nested": {"data": [1, 2, 3]}}
}

# Test encoding/decoding
strand_id = dna_encoder.encode(custom_data)
decoded_data = dna_encoder.decode(strand_id)
assert custom_data == decoded_data
```

### Custom Invariants
```python
# Create domain-specific invariants
custom_invariant = QuadraticInvariant(
    value=your_calculated_value,
    tolerance=appropriate_tolerance,
    metadata={"domain": "your_domain", "type": "custom"}
)
governance.register_invariant("custom_invariant", custom_invariant)
```

### Performance Benchmarking
```python
import time

start_time = time.time()
# Your operations here
end_time = time.time()

performance_time = end_time - start_time
print(f"Operation completed in {performance_time:.4f} seconds")
```

## Continuous Integration

### CI/CD Integration
```yaml
# Example GitHub Actions workflow
name: CQE Framework Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run CQE tests
      run: python3 run_cqe_tests.py
```

### Test Reporting
- Results saved to `cqe_test_results.json`
- Comprehensive reports include system statistics
- Performance metrics tracked over time
- Governance compliance verified

## System Monitoring

### Health Checks
```python
# Check system health
governance_stats = governance.get_governance_statistics()
encoder_stats = dna_encoder.get_encoder_statistics()

# Validate system integrity
integrity_report = governance.validate_system_integrity()
strand_validation = dna_encoder.validate_all_strands()
```

### Performance Monitoring
- Track encoding/decoding times
- Monitor success rates
- Watch for governance violations
- Analyze audit trail patterns

## Best Practices

### Before Testing
1. Ensure clean environment
2. Close unnecessary applications
3. Check available memory and disk space
4. Verify Python version compatibility

### During Testing
1. Monitor test output for warnings
2. Check performance metrics
3. Validate governance compliance
4. Review audit trail entries

### After Testing
1. Analyze test results thoroughly
2. Address any failures immediately
3. Update documentation if needed
4. Archive test reports for comparison

## Support and Documentation

### Getting Help
- Review test output logs for detailed error information
- Check the comprehensive testing manual (`--manual` flag)
- Examine audit trails for operation history
- Validate system integrity regularly

### Documentation
- All code is thoroughly notated for understanding
- Test methods include detailed docstrings
- Error messages provide context and solutions
- Performance metrics help identify bottlenecks

## Conclusion

The CQE framework testing system provides comprehensive validation of all system components, ensuring that the four fundamental laws are enforced and the DNA-based encoding system operates correctly. Regular testing helps maintain system integrity and performance while providing detailed audit trails for compliance and debugging.

For additional support or questions about testing procedures, refer to the inline documentation and error messages, which provide detailed context and guidance for resolving issues.

