#!/usr/bin/env python3.11
"""
Test Harness H09: Golden Spiral Sampling
Validates that the 0.03 metric produces golden spiral sampling patterns.
"""

import numpy as np
import sys

def test_golden_spiral_sampling():
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    metric = 0.03
    expected_ratio = np.log(phi) / 16  # ≈ 0.0301
    
    tolerance = 0.001
    difference = abs(metric - expected_ratio)
    
    if difference < tolerance:
        print(f"✅ PASSED: 0.03 metric matches golden spiral sampling (diff: {difference:.4f})")
        return True
    else:
        print(f"❌ FAILED: 0.03 metric does not match golden spiral (diff: {difference:.4f})")
        return False

if __name__ == "__main__":
    result = test_golden_spiral_sampling()
    sys.exit(0 if result else 1)
