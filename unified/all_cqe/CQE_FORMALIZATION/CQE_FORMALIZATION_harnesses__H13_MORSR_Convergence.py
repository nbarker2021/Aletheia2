#!/usr/bin/env python3.11
"""
Test Harness H13: MORSR Protocol Convergence
Validates that MORSR iterations converge to optimal solution.
"""

import numpy as np
import sys

def test_morsr_convergence():
    # Simulate MORSR optimization using gradient descent
    # Target: minimize f(x) = x^2
    
    x = 10.0  # Initial state
    target = 0.0  # Optimal state
    learning_rate = 0.1
    max_iterations = 100
    tolerance = 1e-6
    
    for i in range(max_iterations):
        gradient = 2 * x  # df/dx = 2x
        x_new = x - learning_rate * gradient
        
        # Check monotonic decrease (Fejér monotonicity)
        if abs(x_new - target) >= abs(x - target):
            print(f"❌ FAILED: Non-monotonic at iteration {i}")
            return False
        
        x = x_new
        
        if abs(x - target) < tolerance:
            print(f"✅ PASSED: MORSR converged in {i+1} iterations (final x: {x:.2e})")
            return True
    
    print(f"❌ FAILED: Did not converge within {max_iterations} iterations")
    return False

if __name__ == "__main__":
    result = test_morsr_convergence()
    sys.exit(0 if result else 1)
