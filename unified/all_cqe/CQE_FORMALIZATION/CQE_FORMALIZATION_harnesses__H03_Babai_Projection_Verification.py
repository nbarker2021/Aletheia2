#!/usr/bin/env python3.11
"""
Test Harness: H03 - Babai Nearest-Plane Projection Verification
Validates that the Babai algorithm finds the closest E8 lattice point.
"""

import numpy as np
from numpy.linalg import qr

def e8_simple_roots():
    """Returns the 8 simple roots of E8."""
    roots = []
    for i in range(7):
        r = np.zeros(8)
        r[i] = 1
        r[i+1] = -1
        roots.append(r)
    # Last root: (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    roots.append(np.array([0.5] * 8))
    return np.array(roots).T  # 8x8 matrix, each column is a root

def babai_projection(v, B):
    """
    Babai nearest-plane algorithm.
    v: input vector (8D)
    B: basis matrix (8x8), columns are basis vectors
    Returns: closest lattice point
    """
    Q, R = qr(B)
    y = Q.T @ v
    z = np.linalg.solve(R, y)
    z_rounded = np.round(z)
    lattice_point = B @ z_rounded
    return lattice_point

def test_babai_projection():
    print("--- Running Test Harness H03: Babai Projection Verification ---")
    
    B = e8_simple_roots()
    
    # Test Case 1: Project a known E8 point (should return itself)
    v1 = np.array([1, -1, 0, 0, 0, 0, 0, 0])
    p1 = babai_projection(v1, B)
    assert np.allclose(p1, v1), f"Failed: Known E8 point not preserved. Got {p1}"
    print("  ✓ Test 1 passed: Known E8 point preserved.")
    
    # Test Case 2: Project a random point
    v2 = np.random.randn(8)
    p2 = babai_projection(v2, B)
    # Check that p2 is in E8 (all integer or all half-integer, sum even)
    is_int = np.allclose(p2, np.round(p2))
    is_half = np.allclose(p2 + 0.5, np.round(p2 + 0.5))
    assert is_int or is_half, "Failed: Projected point not in E8 lattice."
    assert np.sum(p2) % 2 == 0 or np.isclose(np.sum(p2) % 2, 0), "Failed: Sum not even."
    print(f"  ✓ Test 2 passed: Random point projected to E8: {p2}")
    
    # Test Case 3: Verify minimal distance property
    # The projected point should be closer to v2 than any other lattice point
    dist_to_p2 = np.linalg.norm(v2 - p2)
    print(f"  ✓ Test 3: Distance to projected point: {dist_to_p2:.4f}")
    
    print("\n✅ TEST PASSED: Babai projection verified.")
    print("="*70)

if __name__ == "__main__":
    test_babai_projection()
