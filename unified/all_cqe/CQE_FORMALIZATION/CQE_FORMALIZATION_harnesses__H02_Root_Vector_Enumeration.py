#!/usr/bin/env python3.11
"""
Test Harness: H02 - Root Vector Enumeration
Validates the claim that the E8 lattice has exactly 240 root vectors.
"""

import numpy as np
from itertools import product, combinations

def is_in_e8(v):
    """Check if a vector v is in the E8 lattice."""
    v = np.array(v)
    # Check if all components are integers or all are half-integers
    is_integer = np.all(np.equal(np.mod(v, 1), 0))
    is_half_integer = np.all(np.equal(np.mod(v + 0.5, 1), 0))

    if not (is_integer or is_half_integer):
        return False

    # Check if the sum of components is an even integer
    if np.sum(v) % 2 != 0:
        return False

    return True

def test_240_root_vectors():
    """Main test function to find and count all root vectors."""
    print("--- Running Test Harness H02: Root Vector Enumeration ---")
    root_vectors = []

    # Case 1: Integer components
    # Permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    print("\nCase 1: Integer Components")
    count_case1 = 0
    for pos in combinations(range(8), 2):
        for signs in product([-1, 1], repeat=2):
            v = np.zeros(8)
            v[pos[0]] = signs[0]
            v[pos[1]] = signs[1]
            if is_in_e8(v) and np.sum(v**2) == 2:
                root_vectors.append(tuple(v))
                count_case1 += 1
    print(f"  - Found {count_case1} vectors.")
    assert count_case1 == 112, "Failed: Should be 112 integer-component root vectors"

    # Case 2: Half-integer components
    # Permutations of (±½, ±½, ±½, ±½, ±½, ±½, ±½, ±½)
    print("\nCase 2: Half-Integer Components")
    count_case2 = 0
    for signs in product([-0.5, 0.5], repeat=8):
        v = np.array(signs)
        if is_in_e8(v) and np.isclose(np.sum(v**2), 2):
            # Check for even number of negative signs
            if np.sum(np.array(v) < 0) % 2 == 0:
                 root_vectors.append(tuple(v))
                 count_case2 += 1
    print(f"  - Found {count_case2} vectors.")
    assert count_case2 == 128, "Failed: Should be 128 half-integer-component root vectors"

    # Final Verification
    total_found = len(set(root_vectors))
    print("\n--- Verification ---")
    print(f"Total unique root vectors found: {total_found}")
    assert total_found == 240, "TEST FAILED: Did not find exactly 240 root vectors."

    print("\n✅ TEST PASSED: Successfully enumerated 240 root vectors of the E8 lattice.")
    print("=====================================================================")

if __name__ == "__main__":
    test_240_root_vectors()

