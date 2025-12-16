#!/usr/bin/env python3.11
"""
Test Harness: H07 - CRT 24-Ring Cycle Verification
Validates the claim that 24 = lcm(2,3,4,6,8) = 3×8 and gcd(3,8)=1.
"""

import math
from functools import reduce

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

def test_crt_24_ring():
    print("--- Running Test Harness H07: CRT 24-Ring Cycle ---")
    
    # Test 1: 24 = lcm(2,3,4,6,8)
    moduli = [2, 3, 4, 6, 8]
    lcm_result = reduce(lcm, moduli)
    print(f"  lcm(2,3,4,6,8) = {lcm_result}")
    assert lcm_result == 24, f"Failed: lcm is {lcm_result}, not 24"
    
    # Test 2: 24 = 3 × 8
    assert 3 * 8 == 24, "Failed: 3 × 8 ≠ 24"
    print(f"  3 × 8 = {3 * 8}")
    
    # Test 3: gcd(3, 8) = 1
    gcd_result = math.gcd(3, 8)
    print(f"  gcd(3, 8) = {gcd_result}")
    assert gcd_result == 1, f"Failed: gcd(3,8) is {gcd_result}, not 1"
    
    # Test 4: Unique addressing
    # Each ring r ∈ [0, 23] maps to unique (r mod 3, r mod 8) pair
    pairs = set()
    for r in range(24):
        pair = (r % 3, r % 8)
        assert pair not in pairs, f"Failed: Ring {r} has duplicate pair {pair}"
        pairs.add(pair)
    print(f"  Unique (mod 3, mod 8) pairs for 24 rings: {len(pairs)}")
    
    print("\n✅ TEST PASSED: CRT 24-ring cycle verified.")
    print("="*70)

if __name__ == "__main__":
    test_crt_24_ring()
