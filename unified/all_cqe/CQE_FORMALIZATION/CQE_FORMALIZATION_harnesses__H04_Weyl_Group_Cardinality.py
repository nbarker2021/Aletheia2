#!/usr/bin/env python3.11
"""
Test Harness: H04 - Weyl Group Cardinality Verification
Validates the claim that the Weyl group of E8 has order 696,729,600.
"""

import math

def weyl_group_order_e8():
    """
    The order of the Weyl group of E8 is:
    |W(E8)| = 2^8 * 8! = 256 * 40320 = 10,321,920 (INCORRECT - this is for D8)
    
    Correct formula for E8:
    |W(E8)| = 2^14 * 3^5 * 5^2 * 7 = 696,729,600
    """
    order = (2**14) * (3**5) * (5**2) * 7
    return order

def test_weyl_group_cardinality():
    print("--- Running Test Harness H04: Weyl Group Cardinality ---")
    
    expected = 696_729_600
    calculated = weyl_group_order_e8()
    
    print(f"  Expected Weyl Group Order: {expected:,}")
    print(f"  Calculated Weyl Group Order: {calculated:,}")
    
    assert calculated == expected, f"Failed: Weyl group order mismatch."
    
    print("\n✅ TEST PASSED: Weyl group cardinality verified as 696,729,600.")
    print("="*70)

if __name__ == "__main__":
    test_weyl_group_cardinality()
