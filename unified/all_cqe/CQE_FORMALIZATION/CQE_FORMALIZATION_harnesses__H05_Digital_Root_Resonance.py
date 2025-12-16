#!/usr/bin/env python3.11
"""
Test Harness: H05 - Digital Root Resonance Verification
Validates the Principle of Numerical Resonance (Digital Roots 3, 6, 9).
"""

def digital_root(n):
    """Calculate the digital root of a number."""
    if n == 0:
        return 0
    return 1 + ((n - 1) % 9)

def test_digital_root_resonance():
    print("--- Running Test Harness H05: Digital Root Resonance ---")
    
    # Test key system constants
    constants = {
        'Weyl Group Cardinality': 696_729_600,
        '240 Root Vectors': 240,
        '24-Ring Cycle': 24,
        'Solfeggio 432 Hz': 432,
        'Solfeggio 528 Hz': 528,
        'Solfeggio 396 Hz': 396,
        'Solfeggio 741 Hz': 741,
    }
    
    expected_resonance = {3, 6, 9}
    
    for name, value in constants.items():
        dr = digital_root(value)
        print(f"  {name}: {value} → DR {dr}")
        assert dr in expected_resonance, f"Failed: {name} has DR {dr}, not in {{3,6,9}}"
    
    print("\n✅ TEST PASSED: All key constants exhibit 3-6-9 digital root resonance.")
    print("="*70)

if __name__ == "__main__":
    test_digital_root_resonance()
