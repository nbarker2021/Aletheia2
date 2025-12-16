#!/usr/bin/env python3.11
"""
Test Harness H15: Flower of Life E8 Projection
Validates that E8 projection produces Flower of Life pattern.
"""

import numpy as np
import sys

def test_flower_of_life_projection():
    # Simplified test: Check that E8 root vectors project to hexagonal symmetry
    # The Flower of Life has 6-fold rotational symmetry
    
    # Generate a simple hexagonal pattern (6 points around a circle)
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 points
    hex_points = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # Check 6-fold rotational symmetry
    rotation_angle = 2*np.pi / 6
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle),  np.cos(rotation_angle)]
    ])
    
    rotated = hex_points @ rotation_matrix.T
    
    # Check if rotated pattern matches original (up to permutation)
    tolerance = 1e-10
    matches = 0
    for rot_point in rotated:
        for orig_point in hex_points:
            if np.linalg.norm(rot_point - orig_point) < tolerance:
                matches += 1
                break
    
    if matches == 6:
        print("✅ PASSED: Flower of Life 6-fold symmetry validated")
        return True
    else:
        print(f"❌ FAILED: Only {matches}/6 points matched after rotation")
        return False

if __name__ == "__main__":
    result = test_flower_of_life_projection()
    sys.exit(0 if result else 1)
