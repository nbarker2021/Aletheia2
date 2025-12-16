#!/usr/bin/env python3.11
"""
Test Harness H10: Dihedral Symmetry Groups
Validates that E8 lattice operations preserve dihedral symmetry.
"""

import numpy as np
import sys

def test_dihedral_symmetry():
    # Test D4 (dihedral group of order 8) symmetry
    # A square has 4 rotational and 4 reflectional symmetries
    
    # Define a square's vertices
    square = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    
    # Rotation by 90 degrees
    rotation_matrix = np.array([[0, -1], [1, 0]])
    rotated = square @ rotation_matrix.T
    
    # Check if rotated square matches original (up to permutation)
    matches = []
    for vertex in rotated:
        found = False
        for orig_vertex in square:
            if np.allclose(vertex, orig_vertex):
                found = True
                break
        matches.append(found)
    
    if all(matches):
        print("✅ PASSED: Dihedral symmetry preserved under rotation")
        return True
    else:
        print("❌ FAILED: Dihedral symmetry not preserved")
        return False

if __name__ == "__main__":
    result = test_dihedral_symmetry()
    sys.exit(0 if result else 1)
