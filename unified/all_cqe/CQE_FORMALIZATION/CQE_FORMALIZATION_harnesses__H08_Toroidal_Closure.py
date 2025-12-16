#!/usr/bin/env python3.11
"""
Test Harness H08: Toroidal Closure Validation
Validates that computational paths form closed loops on a toroidal manifold.
"""

import numpy as np
import sys

def test_toroidal_closure():
    """
    Test that a path on a torus returns to its starting point.
    """
    # Define a simple toroidal path using parametric equations
    # x = (R + r*cos(v))*cos(u)
    # y = (R + r*cos(v))*sin(u)
    # z = r*sin(v)
    # where R is the major radius, r is the minor radius
    
    R = 3.0  # Major radius
    r = 1.0  # Minor radius
    
    # Generate a closed path by varying u from 0 to 2π
    num_points = 100
    u_values = np.linspace(0, 2*np.pi, num_points)
    v = np.pi/4  # Fixed v value
    
    # Calculate points on the torus
    x = (R + r*np.cos(v)) * np.cos(u_values)
    y = (R + r*np.cos(v)) * np.sin(u_values)
    z = r * np.sin(v) * np.ones_like(u_values)
    
    # Check if the path closes (first point equals last point)
    start_point = np.array([x[0], y[0], z[0]])
    end_point = np.array([x[-1], y[-1], z[-1]])
    
    distance = np.linalg.norm(end_point - start_point)
    
    # The path should close within numerical tolerance
    tolerance = 1e-10
    
    if distance < tolerance:
        print(f"✅ PASSED: Toroidal closure validated (distance: {distance:.2e})")
        return True
    else:
        print(f"❌ FAILED: Path does not close (distance: {distance:.2e})")
        return False

if __name__ == "__main__":
    result = test_toroidal_closure()
    sys.exit(0 if result else 1)
