def demonstrate_sacred_geometry_cqe():
    """Demonstrate the sacred geometry enhanced CQE system"""
    
    print("Sacred Geometry Enhanced CQE System Demonstration")
    print("=" * 60)
    
    # Initialize system
    sacred_cqe = SacredGeometryEnhancedCQE()
    
    # Test with sacred frequencies
    sacred_frequencies = [432, 528, 396, 741, 852, 963]
    
    print("\n1. Sacred Frequency Analysis:")
    for freq in sacred_frequencies:
        atom = sacred_cqe.create_sacred_atom(freq)
        print(f"  {freq} Hz -> Digital Root: {atom.digital_root}, Pattern: {atom.rotational_pattern.value}")
        print(f"    Classification: {atom.carlson_classification}")
        print(f"    Resonance: {atom.resonance_alignment}")
    
    # Test time-space integration
    print("\n2. Time-Space Integration:")
    time_data = [1, 2, 4, 8, 16, 32]  # Doubling sequence
    space_data = [3, 6, 12, 24, 48, 96]  # Tripling sequence
    
    combined_embeddings = sacred_cqe.embed_temporal_patterns_in_e8(time_data, space_data)
    print(f"  Combined {len(combined_embeddings)} time-space embeddings")
    print(f"  First embedding shape: {combined_embeddings[0].shape}")
    
    # Analyze natural constants
    print("\n3. Natural Constants Analysis:")
    constants_analysis = sacred_cqe.analyze_natural_constants()
    
    for constant_name, analysis in constants_analysis.items():
        print(f"  {constant_name}:")
        print(f"    Digital Root: {analysis['digital_root']}")
        print(f"    Pattern: {analysis['rotational_pattern']}")
        print(f"    Sacred Frequency: {analysis['sacred_frequency']} Hz")
        print(f"    Classification: {analysis['carlson_classification']}")
    
    print("\n4. Sacred Geometry Validation:")
    
    # Test 9/6 pattern recognition
    test_values = [9, 18, 27, 6, 12, 24, 3, 21, 30]
    
    for value in test_values:
        atom = sacred_cqe.create_sacred_atom(value)
        expected_pattern = "INWARD" if value % 9 == 0 else ("OUTWARD" if value % 6 == 0 else "CREATIVE")
        actual_pattern = atom.rotational_pattern.value
        
        match = "✓" if expected_pattern in actual_pattern else "✗"
        print(f"  {value} -> Expected: {expected_pattern}, Got: {actual_pattern} {match}")
    
    print("\nSacred Geometry Enhanced CQE System Demonstration Complete!")

if __name__ == "__main__":
    demonstrate_sacred_geometry_cqe()
#!/usr/bin/env python3
"""
Detailed Example: Semantic Extraction from Geometric Processing
Demonstrates how CQE OS extracts meaning from E₈ lattice configurations
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Any
