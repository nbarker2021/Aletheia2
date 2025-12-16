#!/usr/bin/env python3
"""
CQE Ultimate System - Basic Usage Examples
==========================================

This file demonstrates basic usage of the CQE Ultimate System
with practical examples across different data types and applications.

Author: CQE Research Consortium
Version: 1.0.0 Complete
License: Universal Framework License
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cqe_ultimate_system import UltimateCQESystem
import time
import json

def example_1_basic_data_processing():
    """Example 1: Basic data processing with different types"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Data Processing")
    print("=" * 60)
    
    # Initialize the CQE system
    cqe = UltimateCQESystem()
    
    # Test different data types
    test_data = [
        42,                          # Integer
        "Hello, Universe!",          # String
        [1, 2, 3, 4, 5],            # List
        {"key": "value"},           # Dictionary
        3.14159,                    # Float
        complex(0.5, 0.5),          # Complex number
    ]
    
    print("Processing different data types:")
    print()
    
    for i, data in enumerate(test_data, 1):
        print(f"Data {i}: {data} ({type(data).__name__})")
        
        # Process using geometry-first paradigm
        result = cqe.process_data_geometry_first(data)
        
        # Extract key results
        geo_result = result['geometric_result']
        sacred = geo_result['sacred_geometry']
        fractal = geo_result['fractal_analysis']
        toroidal = geo_result['toroidal_analysis']
        
        print(f"  Digital Root: {sacred['digital_root']}")
        print(f"  Sacred Frequency: {sacred['sacred_frequency']} Hz")
        print(f"  Rotational Pattern: {sacred['rotational_pattern']}")
        print(f"  Fractal Behavior: {fractal['behavior']}")
        print(f"  Force Type: {toroidal['force_type']}")
        print(f"  Compression Ratio: {result['storage_efficiency']['compression_ratio']:.3f}")
        print()
    
    print(f"Total atoms created: {len(cqe.atoms)}")
    print()

def example_2_sacred_frequency_analysis():
    """Example 2: Sacred frequency analysis"""
    print("=" * 60)
    print("EXAMPLE 2: Sacred Frequency Analysis")
    print("=" * 60)
    
    cqe = UltimateCQESystem()
    
    # Analyze all sacred frequencies
    sacred_frequencies = [174, 285, 396, 417, 528, 639, 741, 852, 963]
    
    print("Sacred Frequency Analysis:")
    print("Freq (Hz) | Digital Root | Pattern      | Force Type")
    print("-" * 55)
    
    for freq in sacred_frequencies:
        result = cqe.process_data_geometry_first(freq)
        sacred = result['geometric_result']['sacred_geometry']
        toroidal = result['geometric_result']['toroidal_analysis']
        
        print(f"{freq:8} | {sacred['digital_root']:11} | {sacred['rotational_pattern']:12} | {toroidal['force_type']}")
    
    print()
    
    # Analyze the pattern
    analysis = cqe.analyze_system_patterns()
    print("Pattern Analysis:")
    print(f"Digital Root Distribution: {analysis['digital_root_distribution']}")
    print(f"Force Classification Distribution: {analysis['force_classification_distribution']}")
    print()

def example_3_text_analysis():
    """Example 3: Text analysis across languages"""
    print("=" * 60)
    print("EXAMPLE 3: Multi-Language Text Analysis")
    print("=" * 60)
    
    cqe = UltimateCQESystem()
    
    # Text in different languages
    texts = [
        ("English", "Hello, world!"),
        ("French", "Bonjour, le monde!"),
        ("Spanish", "¡Hola, mundo!"),
        ("German", "Hallo, Welt!"),
        ("Italian", "Ciao, mondo!"),
        ("Portuguese", "Olá, mundo!"),
        ("Sacred", "Om Mani Padme Hum"),
        ("Mathematical", "E=mc²"),
    ]
    
    print("Multi-Language Text Analysis:")
    print("Language     | Text                 | Root | Freq (Hz) | Pattern")
    print("-" * 70)
    
    for language, text in texts:
        result = cqe.process_data_geometry_first(text)
        sacred = result['geometric_result']['sacred_geometry']
        
        print(f"{language:12} | {text:20} | {sacred['digital_root']:4} | {sacred['sacred_frequency']:8.0f} | {sacred['rotational_pattern']}")
    
    print()

def example_4_mathematical_constants():
    """Example 4: Mathematical constants analysis"""
    print("=" * 60)
    print("EXAMPLE 4: Mathematical Constants Analysis")
    print("=" * 60)
    
    cqe = UltimateCQESystem()
    
    # Mathematical constants
    constants = {
        "π (Pi)": 3.14159265359,
        "e (Euler)": 2.71828182846,
        "φ (Golden Ratio)": 1.61803398875,
        "√2": 1.41421356237,
        "√3": 1.73205080757,
        "√5": 2.23606797750,
        "γ (Euler-Mascheroni)": 0.57721566490,
        "α (Fine Structure)": 0.00729735257,
    }
    
    print("Mathematical Constants Analysis:")
    print("Constant              | Value        | Root | Pattern      | Force")
    print("-" * 70)
    
    for name, value in constants.items():
        result = cqe.process_data_geometry_first(value)
        sacred = result['geometric_result']['sacred_geometry']
        toroidal = result['geometric_result']['toroidal_analysis']
        
        print(f"{name:20} | {value:12.8f} | {sacred['digital_root']:4} | {sacred['rotational_pattern']:12} | {toroidal['force_type']}")
    
    print()

def example_5_atom_combination():
    """Example 5: Atom combination and compatibility"""
    print("=" * 60)
    print("EXAMPLE 5: Atom Combination and Compatibility")
    print("=" * 60)
    
    cqe = UltimateCQESystem()
    
    # Create atoms for combination
    test_data = [
        ("Sacred Frequency", 432),
        ("Healing Text", "healing"),
        ("Sacred Text", "sacred geometry"),
        ("Golden Ratio", 1.618),
        ("Creative Number", 3),
        ("Harmony List", [1, 2, 3, 5, 8]),  # Fibonacci sequence
    ]
    
    atom_ids = []
    print("Creating atoms for combination:")
    
    for name, data in test_data:
        atom_id = cqe.create_universal_atom(data)
        atom = cqe.get_atom(atom_id)
        atom_ids.append((name, atom_id, atom))
        
        print(f"  {name}: {data} → Root {atom.digital_root}, Freq {atom.sacred_frequency} Hz")
    
    print()
    print("Attempting combinations:")
    
    # Try combining compatible atoms
    combinations_attempted = 0
    combinations_successful = 0
    
    for i in range(len(atom_ids)):
        for j in range(i + 1, len(atom_ids)):
            name1, id1, atom1 = atom_ids[i]
            name2, id2, atom2 = atom_ids[j]
            
            combinations_attempted += 1
            combined_id = cqe.combine_atoms(id1, id2)
            
            if combined_id:
                combinations_successful += 1
                combined_atom = cqe.get_atom(combined_id)
                print(f"  ✓ {name1} + {name2} → Root {combined_atom.digital_root}, Freq {combined_atom.sacred_frequency} Hz")
            else:
                print(f"  ✗ {name1} + {name2} → Incompatible")
    
    print()
    print(f"Combination Results: {combinations_successful}/{combinations_attempted} successful")
    print(f"Total atoms in system: {len(cqe.atoms)}")
    print()

def example_6_performance_benchmarking():
    """Example 6: Performance benchmarking"""
    print("=" * 60)
    print("EXAMPLE 6: Performance Benchmarking")
    print("=" * 60)
    
    cqe = UltimateCQESystem()
    
    # Test different data sizes and types
    test_cases = [
        ("Small Text", ["test"] * 10),
        ("Medium Text", [f"test_string_{i}" for i in range(100)]),
        ("Numbers", list(range(100))),
        ("Complex Data", [{"id": i, "value": f"item_{i}"} for i in range(50)]),
    ]
    
    print("Performance Benchmarking:")
    print("Test Case     | Items | Time (s) | Atoms/sec | Avg Compression")
    print("-" * 65)
    
    for test_name, test_data in test_cases:
        start_time = time.time()
        
        atom_ids = []
        compression_ratios = []
        
        for data in test_data:
            atom_id = cqe.create_universal_atom(data)
            atom = cqe.get_atom(atom_id)
            atom_ids.append(atom_id)
            compression_ratios.append(atom.compression_ratio)
        
        end_time = time.time()
        
        processing_time = end_time - start_time
        atoms_per_second = len(test_data) / processing_time
        avg_compression = sum(compression_ratios) / len(compression_ratios)
        
        print(f"{test_name:12} | {len(test_data):5} | {processing_time:8.3f} | {atoms_per_second:9.1f} | {avg_compression:14.3f}")
    
    print()

def example_7_system_analysis():
    """Example 7: System analysis and patterns"""
    print("=" * 60)
    print("EXAMPLE 7: System Analysis and Patterns")
    print("=" * 60)
    
    cqe = UltimateCQESystem()
    
    # Create diverse dataset
    diverse_data = [
        # Sacred frequencies
        174, 285, 396, 417, 528, 639, 741, 852, 963,
        # Mathematical constants
        3.14159, 2.71828, 1.61803,
        # Text data
        "sacred", "geometry", "healing", "harmony", "resonance",
        # Structured data
        [1, 1, 2, 3, 5, 8], {"frequency": 432}, complex(1, 1),
        # Random data
        42, "random text", [7, 14, 21], {"test": "data"}
    ]
    
    print(f"Creating {len(diverse_data)} diverse atoms...")
    
    for data in diverse_data:
        cqe.create_universal_atom(data)
    
    # Analyze the system
    analysis = cqe.analyze_system_patterns()
    
    print("\nSystem Analysis Results:")
    print(f"Total Atoms: {analysis['total_atoms']}")
    print(f"Average Compression Ratio: {analysis['average_compression_ratio']:.3f}")
    
    print("\nDigital Root Distribution:")
    for root in sorted(analysis['digital_root_distribution'].keys()):
        count = analysis['digital_root_distribution'][root]
        percentage = (count / analysis['total_atoms']) * 100
        print(f"  Root {root}: {count} atoms ({percentage:.1f}%)")
    
    print("\nFractal Behavior Distribution:")
    for behavior, count in analysis['fractal_behavior_distribution'].items():
        percentage = (count / analysis['total_atoms']) * 100
        print(f"  {behavior}: {count} atoms ({percentage:.1f}%)")
    
    print("\nForce Classification Distribution:")
    for force, count in analysis['force_classification_distribution'].items():
        percentage = (count / analysis['total_atoms']) * 100
        print(f"  {force}: {count} atoms ({percentage:.1f}%)")
    
    print("\nAverage Validation Scores:")
    for metric, score in analysis['average_validation_scores'].items():
        status = "EXCELLENT" if score > 0.9 else "GOOD" if score > 0.8 else "ACCEPTABLE" if score > 0.7 else "NEEDS_IMPROVEMENT"
        print(f"  {metric}: {score:.3f} ({status})")
    
    print()

def example_8_export_and_persistence():
    """Example 8: System state export and persistence"""
    print("=" * 60)
    print("EXAMPLE 8: System State Export and Persistence")
    print("=" * 60)
    
    cqe = UltimateCQESystem()
    
    # Create some sample data
    sample_data = [
        "persistence test",
        432,  # Sacred frequency
        {"type": "test", "purpose": "demonstration"},
        [1, 2, 3, 5, 8, 13],  # Fibonacci
        complex(0.707, 0.707),  # Unit circle point
    ]
    
    print("Creating sample atoms for persistence test...")
    
    atom_ids = []
    for data in sample_data:
        atom_id = cqe.create_universal_atom(data)
        atom_ids.append(atom_id)
        print(f"  Created atom: {atom_id}")
    
    # Export system state
    export_filename = "example_system_state.json"
    cqe.export_system_state(export_filename)
    
    print(f"\nSystem state exported to: {export_filename}")
    
    # Verify export file
    if os.path.exists(export_filename):
        with open(export_filename, 'r') as f:
            exported_data = json.load(f)
        
        print(f"Export verification:")
        print(f"  File size: {os.path.getsize(export_filename)} bytes")
        print(f"  Atoms in export: {len(exported_data['atoms'])}")
        print(f"  Export timestamp: {exported_data['export_timestamp']}")
        print(f"  Operation mode: {exported_data['operation_mode']}")
        
        # Clean up
        os.remove(export_filename)
        print(f"  Cleaned up: {export_filename}")
    
    print()

def run_all_examples():
    """Run all basic usage examples"""
    print("CQE ULTIMATE SYSTEM - BASIC USAGE EXAMPLES")
    print("=" * 80)
    print()
    
    examples = [
        example_1_basic_data_processing,
        example_2_sacred_frequency_analysis,
        example_3_text_analysis,
        example_4_mathematical_constants,
        example_5_atom_combination,
        example_6_performance_benchmarking,
        example_7_system_analysis,
        example_8_export_and_persistence,
    ]
    
    start_time = time.time()
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
            print(f"Example {i} completed successfully.")
        except Exception as e:
            print(f"Example {i} failed with error: {e}")
        
        if i < len(examples):
            print("Press Enter to continue to next example...")
            input()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print(f"Total execution time: {total_time:.2f} seconds")
    print("The CQE Ultimate System is ready for your applications!")
    print()

if __name__ == "__main__":
    run_all_examples()
