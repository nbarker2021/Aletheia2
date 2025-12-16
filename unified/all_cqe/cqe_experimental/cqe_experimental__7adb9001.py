#!/usr/bin/env python3
"""
Populate golden test data

Creates reference data and directory structure on cold start.
"""

import os
import json
from pathlib import Path
import sys


def populate_golden_data():
    """Populate golden test data and directory structure"""

    print("=== Populating Golden Test Data ===\n")

    base_dir = Path("data/golden")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create golden overlay samples
    golden_overlays = [
        {
            "name": "scientific_abstract_1",
            "content": "Quantum entanglement demonstrates non-local correlations between spatially separated particles through Bell inequality violations.",
            "domain": "text",
            "expected_cartan_active": {"min": 6, "max": 8},
            "expected_phi_range": {"min": 45.0, "max": 60.0}
        },
        {
            "name": "mathematical_proof",
            "content": "The Fundamental Theorem of Calculus establishes that differentiation and integration are inverse operations.",
            "domain": "text",
            "expected_cartan_active": {"min": 7, "max": 8},
            "expected_phi_range": {"min": 50.0, "max": 55.0}
        },
        {
            "name": "code_snippet",
            "content": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "domain": "text",
            "expected_cartan_active": {"min": 7, "max": 8},
            "expected_phi_range": {"min": 48.0, "max": 58.0}
        }
    ]

    golden_file = base_dir / "golden_overlays.json"
    with open(golden_file, "w") as f:
        json.dump(golden_overlays, f, indent=2)

    print(f"✓ Created golden overlays: {golden_file}")

    # Create test fixtures directory
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    fixtures_file = fixtures_dir / "test_data.json"
    test_data = {
        "sample_texts": [
            "Machine learning enables pattern recognition.",
            "Neural networks approximate complex functions.",
            "Quantum computing exploits superposition."
        ],
        "expected_results": {
            "min_cartan_active": 5,
            "max_phi": 100.0
        }
    }

    with open(fixtures_file, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"✓ Created test fixtures: {fixtures_file}")

    # Create .gitkeep files for empty data directories
    for subdir in ["overlays", "rag", "checkpoints"]:
        gitkeep_path = Path(f"data/{subdir}/.gitkeep")
        gitkeep_path.parent.mkdir(parents=True, exist_ok=True)
        gitkeep_path.touch()
        print(f"✓ Created {gitkeep_path}")

    print("\n✓ Golden data population complete!")
    return 0


if __name__ == "__main__":
    sys.exit(populate_golden_data())
