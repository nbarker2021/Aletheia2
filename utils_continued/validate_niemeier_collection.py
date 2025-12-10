def validate_niemeier_collection(data_path="../embeddings/niemeier_lattices.json"):
    """Validate the generated Niemeier lattice collection."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print("Validating Niemeier lattice collection...")
    
    valid_count = 0
    for name, lattice in data.items():
        if 'error' in lattice:
            print(f"  {name}: FAILED - {lattice['error']}")
        else:
            # Basic validation
            gram = lattice['gram_matrix']
            if len(gram) == 24 and all(len(row) == 24 for row in gram):
                valid_count += 1
                print(f"  {name}: OK (det={lattice.get('determinant', 'unknown')})")
            else:
                print(f"  {name}: FAILED - Invalid Gram matrix shape")
    
    print(f"\\nValidation complete: {valid_count}/24 lattices valid")
    return valid_count == 24

if __name__ == "__main__":
    generate_all_niemeier_lattices()
    validate_niemeier_collection()
'''

with open("sage_scripts/generate_niemeier_lattices.sage", 'w') as f:
    f.write(sage_script)

print("Created: sage_scripts/generate_niemeier_lattices.sage")# Create core CQE system modules

# 1. Domain Adapter
domain_adapter_code = '''"""
Domain Adapter for CQE System

Converts problem instances from various domains (P/NP, optimization, scenes)
into 8-dimensional feature vectors suitable for E₈ lattice embedding.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import hashlib
