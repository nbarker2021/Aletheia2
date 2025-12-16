# Create E8 embedding generator
e8_embedding_code = '''"""
E₈ Lattice Embedding Generator

Generates the complete 240 root system and 8×8 Cartan matrix for the E₈ lattice,
serving as the fundamental 8-dimensional configuration space for CQE operations.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Tuple

def generate_e8_roots() -> List[List[float]]:
    """Generate the 240 E₈ root vectors (8-dimensional)."""
    roots = []
    
    # Type I: ±e_i ± e_j (112 roots)
    for i in range(8):
        for j in range(i+1, 8):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    v = [0.0] * 8
                    v[i], v[j] = float(s1), float(s2)
                    roots.append(v)
    
    # Type II: (±½,±½,±½,±½,±½,±½,±½,±½) with even number of minus signs (128 roots)
    for mask in range(1 << 8):
        v = [(-1.0)**((mask >> k) & 1) * 0.5 for k in range(8)]
        if v.count(-0.5) % 2 == 0:
            roots.append(v)
            if len(roots) == 240:
                break
    
    return roots

def generate_cartan_matrix() -> List[List[int]]:
    """Return the 8×8 E₈ Cartan matrix."""
    return [
        [ 2, -1,  0,  0,  0,  0,  0,  0],
        [-1,  2, -1,  0,  0,  0,  0,  0],
        [ 0, -1,  2, -1,  0,  0,  0,  0],
        [ 0,  0, -1,  2, -1,  0,  0,  0],
        [ 0,  0,  0, -1,  2, -1,  0, -1],
        [ 0,  0,  0,  0, -1,  2, -1,  0],
        [ 0,  0,  0,  0,  0, -1,  2,  0],
        [ 0,  0,  0,  0, -1,  0,  0,  2]
    ]

def validate_e8_structure(roots: List[List[float]], cartan: List[List[int]]) -> bool:
    """Validate the E₈ structure properties."""
    # Check root count
    if len(roots) != 240:
        return False
    
    # Check root dimension
    if not all(len(root) == 8 for root in roots):
        return False
    
    # Check Cartan matrix shape
    if len(cartan) != 8 or not all(len(row) == 8 for row in cartan):
        return False
    
    # Verify some root norms (should be 2.0)
    for root in roots[:10]:  # Check first 10
        norm_sq = sum(x*x for x in root)
        if abs(norm_sq - 2.0) > 1e-10:
            return False
    
    return True

def save_embedding(output_path: str = "embeddings/e8_248_embedding.json") -> None:
    """Generate and save the E₈ embedding data."""
    roots = generate_e8_roots()
    cartan = generate_cartan_matrix()
    
    if not validate_e8_structure(roots, cartan):
        raise ValueError("Generated E₈ structure failed validation")
    
    data = {
        "name": "E8_lattice",
        "dimension": 8,
        "root_count": len(roots),
        "roots_8d": roots,
        "cartan_8x8": cartan,
        "metadata": {
            "generated_by": "CQE-MORSR Framework",
            "description": "Complete E₈ root system and Cartan matrix",
            "validation_passed": True
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"E₈ embedding saved to {output_path}")
    print(f"Generated {len(roots)} roots with 8×8 Cartan matrix")

def load_embedding(path: str = "embeddings/e8_248_embedding.json") -> dict:
    """Load the cached E₈ embedding."""
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    save_embedding()
'''

with open("embeddings/e8_embedding.py", 'w') as f:
    f.write(e8_embedding_code)

print("Created: embeddings/e8_embedding.py")