def load_embedding(path: str = "embeddings/e8_248_embedding.json") -> dict:
    """Load the cached E₈ embedding."""
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    save_embedding()
'''

with open("embeddings/e8_embedding.py", 'w') as f:
    f.write(e8_embedding_code)

print("Created: embeddings/e8_embedding.py")# Create SageMath Niemeier lattice generator
sage_script = '''"""
Niemeier Lattice Generator for SageMath

Generates all 24 unique 24-dimensional perfect lattices using Conway's "holy" construction
methods. Each lattice is characterized by 10 Conway-Golay-Monster seed nodes and specific
glue code patterns that extend E₈ faces into 24D space.
"""

import json
from sage.all import NiemeierLattice

# The 24 Niemeier lattice names in standard notation
NIEMEIER_NAMES = [
    "A1^24", "A2^12", "A3^8", "A4^6", "A5^4D4", "A6^4", 
    "A7^2D5^2", "A8^3", "D4^6", "D6^4", "D8^3", "D10^2E7^2", 
    "D12^2", "D24", "E6^4", "E7^2D10", "E8^3", "Leech", 
    "A3D21", "A1E7^3", "A2E6^3", "A4D4^3", "A5D5^2", "A11D7E6"
]
