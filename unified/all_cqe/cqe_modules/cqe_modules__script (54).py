# Create comprehensive test suite
test_e8_code = '''"""
Test E₈ Embedding Generation and Operations
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.e8_embedding import generate_e8_roots, generate_cartan_matrix, save_embedding, load_embedding
from cqe_system.e8_lattice import E8Lattice

class TestE8Embedding:
    """Test E₈ embedding generation and validation."""
    
    def test_root_generation(self):
        """Test E₈ root system generation."""
        roots = generate_e8_roots()
        
        # Check count
        assert len(roots) == 240, f"Expected 240 roots, got {len(roots)}"
        
        # Check dimension
        for root in roots:
            assert len(root) == 8, f"Root dimension should be 8, got {len(root)}"
        
        # Check root norms (should be 2.0 for E₈)
        for i, root in enumerate(roots[:10]):  # Check first 10
            norm_sq = sum(x*x for x in root)
            assert abs(norm_sq - 2.0) < 1e-10, f"Root {i} has incorrect norm: {norm_sq}"
    
    def test_cartan_matrix(self):
        """Test Cartan matrix generation."""
        cartan = generate_cartan_matrix()
        
        # Check shape
        assert len(cartan) == 8, "Cartan matrix should be 8×8"
        assert all(len(row) == 8 for row in cartan), "Cartan matrix should be 8×8"
        
        # Check diagonal elements (should be 2)
        for i in range(8):
            assert cartan[i][i] == 2, f"Diagonal element {i} should be 2"
        
        # Check symmetry
        for i in range(8):
            for j in range(8):
                assert cartan[i][j] == cartan[j][i], f"Cartan matrix not symmetric at ({i},{j})"
    
    def test_embedding_save_load(self):
        """Test saving and loading E₈ embedding."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save embedding
            save_embedding(temp_path)
            assert Path(temp_path).exists(), "Embedding file was not created"
            
            # Load embedding
            data = load_embedding(temp_path)
            
            # Validate loaded data
            assert "roots_8d" in data, "Missing roots_8d in loaded data"
            assert "cartan_8x8" in data, "Missing cartan_8x8 in loaded data"
            assert len(data["roots_8d"]) == 240, "Incorrect number of roots in loaded data"
            assert len(data["cartan_8x8"]) == 8, "Incorrect Cartan matrix size"
            
        finally:
            # Cleanup
            if Path(temp_path).exists():
                Path(temp_path).unlink()

class TestE8Lattice:
    """Test E₈ lattice operations."""
    
    @pytest.fixture
    def temp_embedding(self):
        """Create temporary E₈ embedding for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        save_embedding(temp_path)
        yield temp_path
        
        # Cleanup
        if Path(temp_path).exists():
            Path(temp_path).unlink()
    
    def test_lattice_initialization(self, temp_embedding):
        """Test E₈ lattice initialization."""
        lattice = E8Lattice(temp_embedding)
        
        assert lattice.roots is not None, "Roots not loaded"
        assert lattice.cartan_matrix is not None, "Cartan matrix not loaded"
        assert lattice.simple_roots is not None, "Simple roots not set up"
        assert lattice.roots.shape == (240, 8), f"Incorrect roots shape: {lattice.roots.shape}"
    
    def test_nearest_root(self, temp_embedding):
        """Test nearest root finding."""
        lattice = E8Lattice(temp_embedding)
        
        # Test with a root vector (should find itself)
        test_root = lattice.roots[0]
        nearest_idx, nearest_root, distance = lattice.nearest_root(test_root)
        
        assert nearest_idx == 0, f"Should find root 0, got {nearest_idx}"
        assert distance < 1e-10, f"Distance to same root should be 0, got {distance}"
        
        # Test with random vector
        random_vector = np.random.randn(8)
        nearest_idx, nearest_root, distance = lattice.nearest_root(random_vector)
        
        assert 0 <= nearest_idx < 240, f"Invalid root index: {nearest_idx}"
        assert distance >= 0, f"Distance should be non-negative: {distance}"
    
    def test_chamber_determination(self, temp_embedding):
        """Test Weyl chamber determination."""
        lattice = E8Lattice(temp_embedding)
        
        # Test with zero vector
        zero_vector = np.zeros(8)
        chamber_sig, inner_prods = lattice.determine_chamber(zero_vector)
        
        assert len(chamber_sig) == 8, f"Chamber signature should have 8 bits"
        assert len(inner_prods) == 8, f"Should have 8 inner products"
        
        # Test with positive vector (should be in fundamental chamber)
        positive_vector = np.ones(8) * 0.1
        chamber_sig, inner_prods = lattice.determine_chamber(positive_vector)
        
        # Should be in fundamental chamber (all positive)
        assert chamber_sig == "11111111", f"Positive vector should be in fundamental chamber"
    
    def test_chamber_projection(self, temp_embedding):
        """Test projection to Weyl chamber."""
        lattice = E8Lattice(temp_embedding)
        
        # Test projection to fundamental chamber
        random_vector = np.random.randn(8)
        projected = lattice.project_to_chamber(random_vector)
        
        # Verify projection is in target chamber
        chamber_sig, _ = lattice.determine_chamber(projected)
        assert chamber_sig == "11111111", "Projection should be in fundamental chamber"
    
    def test_embedding_quality(self, temp_embedding):
        """Test embedding quality assessment."""
        lattice = E8Lattice(temp_embedding)
        
        test_vector = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        quality = lattice.root_embedding_quality(test_vector)
        
        # Check required fields
        required_fields = [
            "nearest_root_distance", "nearest_root_index", "chamber_signature",
            "fundamental_chamber", "vector_norm", "chamber_depth", "symmetry_score"
        ]
        
        for field in required_fields:
            assert field in quality, f"Missing quality field: {field}"
        
        assert isinstance(quality["fundamental_chamber"], bool)
        assert quality["nearest_root_distance"] >= 0
        assert 0 <= quality["nearest_root_index"] < 240
'''

with open("tests/test_e8_embedding.py", 'w') as f:
    f.write(test_e8_code)

print("Created: tests/test_e8_embedding.py")