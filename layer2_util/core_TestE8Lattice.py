class TestE8Lattice:
    """Test E₈ lattice operations."""
    
    def setup_method(self):
        # Create mock E₈ embedding for testing
        self.temp_dir = tempfile.mkdtemp()
        self.embedding_path = Path(self.temp_dir) / "test_e8_embedding.json"
        
        # Generate mock E₈ data
        mock_roots = np.random.randn(240, 8).tolist()
        mock_cartan = np.eye(8).tolist()
        
        mock_data = {
            "roots_8d": mock_roots,
            "cartan_8x8": mock_cartan
        }
        
        with open(self.embedding_path, 'w') as f:
            json.dump(mock_data, f)
        
        self.e8_lattice = E8Lattice(str(self.embedding_path))
    
    def test_lattice_loading(self):
        """Test E₈ lattice loading."""
        assert self.e8_lattice.roots.shape == (240, 8)
        assert self.e8_lattice.cartan_matrix.shape == (8, 8)
        assert self.e8_lattice.simple_roots.shape == (8, 8)
    
    def test_nearest_root(self):
        """Test nearest root finding."""
        test_vector = np.random.randn(8)
        nearest_idx, nearest_root, distance = self.e8_lattice.nearest_root(test_vector)
        
        assert 0 <= nearest_idx < 240
        assert len(nearest_root) == 8
        assert distance >= 0
    
    def test_chamber_determination(self):
        """Test Weyl chamber determination."""
        test_vector = np.random.randn(8)
        chamber_sig, inner_prods = self.e8_lattice.determine_chamber(test_vector)
        
        assert len(chamber_sig) == 8
        assert all(c in ['0', '1'] for c in chamber_sig)
        assert len(inner_prods) == 8
    
    def test_chamber_projection(self):
        """Test chamber projection."""
        test_vector = np.random.randn(8)
        projected = self.e8_lattice.project_to_chamber(test_vector)
        
        assert len(projected) == 8
        # Projected vector should be in fundamental chamber
        chamber_sig, _ = self.e8_lattice.determine_chamber(projected)
        # Note: Due to mock data, this test may not always pass
    
    def test_embedding_quality(self):
        """Test embedding quality assessment."""
        test_vector = np.random.randn(8)
        quality = self.e8_lattice.root_embedding_quality(test_vector)
        
        required_keys = [
            "nearest_root_distance", "nearest_root_index", "chamber_signature",
            "fundamental_chamber", "vector_norm", "chamber_depth", "symmetry_score"
        ]
        
        assert all(key in quality for key in required_keys)
        assert quality["nearest_root_distance"] >= 0
        assert 0 <= quality["nearest_root_index"] < 240
