class TestObjectiveFunction:
    """Test CQE objective function."""
    
    def setup_method(self):
        # Create mock components
        self.temp_dir = tempfile.mkdtemp()
        self.embedding_path = Path(self.temp_dir) / "test_e8_embedding.json"
        
        # Generate mock E₈ data
        mock_data = {
            "roots_8d": np.random.randn(240, 8).tolist(),
            "cartan_8x8": np.eye(8).tolist()
        }
        
        with open(self.embedding_path, 'w') as f:
            json.dump(mock_data, f)
        
        self.e8_lattice = E8Lattice(str(self.embedding_path))
        self.parity_channels = ParityChannels()
        self.objective_function = CQEObjectiveFunction(self.e8_lattice, self.parity_channels)
    
    def test_objective_evaluation(self):
        """Test objective function evaluation."""
        test_vector = np.random.randn(8)
        reference_channels = {"channel_1": 0.5, "channel_2": 0.3}
        
        scores = self.objective_function.evaluate(test_vector, reference_channels)
        
        required_keys = [
            "phi_total", "lattice_quality", "parity_consistency",
            "chamber_stability", "geometric_separation", "domain_coherence"
        ]
        
        assert all(key in scores for key in required_keys)
        assert all(0 <= scores[key] <= 1 for key in required_keys)
    
    def test_gradient_calculation(self):
        """Test gradient calculation."""
        test_vector = np.random.randn(8)
        reference_channels = {"channel_1": 0.5}
        
        gradient = self.objective_function.gradient(test_vector, reference_channels)
        
        assert len(gradient) == 8
        assert not np.allclose(gradient, 0)  # Should have non-zero gradient
    
    def test_improvement_direction(self):
        """Test improvement direction suggestion."""
        test_vector = np.random.randn(8)
        reference_channels = {"channel_1": 0.5}
        
        direction, reasoning = self.objective_function.suggest_improvement_direction(
            test_vector, reference_channels
        )
        
        assert len(direction) == 8
        assert isinstance(reasoning, dict)
        assert np.linalg.norm(direction) <= 1.0  # Should be normalized
