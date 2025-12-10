class TestMORSRExplorer:
    """Test MORSR exploration algorithm."""
    
    def setup_method(self):
        # Create mock components
        self.temp_dir = tempfile.mkdtemp()
        self.embedding_path = Path(self.temp_dir) / "test_e8_embedding.json"
        
        mock_data = {
            "roots_8d": np.random.randn(240, 8).tolist(),
            "cartan_8x8": np.eye(8).tolist()
        }
        
        with open(self.embedding_path, 'w') as f:
            json.dump(mock_data, f)
        
        self.e8_lattice = E8Lattice(str(self.embedding_path))
        self.parity_channels = ParityChannels()
        self.objective_function = CQEObjectiveFunction(self.e8_lattice, self.parity_channels)
        self.morsr_explorer = MORSRExplorer(self.objective_function, self.parity_channels)
    
    def test_exploration(self):
        """Test MORSR exploration."""
        initial_vector = np.random.randn(8)
        reference_channels = {"channel_1": 0.5, "channel_2": 0.3}
        
        best_vector, best_channels, best_score = self.morsr_explorer.explore(
            initial_vector, reference_channels, max_iterations=10
        )
        
        assert len(best_vector) == 8
        assert isinstance(best_channels, dict)
        assert isinstance(best_score, float)
        assert len(self.morsr_explorer.exploration_history) > 0
    
    def test_pulse_exploration(self):
        """Test pulse exploration."""
        test_vector = np.random.randn(8)
        reference_channels = {"channel_1": 0.5}
        
        results = self.morsr_explorer.pulse_exploration(
            test_vector, reference_channels, pulse_count=5
        )
        
        assert len(results) == 5
        assert all(len(result[0]) == 8 for result in results)  # Vectors
        assert all(isinstance(result[1], float) for result in results)  # Scores
    
    def test_exploration_statistics(self):
        """Test exploration statistics."""
        # Run a short exploration first
        initial_vector = np.random.randn(8)
        reference_channels = {"channel_1": 0.5}
        
        self.morsr_explorer.explore(
            initial_vector, reference_channels, max_iterations=5
        )
        
        summary = self.morsr_explorer.get_exploration_summary()
        
        assert "total_steps" in summary
        assert "accepted_steps" in summary
        assert "acceptance_rate" in summary
        assert "best_score" in summary
