class TestValidationFramework:
    """Test validation framework."""
    
    def setup_method(self):
        self.validator = ValidationFramework()
    
    def test_solution_validation(self):
        """Test comprehensive solution validation."""
        # Mock problem and solution
        problem = {"complexity_class": "P", "size": 50}
        solution_vector = np.random.randn(8)
        
        # Mock analysis
        analysis = {
            "embedding_quality": {
                "optimal": {
                    "nearest_root_distance": 0.5,
                    "chamber_depth": 0.3,
                    "symmetry_score": 0.4,
                    "fundamental_chamber": True
                }
            },
            "objective_breakdown": {
                "phi_total": 0.7,
                "lattice_quality": 0.8,
                "parity_consistency": 0.6,
                "chamber_stability": 0.7,
                "geometric_separation": 0.5,
                "domain_coherence": 0.6
            },
            "chamber_analysis": {"optimal_chamber": "11111111"},
            "geometric_metrics": {
                "convergence_quality": "good",
                "vector_improvement": 1.0
            }
        }
        
        validation_report = self.validator.validate_solution(problem, solution_vector, analysis)
        
        assert "overall_score" in validation_report
        assert "validation_category" in validation_report
        assert "dimension_scores" in validation_report
        assert 0 <= validation_report["overall_score"] <= 1
    
    def test_baseline_comparison(self):
        """Test baseline comparison generation."""
        test_vector = np.random.randn(8)
        
        comparison = self.validator.generate_baseline_comparison(test_vector, n_baselines=100)
        
        assert "baseline_count" in comparison
        assert "solution_metrics" in comparison
        assert "baseline_statistics" in comparison
        assert "percentile_rankings" in comparison
        assert comparison["baseline_count"] == 100
