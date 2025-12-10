class TestCQESystem:
    """Test complete CQE system integration."""
    
    def setup_method(self):
        # Create mock E₈ embedding
        self.temp_dir = tempfile.mkdtemp()
        self.embedding_path = Path(self.temp_dir) / "test_e8_embedding.json"
        
        mock_data = {
            "roots_8d": np.random.randn(240, 8).tolist(),
            "cartan_8x8": np.eye(8).tolist()
        }
        
        with open(self.embedding_path, 'w') as f:
            json.dump(mock_data, f)
        
        # Initialize system with mock embedding
        config = {
            "exploration": {"max_iterations": 10},
            "output": {"save_results": False, "verbose": False},
            "validation": {"run_tests": False}
        }
        
        self.system = CQESystem(str(self.embedding_path), config)
    
    def test_computational_problem_solving(self):
        """Test solving computational problems."""
        problem = {
            "complexity_class": "P",
            "size": 50,
            "complexity_hint": 1
        }
        
        solution = self.system.solve_problem(problem, "computational")
        
        assert "objective_score" in solution
        assert "analysis" in solution
        assert "recommendations" in solution
        assert "computation_time" in solution
        assert solution["domain_type"] == "computational"
    
    def test_optimization_problem_solving(self):
        """Test solving optimization problems."""
        problem = {
            "variables": 10,
            "constraints": 5,
            "objective_type": "linear"
        }
        
        solution = self.system.solve_problem(problem, "optimization")
        
        assert "objective_score" in solution
        assert solution["domain_type"] == "optimization"
    
    def test_creative_problem_solving(self):
        """Test solving creative problems."""
        problem = {
            "scene_complexity": 50,
            "narrative_depth": 25,
            "character_count": 5
        }
        
        solution = self.system.solve_problem(problem, "creative")
        
        assert "objective_score" in solution
        assert solution["domain_type"] == "creative"
    
    def test_system_test_suite(self):
        """Test system test suite."""
        test_results = self.system.run_test_suite()
        
        assert isinstance(test_results, dict)
        assert "e8_embedding_load" in test_results
        assert "domain_adaptation" in test_results
        assert "parity_extraction" in test_results
    
    def test_performance_benchmark(self):
        """Test performance benchmarking."""
        benchmark_results = self.system.benchmark_performance([10, 25])
        
        assert "problem_sizes" in benchmark_results
        assert "computation_times" in benchmark_results
        assert "objective_scores" in benchmark_results
        assert len(benchmark_results["computation_times"]) == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
from .framework import ValidationFramework
__all__ = ["ValidationFramework"]
"""
CQE Validation Framework

Comprehensive validation system for assessing CQE solutions across multiple dimensions:
- Mathematical validity
- Computational evidence  
- Statistical significance
- Geometric consistency
- Cross-validation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
from scipy import stats
