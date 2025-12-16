# Create CQE system integration tests
test_cqe_code = '''"""
Test CQE System Integration
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.e8_embedding import save_embedding
from cqe_system import (
    DomainAdapter, E8Lattice, ParityChannels, 
    CQEObjectiveFunction, MORSRExplorer, ChamberBoard, CQERunner
)

class TestCQEIntegration:
    """Integration tests for complete CQE system."""
    
    @pytest.fixture
    def cqe_system(self):
        """Set up complete CQE system for testing."""
        # Create temporary embedding
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        save_embedding(temp_path)
        
        # Initialize system components
        domain_adapter = DomainAdapter()
        e8_lattice = E8Lattice(temp_path)
        parity_channels = ParityChannels()
        objective_function = CQEObjectiveFunction(e8_lattice, parity_channels)
        morsr_explorer = MORSRExplorer(objective_function, parity_channels, random_seed=42)
        chamber_board = ChamberBoard()
        
        yield {
            "domain_adapter": domain_adapter,
            "e8_lattice": e8_lattice, 
            "parity_channels": parity_channels,
            "objective_function": objective_function,
            "morsr_explorer": morsr_explorer,
            "chamber_board": chamber_board,
            "temp_path": temp_path
        }
        
        # Cleanup
        if Path(temp_path).exists():
            Path(temp_path).unlink()
    
    def test_p_vs_np_pipeline(self, cqe_system):
        """Test complete P vs NP analysis pipeline."""
        
        # Generate P and NP problem embeddings
        p_vector = cqe_system["domain_adapter"].embed_p_problem(100, 1)
        np_vector = cqe_system["domain_adapter"].embed_np_problem(100, 0.8)
        
        # Extract parity channels
        p_channels = cqe_system["parity_channels"].extract_channels(p_vector)
        np_channels = cqe_system["parity_channels"].extract_channels(np_vector)
        
        # Evaluate with objective function
        p_scores = cqe_system["objective_function"].evaluate(
            p_vector, p_channels, {"complexity_class": "P", "domain_type": "computational"}
        )
        np_scores = cqe_system["objective_function"].evaluate(
            np_vector, np_channels, {"complexity_class": "NP", "domain_type": "computational"}
        )
        
        # Verify different scores for P vs NP
        assert "phi_total" in p_scores
        assert "phi_total" in np_scores
        assert abs(p_scores["phi_total"] - np_scores["phi_total"]) > 0.1, "P and NP should have different scores"
        
        # Test MORSR exploration on P problem
        optimized_p, opt_channels, opt_score = cqe_system["morsr_explorer"].explore(
            p_vector, p_channels, max_iterations=10
        )
        
        assert len(optimized_p) == 8, "Optimized vector should be 8-dimensional"
        assert opt_score >= p_scores["phi_total"], "MORSR should improve or maintain score"
    
    def test_chamber_board_enumeration(self, cqe_system):
        """Test chamber board gate enumeration."""
        
        # Generate gates
        gates = cqe_system["chamber_board"].enumerate_gates(max_count=20)
        
        assert len(gates) == 20, f"Should generate 20 gates, got {len(gates)}"
        
        # Validate gate structure
        for gate in gates:
            required_fields = ["construction", "policy_channel", "phase", "gate_id", "cells", "parameters"]
            for field in required_fields:
                assert field in gate, f"Gate missing field: {field}"
        
        # Test gate vector generation
        test_gate = gates[0]
        gate_vector = cqe_system["chamber_board"].generate_gate_vector(test_gate, index=0)
        
        assert len(gate_vector) == 8, "Gate vector should be 8-dimensional"
        assert np.all(gate_vector >= 0) and np.all(gate_vector <= 1), "Gate vector should be in [0,1]"
    
    def test_domain_adaptation(self, cqe_system):
        """Test domain adaptation for different problem types."""
        
        adapter = cqe_system["domain_adapter"]
        
        # Test P problem adaptation
        p_vec = adapter.embed_p_problem(50, 1)
        assert len(p_vec) == 8, "P embedding should be 8D"
        assert adapter.validate_features(p_vec), "P features should be valid"
        
        # Test optimization problem adaptation
        opt_vec = adapter.embed_optimization_problem(10, 5, "linear")
        assert len(opt_vec) == 8, "Optimization embedding should be 8D"
        assert adapter.validate_features(opt_vec), "Optimization features should be valid"
        
        # Test creative problem adaptation
        creative_vec = adapter.embed_scene_problem(30, 15, 3)
        assert len(creative_vec) == 8, "Creative embedding should be 8D"
        assert adapter.validate_features(creative_vec), "Creative features should be valid"
        
        # Test hash-based adaptation
        hash_vec = adapter.hash_to_features("test problem description")
        assert len(hash_vec) == 8, "Hash embedding should be 8D"
        assert adapter.validate_features(hash_vec), "Hash features should be valid"
    
    def test_parity_channels(self, cqe_system):
        """Test parity channel operations."""
        
        parity = cqe_system["parity_channels"]
        
        # Test channel extraction
        test_vector = np.array([0.7, 0.3, 0.9, 0.1, 0.5, 0.8, 0.2, 0.6])
        channels = parity.extract_channels(test_vector)
        
        assert len(channels) == 8, "Should extract 8 channels"
        for i in range(8):
            assert f"channel_{i+1}" in channels, f"Missing channel_{i+1}"
        
        # Test parity enforcement
        target_channels = {f"channel_{i+1}": 0.5 for i in range(8)}
        corrected = parity.enforce_parity(test_vector, target_channels)
        
        assert len(corrected) == 8, "Corrected vector should be 8D"
        
        # Test penalty calculation
        penalty = parity.calculate_parity_penalty(test_vector, target_channels)
        assert penalty >= 0, "Penalty should be non-negative"
    
    def test_objective_function_components(self, cqe_system):
        """Test objective function component evaluation."""
        
        obj_func = cqe_system["objective_function"]
        
        test_vector = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        test_channels = {f"channel_{i+1}": 0.5 for i in range(8)}
        domain_context = {"complexity_class": "P", "domain_type": "computational"}
        
        scores = obj_func.evaluate(test_vector, test_channels, domain_context)
        
        # Check all components present
        expected_components = [
            "phi_total", "lattice_quality", "parity_consistency",
            "chamber_stability", "geometric_separation", "domain_coherence"
        ]
        
        for component in expected_components:
            assert component in scores, f"Missing score component: {component}"
            assert 0 <= scores[component] <= 1, f"Score {component} out of range: {scores[component]}"
        
        # Test gradient calculation
        gradient = obj_func.gradient(test_vector, test_channels, domain_context)
        assert len(gradient) == 8, "Gradient should be 8-dimensional"
        
        # Test improvement direction
        direction, reasoning = obj_func.suggest_improvement_direction(
            test_vector, test_channels, domain_context
        )
        assert len(direction) == 8, "Direction should be 8-dimensional"
        assert isinstance(reasoning, dict), "Reasoning should be a dictionary"

class TestCQERunner:
    """Test CQE Runner orchestration."""
    
    @pytest.fixture
    def temp_embedding(self):
        """Create temporary embedding for runner tests."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        save_embedding(temp_path)
        yield temp_path
        
        if Path(temp_path).exists():
            Path(temp_path).unlink()
    
    def test_runner_initialization(self, temp_embedding):
        """Test CQE runner initialization."""
        
        runner = CQERunner(e8_embedding_path=temp_embedding)
        
        # Check all components initialized
        assert runner.domain_adapter is not None
        assert runner.e8_lattice is not None
        assert runner.parity_channels is not None
        assert runner.objective_function is not None
        assert runner.morsr_explorer is not None
        assert runner.chamber_board is not None
    
    def test_problem_solving_pipeline(self, temp_embedding):
        """Test complete problem solving pipeline."""
        
        runner = CQERunner(
            e8_embedding_path=temp_embedding,
            config={"exploration": {"max_iterations": 5}, "output": {"save_results": False}}
        )
        
        # Test P problem
        p_problem = {
            "size": 50,
            "complexity_class": "P",
            "complexity_hint": 1
        }
        
        solution = runner.solve_problem(p_problem, "computational")
        
        # Verify solution structure
        required_fields = [
            "problem", "domain_type", "initial_vector", "optimal_vector",
            "initial_channels", "optimal_channels", "objective_score",
            "analysis", "recommendations", "computation_time", "metadata"
        ]
        
        for field in required_fields:
            assert field in solution, f"Solution missing field: {field}"
        
        assert len(solution["initial_vector"]) == 8
        assert len(solution["optimal_vector"]) == 8
        assert solution["objective_score"] >= 0
        assert isinstance(solution["recommendations"], list)
    
    def test_runner_test_suite(self, temp_embedding):
        """Test runner's internal test suite."""
        
        runner = CQERunner(e8_embedding_path=temp_embedding)
        test_results = runner.run_test_suite()
        
        # Check test structure
        expected_tests = [
            "e8_embedding_load", "domain_adaptation", "parity_extraction",
            "objective_evaluation", "morsr_exploration", "chamber_enumeration"
        ]
        
        for test_name in expected_tests:
            assert test_name in test_results, f"Missing test: {test_name}"
        
        # Most tests should pass
        passed_tests = sum(test_results.values())
        assert passed_tests >= len(expected_tests) * 0.8, "Most tests should pass"
'''

with open("tests/test_cqe_integration.py", 'w') as f:
    f.write(test_cqe_code)

print("Created: tests/test_cqe_integration.py")