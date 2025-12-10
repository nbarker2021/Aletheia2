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
"""
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
