# Create test runner and final setup files
test_runner_code = '''#!/usr/bin/env python3
"""
Test Runner for CQE-MORSR Framework

Comprehensive test execution with reporting.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run all tests with coverage reporting."""
    print("CQE-MORSR Test Runner")
    print("=" * 30)
    
    # Ensure we're in the right directory
    if not Path("cqe_system").exists():
        print("Error: Run from repository root directory")
        sys.exit(1)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/",
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\\n✓ All tests passed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\\n✗ Tests failed with return code {e.returncode}")
        return False
    
    except FileNotFoundError:
        print("\\nError: pytest not found. Install with: pip install pytest")
        return False

def main():
    success = run_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
'''

with open("scripts/run_tests.py", 'w') as f:
    f.write(test_runner_code)

# Create pytest configuration
pytest_config = '''[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --color=yes
    --durations=10
markers =
    integration: marks tests as integration tests
    slow: marks tests as slow running
    unit: marks tests as unit tests
'''

with open("pytest.ini", 'w') as f:
    f.write(pytest_config)

# Create makefile for convenience
makefile_content = '''# CQE-MORSR Framework Makefile

.PHONY: setup test clean install run-golden

# Setup the system
setup:
	python scripts/setup_embeddings.py

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	python scripts/run_tests.py

# Run golden test harness
run-golden:
	python examples/golden_test_harness.py

# Generate Niemeier lattices (requires SageMath)
generate-niemeier:
	sage sage_scripts/generate_niemeier_lattices.sage

# Clean generated files
clean:
	rm -rf data/generated/*
	rm -rf data/cache/*
	rm -rf logs/*
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# Full setup and test
all: install setup test run-golden

# Help
help:
	@echo "CQE-MORSR Framework Build Commands:"
	@echo "  make setup         - Set up E8 embeddings"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run test suite"
	@echo "  make run-golden    - Run golden test harness"
	@echo "  make generate-niemeier - Generate Niemeier lattices (requires SageMath)"
	@echo "  make clean         - Clean generated files"
	@echo "  make all           - Full setup and test"
	@echo "  make help          - Show this help"
'''

with open("Makefile", 'w') as f:
    f.write(makefile_content)

print("Created: scripts/run_tests.py")
print("Created: pytest.ini")
print("Created: Makefile")