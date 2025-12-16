# CQE-MORSR Usage Guide

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up system
python scripts/setup_embeddings.py

# Run tests
python -m pytest tests/

# Execute golden test harness
python examples/golden_test_harness.py
```

## Basic Usage

### Solving P vs NP Problems

```python
from cqe_system import CQERunner

# Initialize CQE system
runner = CQERunner()

# Define P problem
p_problem = {
    "size": 100,
    "complexity_class": "P", 
    "complexity_hint": 1
}

# Solve using CQE
solution = runner.solve_problem(p_problem, "computational")
print(f"Objective score: {solution['objective_score']}")
print(f"Recommendations: {solution['recommendations']}")
```

### Optimization Problems

```python
# Define optimization problem
opt_problem = {
    "variables": 20,
    "constraints": 10,
    "objective_type": "quadratic"
}

# Solve
solution = runner.solve_problem(opt_problem, "optimization")
```

### Creative Scene Generation

```python
# Define creative problem
creative_problem = {
    "scene_complexity": 75,
    "narrative_depth": 30,
    "character_count": 4
}

# Solve
solution = runner.solve_problem(creative_problem, "creative")
```

## Advanced Usage

### Custom Domain Adaptation

```python
from cqe_system import DomainAdapter

adapter = DomainAdapter()

# Custom feature extraction
custom_features = adapter.hash_to_features("custom problem description")
```

### Direct MORSR Exploration

```python
from cqe_system import MORSRExplorer, CQEObjectiveFunction
import numpy as np

# Initialize components
obj_func = CQEObjectiveFunction(e8_lattice, parity_channels)
morsr = MORSRExplorer(obj_func, parity_channels)

# Direct exploration
initial_vector = np.random.randn(8) 
reference_channels = parity_channels.extract_channels(initial_vector)

optimal_vector, optimal_channels, best_score = morsr.explore(
    initial_vector, reference_channels, max_iterations=100
)
```

### Chamber Board Enumeration

```python
from cqe_system import ChamberBoard

board = ChamberBoard()

# Generate all gate configurations
gates = board.enumerate_gates()
print(f"Generated {len(gates)} gates")

# Create gate sequences
sequence = board.explore_gate_sequence(gates[:10], 20)
```

## Configuration

### CQE Runner Configuration

```python
config = {
    "exploration": {
        "max_iterations": 50,
        "convergence_threshold": 1e-4,
        "pulse_count": 10
    },
    "output": {
        "save_results": True,
        "results_dir": "data/generated", 
        "verbose": True
    },
    "validation": {
        "run_tests": True,
        "comparison_baseline": True
    }
}

runner = CQERunner(config=config)
```

### MORSR Parameters

```python
morsr.set_parameters(
    pulse_size=0.05,           # Smaller for fine-grained exploration
    repair_threshold=0.02,     # Stricter parity enforcement
    exploration_decay=0.98,    # Slower decay for longer exploration
    parity_enforcement_strength=0.9  # Stronger parity constraints
)
```

## Output Interpretation

### Solution Structure

```python
{
    "problem": {...},                    # Original problem description
    "domain_type": "computational",      # Problem domain
    "initial_vector": [...],             # 8D starting configuration
    "optimal_vector": [...],             # 8D optimized configuration
    "initial_channels": {...},           # Initial parity channels
    "optimal_channels": {...},           # Optimized parity channels
    "objective_score": 0.847,            # Final Φ score
    "analysis": {
        "embedding_quality": {...},      # E₈ embedding metrics
        "objective_breakdown": {...},    # Component scores
        "chamber_analysis": {...},       # Weyl chamber information
        "geometric_metrics": {...}       # Distance and convergence metrics
    },
    "recommendations": [...],            # Actionable improvements
    "computation_time": 2.341,           # Execution time in seconds
    "metadata": {...}                    # System metadata
}
```

### Score Interpretation

- **0.9 - 1.0**: Excellent embedding and optimization
- **0.7 - 0.9**: Good quality with minor improvements possible
- **0.5 - 0.7**: Acceptable quality, some refinement recommended
- **0.3 - 0.5**: Fair quality, significant improvements needed
- **0.0 - 0.3**: Poor quality, problem representation or parameters need adjustment

## Troubleshooting

### Common Issues

1. **ImportError on CQE modules**: Ensure you're running from repository root
2. **E₈ embedding not found**: Run `python scripts/setup_embeddings.py`
3. **Poor convergence**: Increase `max_iterations` or adjust `pulse_size`
4. **Low objective scores**: Check problem representation and domain type
5. **Parity violations**: Reduce `repair_threshold` or increase enforcement strength
