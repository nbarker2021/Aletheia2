# API Documentation: CQE Production System

**Version**: 1.0.0

This document provides a high-level overview of the programmatic API for the CQE system.

## Core Principle

The API is designed around the concept of **geometric orchestration**. Instead of calling individual functions, you define a problem and let the system orchestrate the optimal solution path using the 8-fold methodology.

## Main Entry Point: `Aletheia.reason()`

The primary way to interact with the system is through the Aletheia reasoning engine.

```python
from aletheia import Aletheia

# Initialize the engine
ai = Aletheia()

# Define a problem
problem_definition = {
    "domain": "data_structure_design",
    "constraints": {
        "users": 1_000_000,
        "latency_ms": 10
    },
    "goal": "Design optimal data structure for real-time recommendations"
}

# Let the engine find the solution
solution = ai.reason(problem_definition)

# The solution object contains the full 8-fold analysis
print(solution.summary)
print(solution.architecture)
print(solution.governance_report)
```

## Direct Tool Access

While not the primary workflow, individual tools can be accessed directly for specific tasks.

### SpeedLight

```python
from tools import speedlight

# Check if a computation has an equivalent result
computation_hash = speedlight.get_hash(my_params)
if speedlight.has_receipt(computation_hash):
    result = speedlight.get_result(computation_hash)
else:
    result = my_computation()
    speedlight.save_receipt(computation_hash, result)
```

### Lattice Viewer

```python
from tools import lattice_viewer

# View a problem from all 24 Niemeier perspectives
perspectives = lattice_viewer.simulate_all_perspectives(problem_definition)

# perspectives is a list of 24 simulated solutions
```

## MonsterMoonshineDB API

The embedding database can be queried directly.

```python
from aletheia import embeddings

# Find concepts related to 'E8'
related_concepts = embeddings.find_nearest('concept_e8', k=5)

# Get the embedding vector for a concept
vector = embeddings.get_vector('concept_speedlight')
```

## Full API Reference

For detailed function signatures and parameters, please refer to the docstrings within the source code. The system is designed to be self-documenting.
