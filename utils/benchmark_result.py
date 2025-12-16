class BenchmarkResult:
    """Single benchmark measurement result."""
    problem_size: int
    runtime_seconds: float
    memory_mb: float
    cache_hit_rate: float
    lattice_operations: int
    objective_evaluations: int
    convergence_iterations: int
    final_objective_value: float
    success: bool

