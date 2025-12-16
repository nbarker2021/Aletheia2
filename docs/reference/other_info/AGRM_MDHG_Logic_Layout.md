
# AGRM + MDHG Logic Layout Overview

## 🔹 1. CLI Entrypoint
- `agrm_cli_launcher.py` → `main()`
- `agrm_cli_allrun.py` → `run_agrm()`, `run_all_tests()`

## 🔹 2. TSP Data Load
- `tsp_node_loader.py` → `auto_load_tsp()`

## 🔹 3. Runtime Orchestrator
- `agrm_runtime_controller.py` → `run_cycle()`
- Uses:
  - `AGRMFeedbackBus`
  - `AGRMComplexityWeightedModulator`
  - `AGRMRecursiveZoneCollapse`

## 🔹 4. Core Engine
- `agrm_core_loop.py` → `evaluate_path()`, `run()`
- `agrm_core.py` → `detect_shell_failure()`, `should_trigger_reset()`

## 🔹 5. Sweep & Builder
- `sweep_scanner.py`, `navigator_and_builder.py` → spiral sweep
- `agrm_dynamic_midpoint.py` → midpoint detection
- `agrm_zone_collapse.py` → zone simplification

## 🔹 6. Validator
- `salesman_and_evaluator.py` → `scan_path()`, `get_total_distance()`
- Uses `AGRMEvaluator`

## 🔹 7. Result Output
- `agrm_results_export.py` → `export_results_to_csv/json`, `plot_comparison_graph()`

## 🔁 State Layers
- `agrm_complexity_modulator.py` → `compute_weights()`
- `agrm_feedback_bus.py` → `log_failure()`, `get_memory_map()`
- `agrm_quadrant_legality.py`, `agrm_distance_cap.py`, `agrm_zone_density.py`
