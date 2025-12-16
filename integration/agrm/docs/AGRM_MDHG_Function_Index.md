# AGRM + MDHG Function and Class Index

## `AGRM_FULL_SYSTEM_FINAL_LIVE/sweep_scanner.py`
### Classes:
- SweepScanner

### Functions:
- __init__
- _compute_center
- _spiral_score
- scan
- get_ranked_nodes

## `AGRM_FULL_SYSTEM_FINAL_LIVE/mdhg_hash (1).py`
### Classes:
- MDHGHashTable

### Functions:
- __init__
- _init_buildings
- put
- get
- remove
- _hash_to_building
- _hash_to_velocity_index
- get_stats

## `AGRM_FULL_SYSTEM_FINAL_LIVE/meta_evaluator.py`
### Classes:
- MetaEvaluator

### Functions:
- __init__
- _compute_center
- _assign_quadrants
- _classify_density
- get_zone
- get_quadrant
- get_hemisphere
- summary

## `AGRM_FULL_SYSTEM_FINAL_LIVE/Hash_testsuite.py`
### Classes:
- HashTableBenchmark

### Functions:
- run_benchmarks
- __init__
- run_standard_tests
- _test_standard_distribution
- _test_skewed_distribution
- _test_collision_resistance
- _test_sequential_access
- _test_load_factor_scaling
- _measure_operations
- plot_results
- _plot_operation_times
- _plot_load_factor_scaling
- print_summary
- _print_test_summary

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_path_engine.py`
### Classes:
- AGRMPathEngine

### Functions:
- __init__
- _precompute_distances
- _classify_density
- _get_candidates
- _recursive_path
- _validate_and_optimize
- solve

## `AGRM_FULL_SYSTEM_FINAL_LIVE/AGRM.py`
### Classes:
- MDHGHashTable
- AGRMStateBus
- NavigatorGR
- AGRMEdgeValidator
- ModulationController
- PathBuilder
- SalesmanValidator
- PathAuditAgent
- AGRMController

### Functions:
- __init__
- _initialize_buildings
- _calculate_dimension_sizes
- _initialize_structure
- _initialize_building_paths
- _generate_corner_points
- _initialize_building_shortcuts
- _create_building_shortcut
- _compute_hamiltonian_path
- _hash
- _secondary_hash
- _murmur_hash
- _fnv_hash
- _hash_to_building
- _hash_to_velocity_index
- _hash_to_coords
- _hash_to_conflict_key
- put
- get
- _update_stats_and_patterns
- _search_building
- _search_path
- _follow_hamiltonian_path_for_put
- remove
- _remove_from_building
- _find_nearest_path_key
- _find_path_start_index
- _update_access_patterns
- _consider_velocity_promotion
- _relocate_from_velocity
- _remove_from_current_location
- _check_optimization_and_resize
- _perform_minor_optimization
- _perform_major_optimization
- _update_shortcuts
- _identify_and_relocate_key_clusters
- _prune_path_cache
- _resize
- __init__
- get_cache
- migrate_data
- update_sweep_data
- get_node_sweep_data
- is_visited
- add_visited
- get_unvisited_nodes
- update_modulation_params
- update_builder_state
- check_convergence
- merge_paths
- add_salesman_proposal
- get_salesman_proposals
- clear_salesman_proposals
- store_accepted_patch
- get_accepted_patches
- clear_accepted_patches
- splice_patch
- __init__
- _calculate_center
- _calculate_radii_and_angles
- _assign_shells_and_sectors
- _calculate_gr_sweep_scores
- _assign_quadrants_and_hemispheres
- _classify_density
- run_sweep
- __init__
- is_edge_legal
- _check_shell_proximity
- _check_sector_continuity
- _check_phase_timing
- _check_curvature
- _check_distance_cap
- __init__
- assess_complexity
- select_cache
- trigger_migration_check
- update_controller_state
- get_current_legality_params
- process_salesman_feedback
- __init__
- step
- _find_k_nearest_unvisited
- _find_reentry_node
- splice_patch_if_instructed
- __init__
- run_validation_and_patching
- __init__
- run_audit
- _calculate_global_metrics
- _run_simple_nn_baseline
- _analyze_patterns
- _generate_recommendations
- __init__
- solve
- calculate_total_path_cost

## `AGRM_FULL_SYSTEM_FINAL_LIVE/tsp_node_loader.py`
### Classes:
- TSPNodeLoader

### Functions:
- auto_load_tsp
- select_tsp_dataset
- __init__
- parse_file
- get_nodes
- summary

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_profiler_diagnostics.py`
### Classes:
- AGRMProfiler
- AGRMDiagnosticController

### Functions:
- __init__
- mark
- start_timer
- stop_timer
- report
- print_report
- __init__
- set_param
- get_config
- print_config

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_results_export.py`

### Functions:
- export_results_to_csv
- export_results_to_json
- plot_comparison_graph

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_runtime_controller.py`
### Classes:
- AGRMRuntimeController

### Functions:
- __init__
- _evaluate_path
- run_cycle
- get_diagnostics
- get_best_path
- get_best_cost

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_cli_launcher.py`

### Functions:
- main

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_cli_allrun.py`

### Functions:
- run_agrm
- run_all_tests
- main

## `AGRM_FULL_SYSTEM_FINAL_LIVE/navigator_and_builder.py`
### Classes:
- NavigatorGR
- BuilderAgent

### Functions:
- __init__
- run_sweep
- _spiral_sort_key
- get_sweep_path
- trigger_reseed
- __init__
- build_path
- _smart_shuffle
- trigger_reroute
- get_last_path

## `AGRM_FULL_SYSTEM_FINAL_LIVE/salesman_and_evaluator.py`
### Classes:
- Salesman
- AGRMEvaluator

### Functions:
- __init__
- scan_path
- _compute_total_distance
- get_total_distance
- collect_feedback
- __init__
- ingest_feedback
- apply_modulation_if_needed
- reset_required

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_core.py`
### Classes:
- AGRMCoreModulator

### Functions:
- __init__
- compute_entropy_slope
- detect_shell_failure
- detect_entropy_floor
- should_trigger_reset

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_modulation_and_legalgraph.py`
### Classes:
- AGRMLegalGraph
- AGRMModulationController

### Functions:
- __init__
- invalidate_edge
- is_legal
- neighbors
- __init__
- analyze_entropy
- _angle_between
- unlock_shell
- should_unlock
- trigger_global_unlock
- mark_shell_failure
- v
- dot
- mag

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_pathbuilder_dual.py`
### Classes:
- AGRMPathBuilderDual

### Functions:
- invalidate_legality
- __init__
- build_path
- _distance
- get_last_path

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_spiral_reentry.py`
### Classes:
- AGRMSpiralReentryEngine

### Functions:
- __init__
- _compute_center
- generate_fallback_path
- _distance

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_core_loop.py`
### Classes:
- AGRMCoreLoopController

### Functions:
- __init__
- evaluate_path
- _dist
- run

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_dynamic_midpoint.py`
### Classes:
- AGRMDynamicMidpointDetector

### Functions:
- __init__
- _compute_center
- detect
- _distance

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_spiral_retention.py`
### Classes:
- AGRMTopXRetainer

### Functions:
- __init__
- _compute_center
- retain_top_x
- _distance

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_zone_collapse.py`
### Classes:
- AGRMRecursiveZoneCollapse

### Functions:
- __init__
- _compute_center
- collapse_shells
- _group_by_shell
- _distance

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_distance_cap.py`
### Classes:
- AGRMDistanceCapByShell

### Functions:
- __init__
- _compute_center
- _compute_shell_thresholds
- shell_index
- is_within_cap
- _dist

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_quadrant_legality.py`
### Classes:
- AGRMQuadrantLegality

### Functions:
- __init__
- _compute_center
- _quadrant
- is_crossing_illegal

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_zone_density.py`
### Classes:
- AGRMZoneDensityClassifier

### Functions:
- __init__
- _compute_center
- _distance
- _classify_nodes
- shell_density
- shell_members

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_phase_reverse_builder.py`
### Classes:
- AGRMPhaseAwareReverseBuilder

### Functions:
- __init__
- _compute_center
- _map_quadrants
- build_reverse
- _dist

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_anchor_reset.py`
### Classes:
- AGRMAnchorReselector

### Functions:
- __init__
- _compute_center
- find_new_anchor

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_complexity_modulator.py`
### Classes:
- AGRMComplexityWeightedModulator

### Functions:
- __init__
- compute_weights
- _angle_between
- _normalize
- get_high_complexity_nodes
- print_summary
- vec
- dot
- mag

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_diagnostics.py`
⚠️ Error: unexpected character after line continuation character (agrm_diagnostics.py, line 58)

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_modulation_brain.py`
⚠️ Error: unexpected character after line continuation character (agrm_modulation_brain.py, line 75)

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_feedback_bus.py`
### Classes:
- AGRMFeedbackBus

### Functions:
- __init__
- broadcast
- collect_all
- log_failure
- get_memory_map
- get_most_failed_nodes
- print_summary

## `AGRM_FULL_SYSTEM_FINAL_LIVE/agrm_adaptive_builder.py`
### Classes:
- AGRMAdaptiveBuilder

### Functions:
- __init__
- build_path
- get_last_path
- _dist

## `cmplx-core (2)/cmplx-core/cli/gpt-cmplx.py`

### Functions:
- load_config
- estimate_tokens
- chunk_document
- reflect_chunk
- save_reflection
- run_ingestion

## `cmplx-core (2)/cmplx-core/cli/schema_version_manager.py`

### Functions:
- load_schema
- save_schema
- bump_version
- update_schema

