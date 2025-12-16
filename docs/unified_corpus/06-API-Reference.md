# API Reference (AGRM + Superperm + SNAP)

## AGRM (high-level)
- `AGRMConfig(...)`
- `MDHGHashTable(cfg)` → `.put(k,v)`, `.get(k,default=None)`, `.remove(k)`
- `.set_hamiltonian_variant(name)`
- `.idle_sweep()`, `.reinsert_worst_rooms(k)`, `.reinsert_clusters(min_count)`
- `.stats()`, `.hit_histogram()`, `.snapshot(path)`, `.save_state(path)`, `.load_state(path)`
- Controller:
  - `AGRMController.run_sweeps(sweeps, n, seed)`
  - `AGRMController.snapshot_pack(label)`

## Superpermutation (selected)
- `generate_hypothetical_superpermutation(...)`
- `complete_from_partial(...)`
- `identify_anti_prodigals(...)` / `create_anti_laminate(...)` / `merge_laminates(...)`
- `build_de_bruijn_graph(...)`, `find_eulerian_path(...)`, `add_weights_to_debruijn(...)`

## SNAP
- Core:
  - `SNAPCore.register_kind(kind, validator)`
  - `SNAPCore.save(key, obj)`, `SNAPCore.load(key)`, `SNAPCore.query(kind=None, prefix=None)`
- Types:
  - `register_basic_types(snap)` → ArXivPaper, PaperDigest, LLMProfile, AlgorithmCard, CodeUnit, ErrorTrace, PlaybookStep, AnyFile
- Personas:
  - `register_snapdna(snap)`
  - `persona_create(snap, domain, level, sources, weights, notes)`
  - `persona_update_score(snap, id, delta, ema=0.6)`
  - `persona_compose(snap, hybrid_id, components)`
- Stitcher:
  - `stitch(snap, goal_signature, constraints={})`
- Playbooks:
  - `apply_playbook(snap, error_key)`
- MoE routing:
  - `route(snap, task={{tags[], tools[]}}, k=2)`
