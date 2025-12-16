# Testing Strategy

## Unit
- AGRM: put/get/remove; promotions/demotions; reinsertion; variant setter; stats quantiles.
- SNAP: validators; save/load/query; stitcher minimal; playbook match.
- Personas: create/compose/update_score roundtrips.

## Integration
- Persona → route → stitched code → run simple workload → error → playbook.

## End-to-End
- Ingest (stub) → persona → AGRM sweeps (few k) → DTT → SNAP provenance.
- Assert: improvements persist; lineage intact; rollbacks possible.
