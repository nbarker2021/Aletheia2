# SNAP System Operational Schema (v0)

## Modules
- SNAPCore: storage, validation, retrieval
- SNAP Types: validators for state kinds
- SNAPDNA: persona registry, composition, scoring
- SNAPBIT Stitcher: combine CodeUnits
- Playbook Executor: match ErrorTraces to Playbooks
- MoE Router: match tasks to personas

## Data Flows
1. Ingest source (manual/web_ingest)
2. Create relevant SNAPTYPE(s)
3. Store in SNAPCore (hash, metadata)
4. Optionally: update persona, stitch code, add playbook
5. Route tasks to persona(s) via MoE
6. Apply results; store outputs + provenance in SNAPCore

## Lifecycles
- Artifact lifecycle: created → validated → linked → archived
- Persona lifecycle: seeded → trained → composed → routed → scored
- CodeUnit lifecycle: authored → tested → stitched → executed → iterated

## Integrations
- AGRM sweeps: register sweep results as SNAPTYPE:AGRMSweep
- Hash table: cache retrieval of related SNAP items
- Governance layer: DTT/AssemblyLine gates SNAP outputs
