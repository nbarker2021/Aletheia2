# Architecture Overview

## High-Level Layers
1. **Execution substrate (AGRM/MDHG)** — In-memory, hierarchical hash with promotion/
   demotion tiers, φ-scaled growth, locality shaping, and adaptive sweeps.
2. **Research engine (Superpermutation)** — Parallel “Bouncing Batch” search with
   data-driven guidance (winners/losers, laminates, De Bruijn, completion).
3. **Knowledge/state (SNAP)** — Typed, persistent states: personas (SNAPDNA), code
   units (SNAPBIT), paper digests, error/playbook ops, AnyFile capture.
4. **Governance & routing** — DTT gates, ThinkTank MoE for persona routing.

## Key Data Flows
- **AGRM Sweep**: run → collect stats/timings → vector score → DTT gate → SNAP snapshot.
- **Persona Learning**: ingest sources → distill to PaperDigest/AlgorithmCard → compose
  SNAPDNA → score via outcomes → route via MoE.
- **Ops Loop**: failure → ErrorTrace → PlaybookStep → fix & verify → FunctionLifecycle update.
- **Code Reuse**: author minimal CodeUnits (SNAPBITs) → stitch to complete blocks.
