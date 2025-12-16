# AGRM / MDHG Hash — Deep Design

## Rationale
Workloads are skewed and bursty. We want sustained low tail-latency, with capacity
to adapt automatically. Hierarchical layout + φ-growth + sweep tuning beats static tables.

## Data Model
- **Building → Floor → Room → Entries**
- `_Entry = {key, value, _Meta}`; `_Meta = {hits, last_touch, floor, room, building}`
- `_loc: Dict[key → (b,f,r,idx)]` accelerates lookup & removal.

## Hashing and Placement
- Stable 64-bit hash → room index.
- `_biased_place()` consults `Shortcuts` to bias target room by co-visit next-hop.

## Tiering Logic
- `_touch()` increments `hits`; promotes when `hits ≥ promote_hits`.
- Demotion: periodic sampling in `get()` + explicit `idle_sweep()`.
- Reinsertion:
  - `reinsert_worst_rooms(top_k)` sorts by `(hits,last_touch)` and repacks.
  - `reinsert_clusters(min_count)` uses room co-visit to pack adjacently.

## Resize / Growth
- `target_load` vs current load; φ-scale growth prevents oscillatory rehashes.
- Add buildings (cheap) or adjust per policy; trace every resize.

## Traversal Variants
- `base, reverse, rot3, stride2, stride3, block_serp` per floor.
- Probe once per `rooms_per_floor` and cache winner (empirical > theoretical optimal).

## Observability
- `stats()` returns load, promotions/demotions/moves, timings, shortcuts, room_len p95/max.
- `hit_histogram()` for quantiles (median, p90).

## Adaptive Sweeps
1) Probe traversal → 2) Inserts → 3) Skewed gets → 4) Idle sweep →
5) Reinsertions → 6) Finalize stats → 7) Tune thresholds.
- Tuning strategy:
  - `promote_hits := max(2, p90_hits)`
  - `demote_after_idle := median_hits`
  - `phi_scale := EMA(phi_scale, delta(load))`
- Snapshots: `stats_*.json`, `trace_*.json` (sampled), `best_config.json`, histories.

## Failure Modes & Fixes (Playbooks)
- **stale_index** → rebuild `_loc` map; verify no stale entries.
- **room_hotspots** (room_len_max >> p95) → run both reinsertion passes.
- **resize_storm** → increase promote threshold, reduce φ temporarily.

## Design Tradeoffs
- Slightly higher code complexity vs flat tables in exchange for sustained tail latency and
cheap adaptivity. EMA slows convergence but avoids oscillation.
