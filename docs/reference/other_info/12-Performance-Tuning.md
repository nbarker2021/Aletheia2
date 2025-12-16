# Performance Tuning

- **AGRM**
  - Raise `promote_hits` if promotions too chatty; lower if hot keys stagnate.
  - Use `reinsert_*` after heavy insert phases.
  - Watch `room_len_p95`; if drifting, run cluster reinsertion or tweak growth.
  - Probe traversal variants whenever `rooms_per_floor` changes.
- **SNAP**
  - Batch writes; dedupe `AnyFile` by sha.
  - Keep personas coarse-grained; specialize via hybrids.
