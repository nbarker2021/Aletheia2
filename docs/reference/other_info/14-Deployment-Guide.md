# Deployment Guide

## Local
- Run SNAP demos and AGRM sweeps locally to validate configs.
- Persist SNAP store on a fast disk; use periodic ZIP backups.

## Server
- A simple Python service exposing: save/load/query; persona ops; stitch; playbooks; run sweeps.
- Add auth and per-namespace quotas.

## CI
- Include unit + integration tests; fail build on regressions in vector score or room_len_p95.
