
# Phase 1 Patch Notes & Checklist

## Schemas
- [x] Add E8-AS to all new SNAPs (require `e8` and `axes`).
- [x] Enforce `snap_manifest_v2.schema.json` on write.
- [ ] Migrate legacy manifests via `tools/migrate_snap_v1_to_v2.py`.

## Version canonicality
- [ ] Lift E8 geometry files into v14 container; record `axes.version_container = "v14"`.
- [ ] Deprecate older geometry imports; centralize under `lattice_ai.geometry.e8`.

## MDHG hybrid policy
- [x] Introduce `tools/mdhg_hybrid_policy.py` and wire decisions at persistence boundaries.

## Security
- [ ] Default `security.allow_pickle = false`; reject unsafe restores unless explicit override.
- [ ] Optional signing: add signer/verifier CLI and key registry.

## Validation
- [x] `e8_snap_schemas/validate.py` (jsonschema) for CI step.
- [ ] Add CI: fail on invalid manifests; publish counts to /metrics.

## Telemetry
- [ ] Add counters: `snap_v2_write_total`, `snap_v2_validate_fail_total`.
- [ ] Track geodesic-cache hit rate (Phase 2).

## Docs
- [ ] Developer README: how to emit E8-AS and SNAP v2; migration guide.
