# SNAP System Ops Management Notes

## Foreseeable Needs
- Persona skill decay monitoring; retraining schedule
- Automated ingestion jobs (cron or triggered by task)
- Backup/restore of SNAP store (versioning, snapshots)
- Integration tests across modules before deployment
- Embedding index for semantic search
- ACL system for multi-operator environments

## Recommendations
- Keep ops_docs updated with changes in schema or behavior
- Log all major actions (ingestion, composition, routing, stitching)
- Periodically audit SNAP store for stale/duplicate entries
