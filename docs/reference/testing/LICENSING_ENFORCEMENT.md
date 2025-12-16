# License Enforcement (Optional)

You can require a local permission file for API access.

1) Create `/mnt/data/permission.json`:
```json
{"grantee":"ACME Labs","scope":"dev-only","term":"2025-12-31","sig":"<licensor-sig-or-anchor>"}
```

2) (If your server supports it) Enable enforcement in config:
```bash
curl -s localhost:8000/config/set -H 'content-type: application/json' -d '{"config":{"license_required": true}}'
```

3) The API should reject requests lacking a permission file with HTTP 451.
