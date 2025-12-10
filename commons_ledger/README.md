# CQE CommonsLedger — Shippable v0.1 (stdlib-only)

**What’s inside**
- `server/` — HTTP server, master ledger, features validator (FEATURES v0.1), mint engine, wallet v2 (vesting/payout wallet), governance (OMPS/E-gate/Ring), CA engines (tick/council/harm), markets, tools registry, bounty board.
- `web/` — single-page UI to exercise endpoints.
- `adapters/` — drop your feature adapters here (see `mock_adapter.py`).
- `data/` — assets and ledgers; `config.json` holds dev HMAC key and coin list.
- `speedlight/` — if present, `speedlight_sidecar_plus.py` is used; else falls back to a simple in-mem sidecar.

**Run**
```bash
python server/server.py
# open http://127.0.0.1:8766/web/
```

**Typical flow**
1. Upload: Ingest JSON `{"name":"doc.txt","content":"fractal lattice note..."}`.
2. Features: Auto-extract → Validate → Mint Score → Mint Now.
3. Wallet: Check balances (MERIT + domains).
4. Tick/Council: Tick → Council Draw/Vote (demo).
5. Markets: Shock/Stabilize/Understanding spend.
6. Verify: Run master verify.

**Notes**
- This is a working prototype designed to be extended by your real adapters (GeoTokenizer, Viewer24, MDHG, etc.).
- All records are immutable JSONL with `prev/hash/sig` and verify via `/master/verify`.
- No external deps; stdlib only.
