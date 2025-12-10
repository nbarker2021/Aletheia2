
# Coherence Suite v1 — Waveform Collapse & Decoherence (CQE-native, stdlib)

A receipts-aware toolkit to measure coherence/decoherence and detect collapse events,
aligned with your geometry-first CQE runtime and the saved embeddings.

## Modules
- coherence_metrics.py — angular/radial coherence, spectral entropy, dPhi proxy, composite score, collapse detector.
- receipts_bridge.py — unify GeoLight (.geolight/ledger.jsonl) and TokLight (.geo_tokenizer/ledger.jsonl) timelines.
- state_store.py — save/recall snapshots (points/tokens/embedding) keyed by receipt.
- callbacks.py — recall-by-receipt, pairwise comparison, timeline metrics.
- analytics_cli.py — CLI for analyses.
- api_server.py — minimal local REST API.

## Run
```bash
python api_server.py   # http://127.0.0.1:8787
```

## Examples
```bash
python analytics_cli.py coherence --points-json examples/hex.json
python analytics_cli.py collapse --prev-json ex/a.json --curr-json ex/b.json
python analytics_cli.py align --emb-a-json ex/ea.json --emb-b-json ex/eb.json
python analytics_cli.py timeline --store ./deco_states --receipts-json ex/receipts.json
```
