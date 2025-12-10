
# Monster/Moonshine VOA Embedding DB — Personal HTML Server

Pure-stdlib personal server for CQE-style embeddings with Moonshine/VOA flavor.

## Run
```bash
python server.py  # http://127.0.0.1:8765
```
Use the UI to add items and search. API: `/api/add`, `/api/search`, `/api/get`, `/api/list`, `/api/stats`.

## Extend
- Edit `embedding/voa_moonshine.py` to inject more McKay–Thompson coefficients (CSV import easy to add).
- Replace `embedding/geometry_bridge.py` with your Scene8/WorldForge descriptors if desired.
- Add more charts (e.g., Niemeier theta) by computing features and storing under a new chart name.
