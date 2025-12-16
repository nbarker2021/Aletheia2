# P5.3: CQE API & Core Library Package

This package provides a minimal API and core library for CQE operations.

## Artifacts

1. **Core Library**: `lib/cqe_core.py`
2. **API Application**: `api/src/cqe_api.py`
3. **Dockerfile for API**: `api/Dockerfile`
4. **API Test Cases**: `api/tests/test_api.py`

## lib/cqe_core.py
```python
import numpy as np

def toroidal_embed(vec, dims):
    """
    Projects an n-dimensional vector into an n-torus of given dimensions.
    Args:
        vec (array-like): Input vector.
        dims (list): Torus dimensions.
    Returns:
        list: Wrapped coordinates.
    """
    return [v % d for v, d in zip(vec, dims)]

# Example: embedding Riemann zero metric
def embedding_distance(root_coords, embed_coords):
    return np.linalg.norm(np.array(root_coords) - np.array(embed_coords))
```

## api/src/cqe_api.py
```python
from fastapi import FastAPI
from pydantic import BaseModel
from lib.cqe_core import toroidal_embed

app = FastAPI(title="CQE API")

class EmbedRequest(BaseModel):
    vec: list
    dims: list

@app.post("/embed")
def embed(req: EmbedRequest):
    """Returns toroidal embedding of the vector."""
    result = toroidal_embed(req.vec, req.dims)
    return {"embedded": result}
```

## api/Dockerfile
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY api/src /app
COPY lib /app/lib
RUN pip install fastapi uvicorn pydantic numpy
CMD ["uvicorn", "cqe_api:app", "--host", "0.0.0.0", "--port", "8080"]
```

## api/tests/test_api.py
```bash
pip install pytest requests
```
```python
import requests

def test_embed():
    url = "http://localhost:8080/embed"
    payload = {"vec": [1.2, -0.5], "dims": [3,3]}
    r = requests.post(url, json=payload)
    assert r.status_code == 200
    data = r.json()
    assert all(0 <= c < 3 for c in data['embedded'])
```