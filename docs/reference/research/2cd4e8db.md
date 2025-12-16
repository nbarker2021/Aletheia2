# CQE API Reference

## Python Client API

### CQEClient

High-level client for CQE operations.

```python
from cqe import CQEClient

client = CQEClient()
```

#### Methods

##### `embed(content, domain="text", optimize=True)`

Embed content into E8 space.

**Parameters:**
- `content` (str): Content to embed
- `domain` (str): Domain type ("text", "code", "scientific")
- `optimize` (bool): Apply MORSR optimization

**Returns:**
- `CQEOverlay`: Embedded overlay

**Example:**
```python
overlay = client.embed("Machine learning text", optimize=True)
```

##### `get_phi_metrics(overlay)`

Compute Φ metrics for overlay.

**Parameters:**
- `overlay` (CQEOverlay): Overlay to analyze

**Returns:**
- `dict`: Φ components and total

**Example:**
```python
metrics = client.get_phi_metrics(overlay)
print(f"Φ_total: {metrics['phi_total']}")
```

##### `apply_operator(operator_name, overlay)`

Apply transformation operator.

**Parameters:**
- `operator_name` (str): Operator name ("rotation", "midpoint", "parity")
- `overlay` (CQEOverlay): Input overlay

**Returns:**
- `CQEOverlay`: Transformed overlay

**Example:**
```python
transformed = client.apply_operator("midpoint", overlay)
```

##### `find_similar(query_overlay, top_k=10)`

Find similar overlays.

**Parameters:**
- `query_overlay` (CQEOverlay): Query
- `top_k` (int): Number of results

**Returns:**
- `list`: [(overlay, distance), ...]

**Example:**
```python
similar = client.find_similar(overlay, top_k=5)
```

## REST API

Base URL: `http://localhost:8000`

### Endpoints

#### `POST /embed`

Embed content into E8 space.

**Request Body:**
```json
{
  "content": "Text to embed",
  "domain": "text",
  "optimize": true
}
```

**Response:**
```json
{
  "overlay_id": "abc123def456",
  "active_slots": 45,
  "cartan_active": 8,
  "phi_metrics": {
    "phi_total": 52.3,
    "phi_geom": 12.1,
    "phi_parity": 4.0,
    "phi_sparsity": 34.2,
    "phi_kissing": 2.0
  },
  "processing_time_ms": 15.3
}
```

#### `POST /query`

Query for similar overlays.

**Request Body:**
```json
{
  "overlay_id": "abc123",
  "top_k": 10
}
```

**Response:**
```json
{
  "results": [
    {
      "overlay_id": "xyz789",
      "distance": 2.3,
      "active_slots": 42,
      "cartan_active": 7
    }
  ],
  "query_overlay_id": "abc123"
}
```

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-10-08T17:00:00",
  "components": {
    "api": "operational",
    "cqe_client": "initialized",
    "e8_lattice": "ready",
    "morsr": "ready"
  }
}
```

#### `GET /metrics/{overlay_id}`

Get metrics for specific overlay.

**Response:**
```json
{
  "overlay_id": "abc123",
  "metrics": {
    "phi_total": 52.3,
    "phi_geom": 12.1,
    "phi_parity": 4.0,
    "phi_sparsity": 34.2,
    "phi_kissing": 2.0
  },
  "active_slots": 45,
  "cartan_active": 8,
  "provenance": ["R_theta(0.7854)", "Midpoint(palindrome)"]
}
```

## Core Classes

### CQEOverlay

Overlay data structure.

**Attributes:**
- `present` (np.ndarray): 248-bit activation mask
- `w` (np.ndarray): Weights for slots
- `phi` (np.ndarray): Phases for slots
- `pose` (dict): Metadata
- `hash_id` (str): Content-addressed hash
- `provenance` (list): Transformation history

**Properties:**
- `active_slots`: Indices of active slots
- `cartan_active`: Count of active Cartan lanes
- `is_canonical`: Whether canonicalized

### Operators

All operators inherit from `CQEOperator` base class.

**Available Operators:**
- `RotationOperator`: Rθ quantized rotations
- `ReflectionOperator`: WeylReflect
- `MidpointOperator`: Palindromic reduction
- `ParityMirrorOperator`: Cartan mirroring
- `ECCParityOperator`: Parity correction
- `SingleInsertOperator`: Slot insertion

## CLI

### Commands

```bash
# Embed text
cqe embed "Your text here"

# System information
cqe info

# Show version
cqe --version
```

## Configuration

Environment variables:

- `REDIS_URL`: Redis connection URL (optional)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MAX_CACHE_SIZE`: Maximum overlay cache size
