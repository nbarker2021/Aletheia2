# Morphonic Operation Platform

## Loading into AI Platforms

This package can be loaded into AI platforms like AnythingLLM, LangChain, or custom AI systems.

### Quick Start

```python
from morphonic_platform import MorphonicPlatform

# Initialize
platform = MorphonicPlatform()

# Process requests
result = platform.process({"action": "status"})
```

### Available Actions

| Action | Description | Parameters |
|--------|-------------|------------|
| `status` | Get platform status | None |
| `initialize` | Initialize state | `lanes` (list), `parity` (bool) |
| `apply` | Apply an operator | `operator` (str), `kwargs` (dict) |
| `morsr` | Run MORSR optimization | `budget` (int) |

### Available Operators

- `R_theta` - Coxeter rotation (k=1-7)
- `Weyl_reflect` - Root reflection (idx=0-7)
- `Midpoint` - Palindromic expansion
- `ECC_parity` - Syndrome repair
- `SingleInsert` - Controlled expansion
- `ParityMirror` - Sector involution

### UVIBS/Monster Metrics

```python
metrics = platform.compute_uvibs_metrics()
```

### Directory Structure

```
Aletheia2/
├── morphonic_platform.py    # Main entry point
├── unified/                  # All CQE modules (856 files)
│   ├── cqe_core/            # Core system
│   ├── cqe_unified/         # Unified runtime
│   └── all_cqe/             # Extended modules
└── layer*/                  # MVP layers
```
