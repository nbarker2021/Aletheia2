# GENESIS_MINI

Minimal Genesis blockchain node with E8 ledger core

## Modules Included

- **Runtime** (39 lines) - blockchain
- **GeometryTransformerStandaloneV2** (438 lines) - geometry
- **GeoTokenizerTieinV1** (501 lines) - ai
- **ReceiptsBridge** (68 lines) - blockchain
- **StateStore** (51 lines) - geometry
- **Callbacks** (42 lines) - ai
- **AnalyticsCli** (62 lines) - ai
- **ApiServer** (79 lines) - ai
- **CqeGovernance** (123 lines) - blockchain
- **SpeedlightSidecar** (328 lines) - video

## Usage

```python
import sys
sys.path.insert(0, './modules')

# Import the modules you need
from LatticeBuilderV1 import *
from CqeGovernance import *
# etc...
```

## Run

```bash
python genesis_mini.py
```
