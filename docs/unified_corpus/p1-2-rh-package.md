# P1.2: Riemann Hypothesis Embedding Invariance & Sensitivity Package

This package delivers all code, data, and analysis for Phase 1.2.

## Artifacts

1. **RH Invariance & Sensitivity Analysis PDF**: Full report with tables and methodology
   - File: rh_invariance_sensitivity.pdf [122]
2. **Analysis Code**: Python script performing embedding variants and computing distances
   - File: rh_analysis.py

## rh_analysis.py
```python
import numpy as np
from mpmath import zetazero

# Load first 10k zeros
t_zeros = [zetazero(n).imag for n in range(1, 10001)]
# Load E8 roots (240×8 CSV) from 'e8_roots.csv'
roots = np.loadtxt('e8_roots.csv', delimiter=',')

def embed(t, variant, param):
    vals = []
    for i in range(8):
        if variant=='base':
            val = ((t**2 + i) % (2*np.pi) - 1)/(2*np.pi)
        elif variant=='shift':
            phi = param
            val = (((t+phi)**2 + i) % (2*np.pi) - 1)/(2*np.pi)
        else:
            r = param
            val = ((t**2 + i) % (2*np.pi*r) - 1)/(2*np.pi*r)
        vals.append(val)
    return np.array(vals)

variants = [
    ('base', [None]),
    ('shift', [np.pi/4]),
    ('rand', [1.5, 2.0, 2.5])
]
results = {}
for var, params in variants:
    res = []
    for p in params:
        dists = [np.linalg.norm(roots - embed(t, var, p), axis=1).min()
                 for t in t_zeros]
        res.append((np.mean(dists), np.std(dists)))
    results[var] = res

print(results)
```  

## Instructions

1. Place `e8_roots.csv` in working directory.  
2. Install requirements: `pip install mpmath numpy`  
3. Run: `python rh_analysis.py > rh_results.json`  
4. Open `rh_invariance_sensitivity.pdf` [122] for tables and discussion.
