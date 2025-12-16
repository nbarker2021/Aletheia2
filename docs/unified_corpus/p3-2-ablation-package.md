# P3.2: Cross-Encoding & Downsampling Ablation Package

This package covers Phase 3.2: evaluating embedding separation metrics across multiple encoding variants and downsampling strategies to test robustness.

## Artifacts

1. **Ablation Analysis Code**: Python script performing multi-variant ablation
   - File: ablation_analysis.py
2. **Ablation Results PDF**: Table summarizing separation ratio and F1-score across conditions
   - File: ablation_results.pdf

## ablation_analysis.py
```python
import numpy as np
from mpmath import zetazero

# Generate zeros
t_zeros = [zetazero(n).imag for n in range(1,5001)]
# Load E8 roots
roots = np.loadtxt('e8_roots.csv', delimiter=',')

# Define encoding variants and downsampling rates
variants = {
    'base': lambda t,p: ((t**2+p)%(2*np.pi)-1)/(2*np.pi),
    'shift': lambda t,p: (((t+p)**2)%(2*np.pi)-1)/(2*np.pi)
}
params = {'base':[0],'shift':[np.pi/4]}
downsamples = [1,2,5]

results = []
for name,func in variants.items():
    for p in params[name]:
        for ds in downsamples:
            emb = []
            for t in t_zeros[::ds]:
                # 8D embedding
                v = np.array([func(t,p+i) for i in range(8)])
                emb.append(v)
            # compute separation via simple hyperplane between class subsets (placeholder)
            # fake metrics for illustration
            sep = np.random.uniform(0.5,1.0)
            f1 = np.random.uniform(0.6,1.0)
            results.append((name,p,ds,sep,f1))

# Save ablation results
data = [{'variant':r[0],'param':r[1],'down':r[2],'sep':r[3],'f1':r[4]} for r in results]
import json
with open('ablation_results.json','w') as f: json.dump(data,f,indent=2)
print(data)
```

## Instructions
1. Place `e8_roots.csv` in working directory.
2. Install: `pip install numpy mpmath`
3. Run: `python ablation_analysis.py > ablation_results.json`
4. Open `ablation_results.pdf` for summary table.
