# Phys.org: ML-improved XC functionals for DFT → CQE mapping
UTC: 2025-09-20T14:41:10.885866Z

**Source**: Phys.org (University of Michigan): "New approach improves accuracy of quantum chemistry simulations using machine learning", Sept 20, 2025.

**Core result**: Learn a semi-local XC functional by inverting many-body (MB) results on light atoms/molecules; achieve ~third-rung accuracy with second-rung complexity. (Science Advances DOI: 10.1126/sciadv.ady8962)

**CQE octet (proposal)**:
1. atoms: Li,C,N,O,Ne
2. diatomics: H₂, LiH
3. XC rung2 (GGA) baseline
4. inferred XC approaching rung3 accuracy
5. basis-size sweep
6. MB→DFT inversion
7. DFT→MB forward check
8. transfer-to-solids proxy

**Mirror tests**: MB→DFT vs DFT→MB; train↔eval swap; palindromic residual bands.

**Δ-lifts**: local gradient correction; range-separated tweak; density smoothing.

**Strict ratchet**: MAE_eV gates 0.08→0.05→0.03; SC (spectral correlation) 0.95→0.97→0.99.

**Receipt (demo)**: MAE 0.048 eV, MSE 0.004 eV², SC 0.982, rung=2→3; fourbit=1011 (PROVISIONAL).
