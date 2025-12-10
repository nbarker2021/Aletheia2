def decompose_into_roots(weight_vector):
    # Express weight vector as linear combination of roots
    roots = generate_e8_roots()
    
    # Solve linear system: weight_vector = sum(c_i * roots[i])
    root_matrix = np.array(roots).T
    coefficients = np.linalg.lstsq(root_matrix, weight_vector)[0]
    
    # Return non-zero coefficients
    decomposition = {}
    for i, coeff in enumerate(coefficients):
        if abs(coeff) > 1e-10:
            decomposition[roots[i]] = coeff
    
    return decomposition
```

\section{Verification Protocols}

\subsection{Cohomology Class Verification}

\textbf{Class Equality Check}
```python