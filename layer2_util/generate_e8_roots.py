def generate_e8_roots() -> List[List[float]]:
    """Generate the 240 E₈ root vectors (8-dimensional)."""
    roots = []
    
    # Type I: ±e_i ± e_j (112 roots)
    for i in range(8):
        for j in range(i+1, 8):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    v = [0.0] * 8
                    v[i], v[j] = float(s1), float(s2)
                    roots.append(v)
    
    # Type II: (±½,±½,±½,±½,±½,±½,±½,±½) with even number of minus signs (128 roots)
    for mask in range(1 << 8):
        v = [(-1.0)**((mask >> k) & 1) * 0.5 for k in range(8)]
        if v.count(-0.5) % 2 == 0:
            roots.append(v)
            if len(roots) == 240:
                break
    
    return roots
