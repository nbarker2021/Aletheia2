def extract_features(x):
    txt = str(x).lower()
    boost = 0.2 if ("fractal" in txt or "lattice" in txt) else 0.0
    return {
        "GEO_GLOBAL": {"Novelty": 0.3 + boost},
        "FIVEWH1": {"What":{"clarity":0.7}},
    }
