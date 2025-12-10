class Policy:
    name: str
    alpha: float = 0.5
    beta: float = 0.1
    gamma: float = 0.3
    delta: float = 0.1
    kappa: float = 0.0
    dihedral_reflection: bool = True
    lattice_candidates: Tuple[int, ...] = (80, 240)
    viewers: Tuple[int, int] = (10, 8)  # decagon, octagon
    max_iter: int = 12

    @staticmethod
    def presets(kind: str) -> "Policy":
        kind = (kind or "channel-collapse").lower()
        if kind == "channel-collapse":
            return Policy("channel-collapse", 0.5, 0.1, 0.3, 0.1, 0.0, True, (80, 240), (10, 8), 12)
        if kind == "knot-sensitive":
            return Policy("knot-sensitive", 0.4, 0.35, 0.15, 0.1, 0.0, True, (80, 240), (10, 8), 12)
        if kind == "numerology-bridge":
            return Policy("numerology-bridge", 0.45, 0.1, 0.35, 0.05, 0.05, True, (80, 240), (10, 8), 12)
        return Policy(kind)

@dc.dataclass