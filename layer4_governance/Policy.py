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
