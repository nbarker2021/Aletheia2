# Extracted from: CQE_CORE_MONOLITH_utils_1.py
# Class: SliceObservables
# Lines: 9

class SliceObservables:
    theta: List[float]                         # lattice angles (radians)
    extreme_idx: List[int]                     # i(θ): index of extreme sample (by projection on θ)
    quadrant_bins: List[Tuple[int,int,int,int]]  # q(θ): counts per quadrant-like bin
    chord_hist: List[Dict[int,int]]            # hΔ(θ): histogram of chord steps (constant in this simple model)
    perm: List[List[int]]                      # π(θ): top-k order (indices) by projection
    braid_current: List[int]                   # B(θ): adjacent transposition count per step
    energies: Dict[str, float]                 # Dirichlet energies over chosen signals
