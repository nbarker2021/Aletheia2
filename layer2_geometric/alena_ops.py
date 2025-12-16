class ALENAOps:
    """ALENA Operators: Rθ/Weyl/Midpoint/ECC for lattice snaps with 3-6-9 projection channels."""
    def __init__(self):
        self.e8_roots = self._gen_e8_roots()
        self.projection_channels = [3, 6, 9]

    def _gen_e8_roots(self) -> np.ndarray:
        """Generate 240 E8 roots with norm √2."""
        roots = []
        for i in range(8):
            for j in range(i+1, 8):
                for s1, s2 in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                    root = [0]*8
                    root[i], root[j] = s1, s2
                    roots.append(root)
        for signs in product([-0.5, 0.5], repeat=8):
            if sum(1 for s in signs if s < 0) % 2 == 0:
                roots.append(list(signs))
        roots = np.array(roots)
        for i in range(len(roots)):
            roots[i] = roots[i] * (E8_NORM / sp_norm(roots[i]))
        return roots

    @ladder_hook
    def r_theta_snap(self, vector: np.ndarray) -> np.ndarray:
        """Rθ rotation snap to nearest root via 3-6-9 channels."""
        theta = np.arctan2(vector[1], vector[0])
        r = sp_norm(vector[:2])
        channel = random.choice(self.projection_channels)
        snapped = np.array([r * np.cos(theta * channel), r * np.sin(theta * channel)] + [0]*(8-channel))
        return snapped

    @ladder_hook
    def weyl_flip(self, vector: np.ndarray) -> np.ndarray:
        """Weyl reflection flip for parity alignment."""
        return -vector

    @ladder_hook
    def midpoint_ecc(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        """Midpoint ECC snap for error correction."""
        mid = (vector1 + vector2) / 2
        return mid * (E8_NORM / sp_norm(mid)) if sp_norm(mid) > 0 else mid
