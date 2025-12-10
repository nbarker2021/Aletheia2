class EnhancedMORSRExplorer:
    """Enhanced MORSR Explorer with dynamic pulse adjustments for lattice optimization."""
    def __init__(self):
        self.radius = MORSR_RADIUS
        self.dwell = MORSR_DWELL
        self.best_score = 0.0

    @ladder_hook
    def explore(self, vector: np.ndarray) -> Tuple[np.ndarray, float]:
        """Explore lattice with MORSR pulses, adjust radius for best score."""
        best_vector = vector.copy()
        for radius in range(5, 10):
            pulsed = vector.copy()
            for _ in range(self.dwell):
                for i in range(len(pulsed)):
                    if i % 2 == 0:
                        pulsed[i] *= radius
                    else:
                        pulsed[i] = -pulsed[i]
            score = sp_norm(pulsed) / sp_norm(vector) if sp_norm(vector) > 0 else 1.0
            if score > self.best_score:
                self.best_score = score
                best_vector = pulsed
        return best_vector, self.best_score

    def morsr_pulse(self, vector: np.ndarray) -> np.ndarray:
        """Apply MORSR pulses for ΔΦ≤0 snap with dynamic adjustment."""
        for _ in range(self.dwell):
            for i in range(len(vector)):
                if i % 2 == 0:
                    vector[i] = vector[i] * self.radius
                else:
                    vector[i] = -vector[i]
        return vector
