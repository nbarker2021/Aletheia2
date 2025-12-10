class TQFEncoder:
    """TQF quaternary encoding and governance system."""
    
    def __init__(self, config: TQFConfig):
        self.config = config
        self.gray_code_map = {1: 0b00, 2: 0b01, 3: 0b11, 4: 0b10}
        self.reverse_gray_map = {v: k for k, v in self.gray_code_map.items()}
    
    def encode_quaternary(self, vector: np.ndarray) -> np.ndarray:
        """Encode vector using 2-bit Gray code for quaternary atoms."""
        # Normalize to quaternary range [1,4]
        normalized = np.clip(vector * 3 + 1, 1, 4).astype(int)
        
        # Apply Gray code encoding
        encoded = np.zeros(len(normalized) * 2, dtype=int)
        for i, val in enumerate(normalized):
            gray_bits = self.gray_code_map[val]
            encoded[2*i] = (gray_bits >> 1) & 1
            encoded[2*i + 1] = gray_bits & 1
        
        return encoded
    
    def decode_quaternary(self, encoded: np.ndarray) -> np.ndarray:
        """Decode Gray-encoded quaternary back to vector."""
        if len(encoded) % 2 != 0:
            raise ValueError("Encoded vector must have even length")
        
        decoded = np.zeros(len(encoded) // 2)
        for i in range(0, len(encoded), 2):
            gray_bits = (encoded[i] << 1) | encoded[i + 1]
            quaternary_val = self.reverse_gray_map[gray_bits]
            decoded[i // 2] = (quaternary_val - 1) / 3.0
        
        return decoded
    
    def orbit4_closure(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply Orbit4 symmetries: Identity, Mirror, Dual, Mirror∘Dual."""
        return {
            "I": q.copy(),
            "M": q[::-1].copy(),  # Mirror (reverse)
            "D": 5 - q,  # Dual (quaternary complement)
            "MD": (5 - q)[::-1]  # Mirror∘Dual
        }
    
    def check_alt_lawful(self, q: np.ndarray) -> bool:
        """Check ALT (alternating parity) and lawful conditions."""
        # ALT: alternating parity along coordinates
        alt_sum = sum(q[i] * ((-1) ** i) for i in range(len(q)))
        alt_condition = (alt_sum % 2) == 0
        
        # W4: linear plane mod 4
        w4_condition = (np.sum(q) % 4) == 0
        
        # Q8: quadratic mod 8 (simplified)
        q8_condition = (np.sum(q * q) % 8) == 0
        
        return alt_condition and (w4_condition or q8_condition)
    
    def cltmp_projection(self, q: np.ndarray) -> Tuple[np.ndarray, float]:
        """Find nearest lawful element under Lee distance."""
        best_q = q.copy()
        best_distance = float('inf')
        
        # Search in local neighborhood for lawful element
        for delta in range(-2, 3):
            for i in range(len(q)):
                candidate = q.copy()
                candidate[i] = np.clip(candidate[i] + delta, 1, 4)
                
                if self.check_alt_lawful(candidate):
                    # Lee distance (Hamming distance in Gray code)
                    distance = np.sum(np.abs(candidate - q))
                    if distance < best_distance:
                        best_distance = distance
                        best_q = candidate
        
        return best_q, best_distance
    
    def compute_e_scalars(self, q: np.ndarray, orbit: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute E2/E4/E6/E8 scalar metrics."""
        # E2: Atom Legality
        lawful_count = sum(1 for variant in orbit.values() if self.check_alt_lawful(variant))
        e2 = lawful_count / len(orbit)
        
        # E4: Join Quality (simplified)
        _, cltmp_distance = self.cltmp_projection(q)
        e4 = max(0, 1 - cltmp_distance / 4)
        
        # E6: Session Health (placeholder)
        e6 = (e2 + e4) / 2
        
        # E8: Boundary Uncertainty
        uncertainty = np.std(list(orbit.values())) / 4  # Normalized
        e8 = max(0, 1 - uncertainty)
        
        return {"E2": e2, "E4": e4, "E6": e6, "E8": e8}
