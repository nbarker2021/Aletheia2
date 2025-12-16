class E8GeometryValidator:
    """E8 geometric consistency validation utilities"""
    
    def __init__(self):
        self.e8_roots = self._generate_e8_roots()
        self.logger = logging.getLogger("E8GeometryValidator")
        
    def _generate_e8_roots(self) -> np.ndarray:
        """Generate complete E8 root system"""
        roots = []
        
        # Type 1: ±e_i ± e_j (i < j) - 112 roots
        for i in range(8):
            for j in range(i+1, 8):
                for sign1 in [-1, 1]:
                    for sign2 in [-1, 1]:
                        root = np.zeros(8)
                        root[i] = sign1
                        root[j] = sign2
                        roots.append(root)
        
        # Type 2: (±1,±1,±1,±1,±1,±1,±1,±1)/2 with even # of minus signs - 128 roots
        for i in range(256):
            root = np.array([((-1)**(i >> j)) for j in range(8)]) / 2
            if np.sum(root < 0) % 2 == 0:  # Even number of minus signs
                roots.append(root)
                
        return np.array(roots)
    
    def validate_weight_vector(self, weight: np.ndarray) -> bool:
        """Validate E8 weight vector constraints"""
        if len(weight) != 8:
            return False
            
        # Weight norm constraint
        if np.dot(weight, weight) > 2.01:  # Allow small numerical error
            return False
            
        return True
    
    def compute_root_proximity(self, weight: np.ndarray) -> float:
        """Compute minimum distance to E8 roots"""
        if not self.validate_weight_vector(weight):
            return np.inf
            
        distances = [np.linalg.norm(weight - root) for root in self.e8_roots]
        return min(distances)
    
    def validate_e8_consistency(self, configuration: Dict) -> float:
        """Validate overall E8 consistency of configuration"""
        try:
            weights = configuration.get('weight_vectors', [])
            if not weights:
                return 0.0
            
            consistency_scores = []
            for weight in weights:
                weight_array = np.array(weight)
                if self.validate_weight_vector(weight_array):
                    consistency_scores.append(1.0)
                else:
                    norm = np.linalg.norm(weight_array)
                    if norm <= 2.5:
                        consistency_scores.append(max(0.0, 1.0 - (norm - 2.0) / 0.5))
                    else:
                        consistency_scores.append(0.0)
            
            return np.mean(consistency_scores)
            
        except Exception as e:
            self.logger.error(f"E8 validation error: {e}")
            return 0.0

# Specialized validators for different mathematical claims