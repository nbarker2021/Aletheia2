class E8Position:
    """Represents a position in E₈ lattice space"""
    def __init__(self, coordinates: List[float]):
        self.coords = np.array(coordinates[:8])  # Ensure 8 dimensions
        
    def distance_to(self, other: 'E8Position') -> float:
        """Calculate E₈ lattice distance"""
        return np.linalg.norm(self.coords - other.coords)
    
    def angle_with(self, other: 'E8Position', reference: 'E8Position') -> float:
        """Calculate angle between vectors in E₈ space"""
        vec1 = self.coords - reference.coords
        vec2 = other.coords - reference.coords
        
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return 0
        
        cos_angle = dot_product / norms
        cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
        return math.acos(cos_angle)
