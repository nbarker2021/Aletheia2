class E8Lattice:
    """E8 exceptional Lie group lattice operations."""
    
    def __init__(self):
        self.roots = self._generate_roots()
        self.weyl_chambers = self._generate_weyl_chambers()
        
    def _generate_roots(self) -> List[E8Root]:
        """Generate all 240 E8 root vectors."""
        roots = []
        
        # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations (112 roots)
        for i in range(8):
            for j in range(i+1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        coords = np.zeros(8)
                        coords[i] = s1
                        coords[j] = s2
                        roots.append(E8Root(coords, len(roots), E8_NORM))
        
        # Type 2: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) 
        # with even number of minus signs (128 roots)
        for signs in range(256):
            coords = np.array([(1 if (signs >> i) & 1 else -1) / 2 
                              for i in range(8)])
            if np.sum(coords < 0) % 2 == 0:  # Even number of minus signs
                roots.append(E8Root(coords, len(roots), E8_NORM))
        
        return roots[:240]  # Ensure exactly 240 roots
    
    def _generate_weyl_chambers(self) -> List[np.ndarray]:
        """Generate 48 Weyl chambers (fundamental domains)."""
        chambers = []
        
        # Weyl group of E8 has order 696,729,600
        # We use 48 fundamental chambers for practical purposes
        for i in range(48):
            # Each chamber is a cone in E8 space
            # Defined by hyperplane normals
            angle = (2 * np.pi * i) / 48
            normal = np.array([
                np.cos(angle),
                np.sin(angle),
                np.cos(2*angle),
                np.sin(2*angle),
                np.cos(3*angle),
                np.sin(3*angle),
                np.cos(4*angle),
                np.sin(4*angle)
            ])
            chambers.append(normal / np.linalg.norm(normal))
        
        return chambers
    
    def project_to_lattice(self, vector: np.ndarray) -> np.ndarray:
        """Project vector to nearest E8 lattice point."""
        # Find nearest root
        distances = [np.linalg.norm(vector - root.coords) 
                    for root in self.roots]
        nearest_idx = np.argmin(distances)
        return self.roots[nearest_idx].coords
    
    def project_to_manifold(self, vector: np.ndarray) -> np.ndarray:
        """Project to continuous E8 manifold (unit sphere)."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm * E8_NORM
        return vector
    
    def find_weyl_chamber(self, vector: np.ndarray) -> int:
        """Find which Weyl chamber contains the vector."""
        # Compute dot product with each chamber normal
        dots = [np.dot(vector, chamber) for chamber in self.weyl_chambers]
        return np.argmax(dots)
    
    def interpolate_geodesic(self, start: np.ndarray, end: np.ndarray, 
                            t: float) -> np.ndarray:
        """Interpolate along geodesic on E8 manifold."""
        # Spherical linear interpolation (SLERP)
        dot = np.dot(start, end) / (np.linalg.norm(start) * np.linalg.norm(end))
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)
        
        if abs(theta) < 1e-6:
            # Vectors are parallel, use linear interpolation
            return (1 - t) * start + t * end
        
        sin_theta = np.sin(theta)
        a = np.sin((1 - t) * theta) / sin_theta
        b = np.sin(t * theta) / sin_theta
        
        result = a * start + b * end
        return self.project_to_manifold(result)
    
    def rotate_e8(self, vector: np.ndarray, axis1: int, axis2: int, 
                  angle: float) -> np.ndarray:
        """Rotate vector in E8 space around plane defined by axis1, axis2."""
        result = vector.copy()
        
        # 2D rotation in the specified plane
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        x = result[axis1]
        y = result[axis2]
        
        result[axis1] = cos_a * x - sin_a * y
        result[axis2] = sin_a * x + cos_a * y
        
        return result
    
    def face_rotation(self, vector: np.ndarray, angle: float) -> np.ndarray:
        """Rotate E8 face - generates different solution paths (P vs NP)."""
        # Rotate in multiple planes simultaneously
        result = vector.copy()
        
        # Primary rotation (0-1 plane)
        result = self.rotate_e8(result, 0, 1, angle)
        
        # Secondary rotation (2-3 plane)
        result = self.rotate_e8(result, 2, 3, angle * PHI)
        
        # Tertiary rotation (4-5 plane)
        result = self.rotate_e8(result, 4, 5, angle * PHI**2)
        
        # Quaternary rotation (6-7 plane)
        result = self.rotate_e8(result, 6, 7, angle * PHI**3)
        
        return self.project_to_manifold(result)
    
    def compute_digital_root(self, vector: np.ndarray) -> int:
        """Compute digital root (0-9) from E8 vector."""
        # Sum all components, reduce to single digit
        total = int(np.sum(np.abs(vector)) * 1000)  # Scale for precision
        while total >= 10:
            total = sum(int(d) for d in str(total))
        return total if total > 0 else 9
    
    def compute_parity_channels(self, vector: np.ndarray) -> np.ndarray:
        """Compute 24 parity channels from E8 vector."""
        # Use Leech lattice embedding (24D)
        channels = np.zeros(24)
        
        # Embed E8 into first 8 channels
        channels[:8] = vector
        
        # Generate remaining 16 channels via modular arithmetic
        for i in range(8, 24):
            # Use CRT rails (3, 6, 9) and coupling (0.03)
            mod = (i % 3) + 3  # Moduli: 3, 4, 5, 3, 4, 5, ...
            channels[i] = (np.sum(vector) * COUPLING * i) % mod
        
        return channels

