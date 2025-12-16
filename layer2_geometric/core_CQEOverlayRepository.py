class CQEOverlayRepository:
    """Repository of overlay states for CQE test harness warm-starts"""
    
    def __init__(self):
        # Initialize E8 root system (representative subset)
        self.e8_roots = self._generate_e8_roots()
        self.overlay_states = []
        self.dimensional_scopes = {}
        self.angular_views = {}
        self.modulo_forms = {}
        
    def _generate_e8_roots(self) -> np.ndarray:
        """Generate representative E8 root vectors"""
        # E8 roots include all vectors of the form:
        # (±1, ±1, 0, 0, 0, 0, 0, 0) and cyclic permutations
        # (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs
        
        roots = []
        
        # Type 1: ±1, ±1 in two positions, 0 elsewhere
        for i in range(8):
            for j in range(i+1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        root = [0.0] * 8
                        root[i] = s1
                        root[j] = s2
                        roots.append(root)
        
        # Type 2: ±1/2 in all positions with even number of minus signs
        import itertools
        for signs in itertools.product([-0.5, 0.5], repeat=8):
            if sum(1 for s in signs if s < 0) % 2 == 0:  # Even number of minus signs
                roots.append(list(signs))
        
        return np.array(roots)
    
    def add_overlay_state(self, state: OverlayState):
        """Add a new overlay state to the repository"""
        self.overlay_states.append(state)
        
        # Categorize by dimensional scope
        scope_key = f"{state.domain}_{len(state.embedding)}D"
        if scope_key not in self.dimensional_scopes:
            self.dimensional_scopes[scope_key] = []
        self.dimensional_scopes[scope_key].append(state)
        
        # Analyze angular view
        angular_key = self._compute_angular_signature(state.embedding)
        if angular_key not in self.angular_views:
            self.angular_views[angular_key] = []
        self.angular_views[angular_key].append(state)
    
    def _compute_angular_signature(self, embedding: List[float]) -> str:
        """Compute angular signature for categorization"""
        v = np.array(embedding)
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return "zero_vector"
        
        # Normalize and compute angle classes
        v_norm = v / norm
        
        # Compute angles with standard basis vectors
        angles = []
        for i in range(8):
            basis = np.zeros(8)
            basis[i] = 1.0
            angle = np.arccos(np.clip(np.dot(v_norm, basis), -1, 1))
            angles.append(angle)
        
        # Discretize angles into octants
        angle_classes = [int(angle / (np.pi / 4)) for angle in angles]
        return "_".join(map(str, angle_classes))
    
    def compute_e8_distances(self, embedding: List[float]) -> List[E8NodeDistance]:
        """Compute distances to all E8 lattice points"""
        v = np.array(embedding)
        distances = []
        
        for i, root in enumerate(self.e8_roots):
            dist = np.linalg.norm(v - root)
            
            # Angular separation
            v_norm = np.linalg.norm(v)
            root_norm = np.linalg.norm(root)
            
            if v_norm > 1e-10 and root_norm > 1e-10:
                cos_angle = np.dot(v, root) / (v_norm * root_norm)
                angular_sep = np.arccos(np.clip(cos_angle, -1, 1))
            else:
                angular_sep = 0.0
            
            # Modulo form (lattice reduction)
            modulo_coords = [(v[j] - root[j]) % 2 for j in range(8)]
            modulo_form = "_".join([f"{x:.3f}" for x in modulo_coords])
            
            distances.append(E8NodeDistance(
                node_id=i,
                coordinates=root.tolist(),
                distance=dist,
                angular_separation=angular_sep,
                modulo_form=modulo_form
            ))
        
        return sorted(distances, key=lambda x: x.distance)

# Create overlay repository and populate with session data
overlay_repo = CQEOverlayRepository()

print(f"Generated {len(overlay_repo.e8_roots)} E8 root vectors")
print("E8 root system shape:", overlay_repo.e8_roots.shape)

# Sample the first few roots
print("\nFirst 10 E8 roots:")
for i in range(min(10, len(overlay_repo.e8_roots))):
    root = overlay_repo.e8_roots[i]
    print(f"Root {i}: [{', '.join([f'{x:5.1f}' for x in root])}] norm={np.linalg.norm(root):.3f}")# Now generate representative overlay states from the CQE session data
# These represent initial and final states across different test scenarios

# Simulate test run states based on session findings
test_scenarios = [
    {
        'domain': 'audio',
        'test_name': 'E8_Embedding_Accuracy',
        'initial_embedding': [0.2, -0.3, 0.1, 0.4, -0.2, 0.1, 0.3, -0.1],
        'final_embedding': [0.18, -0.29, 0.11, 0.39, -0.19, 0.12, 0.31, -0.09],
        'initial_objective': 0.847,
        'final_objective': 0.023,
        'iterations': 47
    },
    {
        'domain': 'scene_graph', 
        'test_name': 'Policy_Channel_Orthogonality',
        'initial_embedding': [0.5, 0.2, -0.1, 0.3, 0.1, -0.4, 0.2, 0.1],
        'final_embedding': [0.48, 0.21, -0.08, 0.32, 0.09, -0.38, 0.19, 0.11],
        'initial_objective': 1.234,
        'final_objective': 0.045,
        'iterations': 63
    },
    {
        'domain': 'permutation',
        'test_name': 'MORSR_Convergence', 
        'initial_embedding': [-0.3, 0.1, 0.4, -0.2, 0.5, 0.1, -0.1, 0.2],
        'final_embedding': [-0.31, 0.12, 0.41, -0.18, 0.52, 0.08, -0.12, 0.19],
        'initial_objective': 2.156,
        'final_objective': 0.089,
        'iterations': 82
    },
    {
        'domain': 'creative_ai',
        'test_name': 'TSP_Optimization_Quality',
        'initial_embedding': [0.1, -0.2, 0.3, 0.1, -0.1, 0.4, -0.3, 0.2],
        'final_embedding': [0.09, -0.18, 0.32, 0.12, -0.08, 0.42, -0.28, 0.21],
        'initial_objective': 3.421,
        'final_objective': 0.156,
        'iterations': 95
    },
    {
        'domain': 'scaling',
        'test_name': 'Scaling_Performance_64D',
        'initial_embedding': [0.4, 0.3, -0.2, -0.1, 0.2, -0.3, 0.1, 0.4],
        'final_embedding': [0.39, 0.31, -0.19, -0.08, 0.21, -0.29, 0.12, 0.38],
        'initial_objective': 1.876,
        'final_objective': 0.067,
        'iterations': 71
    },
    {
        'domain': 'distributed',
        'test_name': 'Distributed_MORSR_8_Nodes',
        'initial_embedding': [-0.1, 0.4, 0.2, -0.3, 0.1, 0.2, -0.4, 0.1],
        'final_embedding': [-0.09, 0.42, 0.19, -0.31, 0.12, 0.18, -0.39, 0.09],
        'initial_objective': 2.543,
        'final_objective': 0.134,
        'iterations': 58
    }
]

# Generate policy channels using harmonic decomposition