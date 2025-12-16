class CQEMemoryManager:
    """CQE-based memory management system"""
    
    def __init__(self, max_atoms: int = 1000000):
        self.atoms: Dict[str, CQEAtom] = {}
        self.max_atoms = max_atoms
        self.access_history = deque(maxlen=max_atoms)
        self.governance_index = defaultdict(list)  # Index by governance state
        self.quad_index = defaultdict(list)  # Index by quad encoding
        self.e8_spatial_index = {}  # Spatial index for E8 embeddings
        self.lock = threading.RLock()
    
    def store_atom(self, atom: CQEAtom) -> str:
        """Store atom in CQE memory"""
        with self.lock:
            # Check capacity
            if len(self.atoms) >= self.max_atoms:
                self._evict_atoms()
            
            # Store atom
            self.atoms[atom.id] = atom
            self.access_history.append(atom.id)
            
            # Update indices
            self.governance_index[atom.governance_state].append(atom.id)
            self.quad_index[atom.quad_encoding].append(atom.id)
            self._update_e8_spatial_index(atom)
            
            return atom.id
    
    def retrieve_atom(self, atom_id: str) -> Optional[CQEAtom]:
        """Retrieve atom by ID"""
        with self.lock:
            if atom_id in self.atoms:
                self.access_history.append(atom_id)  # Update access
                return self.atoms[atom_id]
            return None
    
    def find_similar_atoms(self, target_atom: CQEAtom, max_distance: float = 2.0, 
                          limit: int = 10) -> List[Tuple[CQEAtom, float]]:
        """Find atoms similar to target atom"""
        with self.lock:
            similar = []
            
            for atom in self.atoms.values():
                if atom.id != target_atom.id and atom.is_compatible(target_atom):
                    distance = target_atom.distance_to(atom)
                    if distance <= max_distance:
                        similar.append((atom, distance))
            
            # Sort by distance and limit results
            similar.sort(key=lambda x: x[1])
            return similar[:limit]
    
    def find_by_governance(self, governance_state: str) -> List[CQEAtom]:
        """Find atoms by governance state"""
        with self.lock:
            atom_ids = self.governance_index.get(governance_state, [])
            return [self.atoms[aid] for aid in atom_ids if aid in self.atoms]
    
    def find_by_quad_pattern(self, quad_pattern: Tuple[int, int, int, int]) -> List[CQEAtom]:
        """Find atoms by quad encoding pattern"""
        with self.lock:
            atom_ids = self.quad_index.get(quad_pattern, [])
            return [self.atoms[aid] for aid in atom_ids if aid in self.atoms]
    
    def _evict_atoms(self):
        """Evict least recently used atoms"""
        # Remove oldest 10% of atoms
        evict_count = max(1, len(self.atoms) // 10)
        
        # Get least recently used atoms
        access_counts = defaultdict(int)
        for atom_id in self.access_history:
            access_counts[atom_id] += 1
        
        # Sort by access count
        sorted_atoms = sorted(self.atoms.keys(), 
                            key=lambda aid: access_counts.get(aid, 0))
        
        # Evict least accessed atoms
        for atom_id in sorted_atoms[:evict_count]:
            self._remove_atom(atom_id)
    
    def _remove_atom(self, atom_id: str):
        """Remove atom and update indices"""
        if atom_id not in self.atoms:
            return
        
        atom = self.atoms[atom_id]
        
        # Remove from indices
        self.governance_index[atom.governance_state].remove(atom_id)
        self.quad_index[atom.quad_encoding].remove(atom_id)
        
        # Remove from main storage
        del self.atoms[atom_id]
    
    def _update_e8_spatial_index(self, atom: CQEAtom):
        """Update E8 spatial index for efficient similarity search"""
        # Simplified spatial indexing - in practice would use k-d tree or similar
        e8_key = tuple(np.round(atom.e8_embedding, 1))  # Discretize for indexing
        if e8_key not in self.e8_spatial_index:
            self.e8_spatial_index[e8_key] = []
        self.e8_spatial_index[e8_key].append(atom.id)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        with self.lock:
            governance_counts = {state: len(atoms) for state, atoms in self.governance_index.items()}
            
            return {
                'total_atoms': len(self.atoms),
                'max_capacity': self.max_atoms,
                'utilization': len(self.atoms) / self.max_atoms,
                'governance_distribution': governance_counts,
                'unique_quad_patterns': len(self.quad_index),
                'e8_spatial_regions': len(self.e8_spatial_index)
            }
