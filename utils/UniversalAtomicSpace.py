class UniversalAtomicSpace:
    """Complete atomic space managing all universal atoms"""
    
    def __init__(self):
        self.atoms: Dict[str, UniversalAtom] = {}
        self.factory = UniversalAtomFactory()
        self.combination_engine = AtomicCombinationEngine()
        
        # Space statistics
        self.total_atoms = 0
        self.total_storage_bits = 0
        self.combination_count = 0
        
        # Indexing for fast retrieval
        self.frequency_index: Dict[float, List[str]] = {}
        self.digital_root_index: Dict[int, List[str]] = {}
        self.fractal_behavior_index: Dict[str, List[str]] = {}
    
    def create_atom(self, data: Any, atom_id: str = None) -> str:
        """Create new universal atom from data"""
        if atom_id is None:
            atom_id = hashlib.md5(str(data).encode()).hexdigest()[:16]
        
        atom = self.factory.create_atom_from_data(data)
        self.atoms[atom_id] = atom
        
        # Update statistics
        self.total_atoms += 1
        self.total_storage_bits += atom.storage_size
        
        # Update indices
        self.update_indices(atom_id, atom)
        
        return atom_id
    
    def get_atom(self, atom_id: str) -> Optional[UniversalAtom]:
        """Retrieve atom by ID"""
        atom = self.atoms.get(atom_id)
        if atom:
            atom.access_count += 1
        return atom
    
    def combine_atoms(self, atom_id1: str, atom_id2: str, 
                     combination_type: AtomCombinationType = None) -> str:
        """Combine two atoms and return new atom ID"""
        atom1 = self.get_atom(atom_id1)
        atom2 = self.get_atom(atom_id2)
        
        if not atom1 or not atom2:
            raise ValueError("One or both atoms not found")
        
        # Determine combination type if not specified
        if combination_type is None:
            possible_types = self.combination_engine.can_combine(atom1, atom2)
            if not possible_types:
                raise ValueError("Atoms cannot be combined")
            combination_type = possible_types[0]  # Use first available type
        
        # Perform combination
        combined_atom = self.combination_engine.combine_atoms(atom1, atom2, combination_type)
        
        # Generate new ID for combined atom
        combined_id = f"COMBINED_{atom_id1}_{atom_id2}_{combination_type.value}"
        combined_id = hashlib.md5(combined_id.encode()).hexdigest()[:16]
        
        # Store combined atom
        self.atoms[combined_id] = combined_atom
        self.total_atoms += 1
        self.total_storage_bits += combined_atom.storage_size
        self.combination_count += 1
        
        # Update indices
        self.update_indices(combined_id, combined_atom)
        
        return combined_id
    
    def find_atoms_by_frequency(self, frequency: float, tolerance: float = 1.0) -> List[str]:
        """Find atoms by sacred frequency"""
        matching_atoms = []
        for freq, atom_ids in self.frequency_index.items():
            if abs(freq - frequency) <= tolerance:
                matching_atoms.extend(atom_ids)
        return matching_atoms
    
    def find_atoms_by_digital_root(self, digital_root: int) -> List[str]:
        """Find atoms by digital root"""
        return self.digital_root_index.get(digital_root, [])
    
    def find_atoms_by_fractal_behavior(self, behavior: str) -> List[str]:
        """Find atoms by fractal behavior"""
        return self.fractal_behavior_index.get(behavior, [])
    
    def get_combination_possibilities(self, atom_id: str) -> Dict[str, List[str]]:
        """Get all possible combinations for an atom"""
        atom = self.get_atom(atom_id)
        if not atom:
            return {}
        
        possibilities = {}
        
        for other_id, other_atom in self.atoms.items():
            if other_id != atom_id:
                combination_types = self.combination_engine.can_combine(atom, other_atom)
                if combination_types:
                    for combo_type in combination_types:
                        if combo_type.value not in possibilities:
                            possibilities[combo_type.value] = []
                        possibilities[combo_type.value].append(other_id)
        
        return possibilities
    
    def get_space_statistics(self) -> Dict[str, Any]:
        """Get comprehensive space statistics"""
        stats = {
            'total_atoms': self.total_atoms,
            'total_storage_bits': self.total_storage_bits,
            'average_atom_size_bits': self.total_storage_bits / max(1, self.total_atoms),
            'combination_count': self.combination_count,
            'frequency_distribution': {freq: len(atoms) for freq, atoms in self.frequency_index.items()},
            'digital_root_distribution': {root: len(atoms) for root, atoms in self.digital_root_index.items()},
            'fractal_behavior_distribution': {behavior: len(atoms) for behavior, atoms in self.fractal_behavior_index.items()}
        }
        
        return stats
    
    def update_indices(self, atom_id: str, atom: UniversalAtom):
        """Update all indices with new atom"""
        # Frequency index
        freq = atom.sacred_frequency
        if freq not in self.frequency_index:
            self.frequency_index[freq] = []
        self.frequency_index[freq].append(atom_id)
        
        # Digital root index
        root = atom.digital_root
        if root not in self.digital_root_index:
            self.digital_root_index[root] = []
        self.digital_root_index[root].append(atom_id)
        
        # Fractal behavior index
        behavior = atom.fractal_behavior
        if behavior not in self.fractal_behavior_index:
            self.fractal_behavior_index[behavior] = []
        self.fractal_behavior_index[behavior].append(atom_id)
    
    def export_space_state(self, filename: str):
        """Export complete space state to file"""
        space_data = {
            'atoms': {atom_id: {
                'e8_coordinates': atom.e8_coordinates.tolist(),
                'quad_encoding': atom.quad_encoding,
                'parity_channels': atom.parity_channels.tolist(),
                'digital_root': atom.digital_root,
                'sacred_frequency': atom.sacred_frequency,
                'binary_guidance': atom.binary_guidance,
                'rotational_pattern': atom.rotational_pattern,
                'fractal_coordinate': [atom.fractal_coordinate.real, atom.fractal_coordinate.imag],
                'fractal_behavior': atom.fractal_behavior,
                'compression_ratio': atom.compression_ratio,
                'iteration_depth': atom.iteration_depth,
                'storage_size': atom.storage_size,
                'combination_mask': atom.combination_mask,
                'creation_timestamp': atom.creation_timestamp,
                'access_count': atom.access_count,
                'combination_history': atom.combination_history
            } for atom_id, atom in self.atoms.items()},
            'statistics': self.get_space_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(space_data, f, indent=2)
