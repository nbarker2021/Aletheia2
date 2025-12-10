class AtomicCombinationEngine:
    """Engine for combining universal atoms"""
    
    def __init__(self):
        self.combination_rules = {
            AtomCombinationType.RESONANT_BINDING: self.resonant_binding,
            AtomCombinationType.HARMONIC_COUPLING: self.harmonic_coupling,
            AtomCombinationType.GEOMETRIC_FUSION: self.geometric_fusion,
            AtomCombinationType.FRACTAL_NESTING: self.fractal_nesting,
            AtomCombinationType.QUANTUM_ENTANGLEMENT: self.quantum_entanglement,
            AtomCombinationType.PHASE_COHERENCE: self.phase_coherence
        }
    
    def can_combine(self, atom1: UniversalAtom, atom2: UniversalAtom) -> List[AtomCombinationType]:
        """Determine which combination types are possible"""
        possible_combinations = []
        
        # Check resonant binding (same frequency)
        if abs(atom1.sacred_frequency - atom2.sacred_frequency) < 1.0:
            possible_combinations.append(AtomCombinationType.RESONANT_BINDING)
        
        # Check harmonic coupling (harmonic frequencies)
        freq_ratio = atom1.sacred_frequency / atom2.sacred_frequency
        if self.is_harmonic_ratio(freq_ratio):
            possible_combinations.append(AtomCombinationType.HARMONIC_COUPLING)
        
        # Check geometric fusion (compatible digital roots)
        if self.are_geometrically_compatible(atom1.digital_root, atom2.digital_root):
            possible_combinations.append(AtomCombinationType.GEOMETRIC_FUSION)
        
        # Check fractal nesting (compatible behaviors)
        if self.can_fractal_nest(atom1.fractal_behavior, atom2.fractal_behavior):
            possible_combinations.append(AtomCombinationType.FRACTAL_NESTING)
        
        # Check quantum entanglement (E₈ correlation)
        if self.have_e8_correlation(atom1.e8_coordinates, atom2.e8_coordinates):
            possible_combinations.append(AtomCombinationType.QUANTUM_ENTANGLEMENT)
        
        # Check phase coherence (binary pattern compatibility)
        if self.have_phase_coherence(atom1.binary_guidance, atom2.binary_guidance):
            possible_combinations.append(AtomCombinationType.PHASE_COHERENCE)
        
        return possible_combinations
    
    def combine_atoms(self, atom1: UniversalAtom, atom2: UniversalAtom, 
                     combination_type: AtomCombinationType) -> UniversalAtom:
        """Combine two atoms using specified combination type"""
        
        if combination_type not in self.can_combine(atom1, atom2):
            raise ValueError(f"Cannot combine atoms using {combination_type}")
        
        combination_func = self.combination_rules[combination_type]
        return combination_func(atom1, atom2)
    
    def resonant_binding(self, atom1: UniversalAtom, atom2: UniversalAtom) -> UniversalAtom:
        """Combine atoms through resonant frequency binding"""
        # Average properties for resonant binding
        combined_e8 = (atom1.e8_coordinates + atom2.e8_coordinates) / 2
        combined_quad = tuple((a + b) // 2 for a, b in zip(atom1.quad_encoding, atom2.quad_encoding))
        combined_parity = (atom1.parity_channels + atom2.parity_channels) % 2
        
        # Use dominant sacred properties
        dominant_root = atom1.digital_root if atom1.sacred_frequency >= atom2.sacred_frequency else atom2.digital_root
        combined_frequency = (atom1.sacred_frequency + atom2.sacred_frequency) / 2
        
        # Combine fractal properties
        combined_fractal = (atom1.fractal_coordinate + atom2.fractal_coordinate) / 2
        combined_compression = (atom1.compression_ratio + atom2.compression_ratio) / 2
        
        factory = UniversalAtomFactory()
        
        return UniversalAtom(
            e8_coordinates=combined_e8,
            quad_encoding=combined_quad,
            parity_channels=combined_parity,
            digital_root=dominant_root,
            sacred_frequency=combined_frequency,
            binary_guidance=atom1.binary_guidance,  # Keep first atom's pattern
            rotational_pattern=atom1.rotational_pattern,
            fractal_coordinate=combined_fractal,
            fractal_behavior=atom1.fractal_behavior,
            compression_ratio=combined_compression,
            iteration_depth=max(atom1.iteration_depth, atom2.iteration_depth),
            bit_representation=b'',
            storage_size=0,
            combination_mask=0,
            creation_timestamp=np.random.random(),
            access_count=0,
            combination_history=[f"RESONANT_BINDING({atom1.digital_root},{atom2.digital_root})"]
        )
    
    def harmonic_coupling(self, atom1: UniversalAtom, atom2: UniversalAtom) -> UniversalAtom:
        """Combine atoms through harmonic frequency coupling"""
        # Create harmonic interference pattern
        freq_ratio = atom1.sacred_frequency / atom2.sacred_frequency
        harmonic_frequency = atom1.sacred_frequency * freq_ratio
        
        # E₈ coordinates show interference pattern
        combined_e8 = atom1.e8_coordinates * np.cos(freq_ratio) + atom2.e8_coordinates * np.sin(freq_ratio)
        
        # Fractal coordinates show beat pattern
        beat_frequency = abs(atom1.sacred_frequency - atom2.sacred_frequency)
        phase_shift = 2 * np.pi * beat_frequency / 1000.0
        combined_fractal = atom1.fractal_coordinate * complex(np.cos(phase_shift), np.sin(phase_shift))
        
        factory = UniversalAtomFactory()
        
        return UniversalAtom(
            e8_coordinates=combined_e8 / np.linalg.norm(combined_e8),
            quad_encoding=atom1.quad_encoding,
            parity_channels=(atom1.parity_channels + atom2.parity_channels) % 2,
            digital_root=factory.calculate_digital_root(harmonic_frequency),
            sacred_frequency=harmonic_frequency,
            binary_guidance=atom1.binary_guidance,
            rotational_pattern=atom1.rotational_pattern,
            fractal_coordinate=combined_fractal,
            fractal_behavior=atom1.fractal_behavior,
            compression_ratio=(atom1.compression_ratio + atom2.compression_ratio) / 2,
            iteration_depth=atom1.iteration_depth + atom2.iteration_depth,
            bit_representation=b'',
            storage_size=0,
            combination_mask=0,
            creation_timestamp=np.random.random(),
            access_count=0,
            combination_history=[f"HARMONIC_COUPLING({atom1.digital_root},{atom2.digital_root})"]
        )
    
    def geometric_fusion(self, atom1: UniversalAtom, atom2: UniversalAtom) -> UniversalAtom:
        """Combine atoms through sacred geometric fusion"""
        # Geometric fusion based on digital root relationships
        fused_root = (atom1.digital_root + atom2.digital_root) % 9
        if fused_root == 0:
            fused_root = 9
        
        # E₈ coordinates follow golden ratio relationships
        golden_ratio = (1 + np.sqrt(5)) / 2
        combined_e8 = atom1.e8_coordinates * golden_ratio + atom2.e8_coordinates / golden_ratio
        
        factory = UniversalAtomFactory()
        
        return UniversalAtom(
            e8_coordinates=combined_e8 / np.linalg.norm(combined_e8),
            quad_encoding=tuple((a * b) % 256 for a, b in zip(atom1.quad_encoding, atom2.quad_encoding)),
            parity_channels=(atom1.parity_channels * atom2.parity_channels) % 2,
            digital_root=fused_root,
            sacred_frequency=factory.sacred_frequencies[fused_root],
            binary_guidance=factory.binary_patterns[fused_root].value,
            rotational_pattern=factory.rotational_patterns[fused_root],
            fractal_coordinate=(atom1.fractal_coordinate * atom2.fractal_coordinate),
            fractal_behavior=atom1.fractal_behavior,
            compression_ratio=atom1.compression_ratio * atom2.compression_ratio,
            iteration_depth=max(atom1.iteration_depth, atom2.iteration_depth),
            bit_representation=b'',
            storage_size=0,
            combination_mask=0,
            creation_timestamp=np.random.random(),
            access_count=0,
            combination_history=[f"GEOMETRIC_FUSION({atom1.digital_root},{atom2.digital_root})"]
        )
    
    def fractal_nesting(self, atom1: UniversalAtom, atom2: UniversalAtom) -> UniversalAtom:
        """Combine atoms through fractal nesting"""
        # Nest smaller atom inside larger atom's fractal structure
        if atom1.compression_ratio > atom2.compression_ratio:
            outer_atom, inner_atom = atom1, atom2
        else:
            outer_atom, inner_atom = atom2, atom1
        
        # Nested fractal coordinate
        nested_coord = outer_atom.fractal_coordinate + inner_atom.fractal_coordinate * 0.1
        
        # E₈ coordinates show nested structure
        nested_e8 = outer_atom.e8_coordinates + inner_atom.e8_coordinates * 0.1
        
        return UniversalAtom(
            e8_coordinates=nested_e8 / np.linalg.norm(nested_e8),
            quad_encoding=outer_atom.quad_encoding,
            parity_channels=outer_atom.parity_channels,
            digital_root=outer_atom.digital_root,
            sacred_frequency=outer_atom.sacred_frequency,
            binary_guidance=outer_atom.binary_guidance,
            rotational_pattern=outer_atom.rotational_pattern,
            fractal_coordinate=nested_coord,
            fractal_behavior=outer_atom.fractal_behavior,
            compression_ratio=outer_atom.compression_ratio,
            iteration_depth=outer_atom.iteration_depth + inner_atom.iteration_depth,
            bit_representation=b'',
            storage_size=0,
            combination_mask=0,
            creation_timestamp=np.random.random(),
            access_count=0,
            combination_history=[f"FRACTAL_NESTING({outer_atom.digital_root},{inner_atom.digital_root})"]
        )
    
    def quantum_entanglement(self, atom1: UniversalAtom, atom2: UniversalAtom) -> UniversalAtom:
        """Combine atoms through quantum entanglement"""
        # Entangled state maintains correlation
        correlation = np.dot(atom1.e8_coordinates, atom2.e8_coordinates)
        
        # Entangled E₈ coordinates
        entangled_e8 = (atom1.e8_coordinates + atom2.e8_coordinates * correlation) / (1 + correlation)
        
        # Entangled properties maintain quantum correlation
        entangled_root = atom1.digital_root if correlation > 0 else atom2.digital_root
        
        factory = UniversalAtomFactory()
        
        return UniversalAtom(
            e8_coordinates=entangled_e8,
            quad_encoding=atom1.quad_encoding,
            parity_channels=(atom1.parity_channels + atom2.parity_channels) % 2,
            digital_root=entangled_root,
            sacred_frequency=factory.sacred_frequencies[entangled_root],
            binary_guidance=factory.binary_patterns[entangled_root].value,
            rotational_pattern=factory.rotational_patterns[entangled_root],
            fractal_coordinate=atom1.fractal_coordinate,
            fractal_behavior=atom1.fractal_behavior,
            compression_ratio=abs(correlation),
            iteration_depth=max(atom1.iteration_depth, atom2.iteration_depth),
            bit_representation=b'',
            storage_size=0,
            combination_mask=0,
            creation_timestamp=np.random.random(),
            access_count=0,
            combination_history=[f"QUANTUM_ENTANGLEMENT({atom1.digital_root},{atom2.digital_root})"]
        )
    
    def phase_coherence(self, atom1: UniversalAtom, atom2: UniversalAtom) -> UniversalAtom:
        """Combine atoms through phase coherence"""
        # Phase-locked combination
        phase_diff = self.calculate_phase_difference(atom1.binary_guidance, atom2.binary_guidance)
        
        # Coherent E₈ coordinates
        coherent_e8 = atom1.e8_coordinates * np.cos(phase_diff) + atom2.e8_coordinates * np.sin(phase_diff)
        
        return UniversalAtom(
            e8_coordinates=coherent_e8 / np.linalg.norm(coherent_e8),
            quad_encoding=tuple((a + b) % 256 for a, b in zip(atom1.quad_encoding, atom2.quad_encoding)),
            parity_channels=(atom1.parity_channels + atom2.parity_channels) % 2,
            digital_root=atom1.digital_root,
            sacred_frequency=atom1.sacred_frequency,
            binary_guidance=atom1.binary_guidance,
            rotational_pattern=atom1.rotational_pattern,
            fractal_coordinate=(atom1.fractal_coordinate + atom2.fractal_coordinate) / 2,
            fractal_behavior=atom1.fractal_behavior,
            compression_ratio=(atom1.compression_ratio + atom2.compression_ratio) / 2,
            iteration_depth=max(atom1.iteration_depth, atom2.iteration_depth),
            bit_representation=b'',
            storage_size=0,
            combination_mask=0,
            creation_timestamp=np.random.random(),
            access_count=0,
            combination_history=[f"PHASE_COHERENCE({atom1.digital_root},{atom2.digital_root})"]
        )
    
    def is_harmonic_ratio(self, ratio: float) -> bool:
        """Check if frequency ratio is harmonic"""
        harmonic_ratios = [1/2, 2/3, 3/4, 4/5, 5/6, 1.0, 6/5, 5/4, 4/3, 3/2, 2.0]
        return any(abs(ratio - hr) < 0.1 for hr in harmonic_ratios)
    
    def are_geometrically_compatible(self, root1: int, root2: int) -> bool:
        """Check if digital roots are geometrically compatible"""
        # Sacred geometry compatibility rules
        compatible_pairs = [
            (3, 6), (6, 9), (9, 3),  # Primary sacred triangle
            (1, 4), (4, 7), (7, 1),  # Secondary triangle
            (2, 5), (5, 8), (8, 2)   # Tertiary triangle
        ]
        return (root1, root2) in compatible_pairs or (root2, root1) in compatible_pairs
    
    def can_fractal_nest(self, behavior1: str, behavior2: str) -> bool:
        """Check if fractal behaviors can nest"""
        nesting_rules = {
            'BOUNDED': ['PERIODIC', 'BOUNDARY'],
            'ESCAPING': ['BOUNDED', 'BOUNDARY'],
            'BOUNDARY': ['BOUNDED', 'ESCAPING', 'PERIODIC'],
            'PERIODIC': ['BOUNDED']
        }
        return behavior2 in nesting_rules.get(behavior1, [])
    
    def have_e8_correlation(self, coords1: np.ndarray, coords2: np.ndarray) -> bool:
        """Check if E₈ coordinates have significant correlation"""
        correlation = abs(np.dot(coords1, coords2))
        return correlation > 0.5
    
    def have_phase_coherence(self, binary1: str, binary2: str) -> bool:
        """Check if binary patterns have phase coherence"""
        # Calculate Hamming distance
        hamming_distance = sum(b1 != b2 for b1, b2 in zip(binary1, binary2))
        return hamming_distance <= 1  # Allow 1 bit difference
    
    def calculate_phase_difference(self, binary1: str, binary2: str) -> float:
        """Calculate phase difference between binary patterns"""
        # Convert binary to phase
        phase1 = sum(int(b) * (2**i) for i, b in enumerate(reversed(binary1)))
        phase2 = sum(int(b) * (2**i) for i, b in enumerate(reversed(binary2)))
        
        return abs(phase1 - phase2) * np.pi / 8.0
