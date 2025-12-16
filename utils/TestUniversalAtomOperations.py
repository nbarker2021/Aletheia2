class TestUniversalAtomOperations(unittest.TestCase):
    """Test Universal Atom operations"""
    
    def setUp(self):
        self.cqe = UltimateCQESystem()
    
    def test_atom_creation(self):
        """Test Universal Atom creation"""
        test_data = "test atom creation"
        
        atom_id = self.cqe.create_universal_atom(test_data)
        
        # Check atom ID is valid
        self.assertIsInstance(atom_id, str)
        self.assertIn(atom_id, self.cqe.atoms)
        
        # Check atom properties
        atom = self.cqe.get_atom(atom_id)
        self.assertIsInstance(atom, UniversalAtom)
        self.assertEqual(atom.original_data, test_data)
        self.assertEqual(len(atom.e8_coordinates), 8)
        self.assertIn(atom.digital_root, range(1, 10))
        self.assertIsInstance(atom.fractal_coordinate, complex)
    
    def test_atom_combination(self):
        """Test atomic combination"""
        # Create two atoms
        atom_id1 = self.cqe.create_universal_atom(432)
        atom_id2 = self.cqe.create_universal_atom("sacred")
        
        # Combine atoms
        combined_id = self.cqe.combine_atoms(atom_id1, atom_id2)
        
        if combined_id:  # Combination succeeded
            # Check combined atom exists
            self.assertIn(combined_id, self.cqe.atoms)
            
            # Check combination is recorded
            combination_key = f"{atom_id1}+{atom_id2}"
            self.assertIn(combination_key, self.cqe.atom_combinations)
    
    def test_geometry_first_processing(self):
        """Test geometry-first processing paradigm"""
        test_data = "geometry first test"
        
        result = self.cqe.process_data_geometry_first(test_data)
        
        # Check result structure
        self.assertIn('atom_id', result)
        self.assertIn('geometric_result', result)
        self.assertIn('semantic_result', result)
        self.assertIn('validation', result)
        
        # Check geometric result completeness
        geo_result = result['geometric_result']
        self.assertIn('e8_embedding', geo_result)
        self.assertIn('sacred_geometry', geo_result)
        self.assertIn('fractal_analysis', geo_result)
        self.assertIn('toroidal_analysis', geo_result)
        
        # Check validation scores
        validation = result['validation']
        self.assertIn('mathematical_validity', validation)
        self.assertIn('geometric_consistency', validation)
        self.assertIn('semantic_coherence', validation)
    
    def test_system_analysis(self):
        """Test system pattern analysis"""
        # Create several atoms
        test_data = [432, "sacred", [1, 2, 3], {"test": "data"}, 3.14159]
        
        for data in test_data:
            self.cqe.create_universal_atom(data)
        
        # Analyze patterns
        analysis = self.cqe.analyze_system_patterns()
        
        # Check analysis completeness
        self.assertIn('total_atoms', analysis)
        self.assertIn('digital_root_distribution', analysis)
        self.assertIn('fractal_behavior_distribution', analysis)
        self.assertIn('force_classification_distribution', analysis)
        self.assertIn('average_compression_ratio', analysis)
        self.assertIn('average_validation_scores', analysis)
        
        # Check data consistency
        self.assertEqual(analysis['total_atoms'], len(test_data))
        self.assertGreater(analysis['average_compression_ratio'], 0)
