class TestE8LatticeFoundations(unittest.TestCase):
    """Test E₈ lattice mathematical foundations"""
    
    def setUp(self):
        self.processor = E8LatticeProcessor()
    
    def test_root_system_completeness(self):
        """Test that E₈ root system has exactly 240 roots"""
        self.assertEqual(len(self.processor.root_system), 240)
    
    def test_root_vector_orthogonality(self):
        """Test orthogonality relationships between root vectors"""
        roots = self.processor.root_system
        
        # Test sample of root pairs for orthogonality or specific angles
        orthogonal_count = 0
        total_pairs = 0
        
        for i in range(0, min(50, len(roots))):
            for j in range(i+1, min(50, len(roots))):
                dot_product = np.dot(roots[i], roots[j])
                total_pairs += 1
                
                # Check for orthogonality (dot product ≈ 0)
                if abs(dot_product) < 1e-10:
                    orthogonal_count += 1
        
        # At least 30% of root pairs should be orthogonal
        orthogonal_ratio = orthogonal_count / total_pairs
        self.assertGreater(orthogonal_ratio, 0.3)
    
    def test_universal_embedding_existence(self):
        """Test that any data can be embedded in E₈ space"""
        test_data = [
            42, "hello", [1, 2, 3], {"key": "value"}, 3.14159,
            complex(1, 1), None, True, "sacred geometry"
        ]
        
        for data in test_data:
            embedding = self.processor.embed_data_in_e8(data)
            
            # Check embedding is 8-dimensional
            self.assertEqual(len(embedding), 8)
            
            # Check embedding is normalized
            norm = np.linalg.norm(embedding)
            self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_embedding_consistency(self):
        """Test that same data produces same embedding"""
        test_data = "consistency_test"
        
        embedding1 = self.processor.embed_data_in_e8(test_data)
        embedding2 = self.processor.embed_data_in_e8(test_data)
        
        np.testing.assert_array_almost_equal(embedding1, embedding2)
    
    def test_lattice_quality_calculation(self):
        """Test lattice quality calculation"""
        # Test with known good embedding
        good_embedding = self.processor.root_system[0]  # Use actual root
        quality = self.processor.calculate_lattice_quality(good_embedding)
        
        # Quality should be high for actual root
        self.assertGreater(quality, 0.8)
        
        # Test with random point
        random_point = np.random.randn(8)
        random_quality = self.processor.calculate_lattice_quality(random_point)
        
        # Random point should have lower quality
        self.assertLess(random_quality, quality)
