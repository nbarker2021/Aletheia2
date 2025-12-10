class TestE8Embedding:
    """Test E₈ embedding generation and validation."""

    def test_root_generation(self):
        """Test E₈ root system generation."""
        roots = generate_e8_roots()

        # Check count
        assert len(roots) == 240, f"Expected 240 roots, got {len(roots)}"

        # Check dimension
        for root in roots:
            assert len(root) == 8, f"Root dimension should be 8, got {len(root)}"

        # Check root norms (should be 2.0 for E₈)
        for i, root in enumerate(roots[:10]):  # Check first 10
            norm_sq = sum(x*x for x in root)
            assert abs(norm_sq - 2.0) < 1e-10, f"Root {i} has incorrect norm: {norm_sq}"

    def test_cartan_matrix(self):
        """Test Cartan matrix generation."""
        cartan = generate_cartan_matrix()

        # Check shape
        assert len(cartan) == 8, "Cartan matrix should be 8×8"
        assert all(len(row) == 8 for row in cartan), "Cartan matrix should be 8×8"

        # Check diagonal elements (should be 2)
        for i in range(8):
            assert cartan[i][i] == 2, f"Diagonal element {i} should be 2"

        # Check symmetry
        for i in range(8):
            for j in range(8):
                assert cartan[i][j] == cartan[j][i], f"Cartan matrix not symmetric at ({i},{j})"

    def test_embedding_save_load(self):
        """Test saving and loading E₈ embedding."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save embedding
            save_embedding(temp_path)
            assert Path(temp_path).exists(), "Embedding file was not created"

            # Load embedding
            data = load_embedding(temp_path)

            # Validate loaded data
            assert "roots_8d" in data, "Missing roots_8d in loaded data"
            assert "cartan_8x8" in data, "Missing cartan_8x8 in loaded data"
            assert len(data["roots_8d"]) == 240, "Incorrect number of roots in loaded data"
            assert len(data["cartan_8x8"]) == 8, "Incorrect Cartan matrix size"

        finally:
            # Cleanup
            if Path(temp_path).exists():
                Path(temp_path).unlink()
