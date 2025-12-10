def save_embedding(output_path: str = "embeddings/e8_248_embedding.json") -> None:
    """Generate and save the E₈ embedding data."""
    roots = generate_e8_roots()
    cartan = generate_cartan_matrix()
    
    if not validate_e8_structure(roots, cartan):
        raise ValueError("Generated E₈ structure failed validation")
    
    data = {
        "name": "E8_lattice",
        "dimension": 8,
        "root_count": len(roots),
        "roots_8d": roots,
        "cartan_8x8": cartan,
        "metadata": {
            "generated_by": "CQE-MORSR Framework",
            "description": "Complete E₈ root system and Cartan matrix",
            "validation_passed": True
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"E₈ embedding saved to {output_path}")
    print(f"Generated {len(roots)} roots with 8×8 Cartan matrix")
