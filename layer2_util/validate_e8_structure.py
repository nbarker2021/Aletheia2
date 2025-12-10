def validate_e8_structure(roots: List[List[float]], cartan: List[List[int]]) -> bool:
    """Validate the E₈ structure properties."""
    # Check root count
    if len(roots) != 240:
        return False
    
    # Check root dimension
    if not all(len(root) == 8 for root in roots):
        return False
    
    # Check Cartan matrix shape
    if len(cartan) != 8 or not all(len(row) == 8 for row in cartan):
        return False
    
    # Verify some root norms (should be 2.0)
    for root in roots[:10]:  # Check first 10
        norm_sq = sum(x*x for x in root)
        if abs(norm_sq - 2.0) > 1e-10:
            return False
    
    return True
