def run_validation_suite():
    """Run complete validation of P vs NP proof claims"""
    print("="*60)
    print("P ≠ NP E8 PROOF COMPUTATIONAL VALIDATION")
    print("="*60)

    validator = E8WeylChamberGraph()

    # Test 1: Variable encoding validation
    print("\n=== Test 1: SAT to E8 Encoding ===")
    test_assignments = [
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0]
    ]

    for i, assignment in enumerate(test_assignments):
        chamber = validator.sat_to_chamber(assignment)
