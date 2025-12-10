def demonstrate_semantic_extraction():
    """Demonstrate semantic extraction with a concrete example"""
    
    print("DEMONSTRATION: Semantic Extraction from Geometric Processing")
    print("Example: Processing the sentence 'The cat sat on the mat'")
    print()
    
    # Simulate final E₈ configuration after geometric processing
    geometric_config = {
        'the': E8Position([1.2, 0.3, 0.1, 0.0, 0.2, 0.0, 0.0, 0.1]),
        'cat': E8Position([0.1, 1.1, 0.4, 0.2, 0.0, 0.3, 0.1, 0.0]),
        'sat': E8Position([0.0, 0.2, 1.3, 0.6, 0.1, 0.0, 0.4, 0.0]),
        'on': E8Position([0.3, 0.1, 0.5, 1.0, 0.0, 0.2, 0.0, 0.3]),
        'mat': E8Position([0.0, 0.4, 0.2, 0.3, 0.1, 0.8, 0.0, 0.2])
    }
    
    print("Initial E₈ Configuration:")
    for entity, position in geometric_config.items():
        coords_str = ', '.join(f'{x:.1f}' for x in position.coords)
        print(f"  {entity}: [{coords_str}]")
    
    # Extract semantics
    extractor = SemanticExtractor()
    semantic_results = extractor.extract_semantics_from_configuration(geometric_config)
    
    # Display final results
    print("\n" + "=" * 60)
    print("FINAL SEMANTIC EXTRACTION RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Consistency Score: {semantic_results['consistency_score']:.3f}")
    
    print("\nConfidence Scores:")
    for semantic_type, confidence in semantic_results['confidence_scores'].items():
        print(f"  {semantic_type}: {confidence:.3f}")
    
    print("\nExtracted Semantic Content:")
    semantics = semantic_results['validated_semantics']
    
    if 'holistic_semantics' in semantics:
        holistic = semantics['holistic_semantics']
        print(f"  Overall Theme: {holistic.get('overall_theme', 'N/A')}")
        print(f"  Dominant Pattern: {holistic.get('dominant_relationship_pattern', 'N/A')}")
        print(f"  Semantic Coherence: {holistic.get('semantic_coherence', 0):.3f}")
    
    if 'emergent_properties' in semantics:
        print(f"  Emergent Properties: {', '.join(semantics['emergent_properties'])}")
    
    print("\nKey Relationships Discovered:")
    if 'relational_semantics' in semantics:
        for (entity1, entity2), relationship in semantics['relational_semantics'].items():
            print(f"  {entity1} ↔ {entity2}: {relationship}")
    
    print("\n" + "=" * 60)
    print("SEMANTIC EXTRACTION COMPLETE")
    print("Meaning successfully derived from pure geometric relationships!")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_semantic_extraction()
"""
Setup script for CQE (Cartan Quadratic Equivalence) System
"""

from setuptools import setup, find_packages
import os

# Read README file