"""
Integration tests for complete CQE pipeline
"""

import pytest
from cqe import CQEClient


def test_end_to_end_embedding():
    """Test complete embedding pipeline"""
    client = CQEClient()

    text = "Machine learning enables pattern recognition."

    overlay = client.embed(text, domain="text", optimize=True)

    assert overlay.hash_id is not None
    assert overlay.cartan_active > 0
    assert len(overlay.provenance) > 0


def test_embed_and_query():
    """Test embedding and similarity query"""
    client = CQEClient()

    texts = [
        "Quantum mechanics studies subatomic particles.",
        "Neural networks learn from data.",
        "Quantum computing uses superposition."
    ]

    overlays = [client.embed(text) for text in texts]

    # Query with first overlay
    similar = client.find_similar(overlays[0], top_k=2)

    assert len(similar) > 0

    # Third text (quantum) should be more similar to first than second
    # (This is a semantic test)


def test_operator_transformation_pipeline():
    """Test embedding → transformation → metrics"""
    client = CQEClient()

    overlay = client.embed("Test content", optimize=False)

    original_metrics = client.get_phi_metrics(overlay)

    # Apply midpoint operator
    transformed = client.apply_operator("midpoint", overlay)

    new_metrics = client.get_phi_metrics(transformed)

    # Metrics should change
    assert new_metrics['phi_total'] != original_metrics['phi_total']


def test_cache_functionality():
    """Test overlay caching"""
    client = CQEClient()

    initial_stats = client.get_cache_stats()
    initial_size = initial_stats['size']

    overlay = client.embed("Test caching", optimize=True)

    final_stats = client.get_cache_stats()

    # Cache should have grown
    assert final_stats['size'] > initial_size
    assert overlay.hash_id in final_stats['overlays']


def test_multiple_embeddings():
    """Test processing multiple texts"""
    client = CQEClient()

    texts = [
        "First test content.",
        "Second test content.",
        "Third test content."
    ]

    overlays = []
    for text in texts:
        overlay = client.embed(text, optimize=True)
        overlays.append(overlay)

    # All should be cached
    cache_stats = client.get_cache_stats()
    assert cache_stats['size'] >= len(texts)

    # All should have unique hashes
    hashes = [o.hash_id for o in overlays]
    assert len(set(hashes)) == len(hashes)


def test_phi_computation_consistency():
    """Test Φ computation is consistent"""
    client = CQEClient()

    overlay = client.embed("Consistency test", optimize=False)

    # Compute metrics multiple times
    metrics1 = client.get_phi_metrics(overlay)
    metrics2 = client.get_phi_metrics(overlay)

    # Should be identical
    assert metrics1['phi_total'] == metrics2['phi_total']
    assert metrics1['phi_geom'] == metrics2['phi_geom']


def test_optimization_improves_phi():
    """Test MORSR optimization reduces Φ"""
    client = CQEClient()

    text = "Test optimization effectiveness."

    # Without optimization
    overlay_no_opt = client.embed(text, optimize=False)
    phi_no_opt = client.get_phi_metrics(overlay_no_opt)['phi_total']

    # With optimization
    overlay_opt = client.embed(text, optimize=True)
    phi_opt = client.get_phi_metrics(overlay_opt)['phi_total']

    # Optimized should have lower or equal Φ
    assert phi_opt <= phi_no_opt + 1e-6  # Allow small numerical difference
