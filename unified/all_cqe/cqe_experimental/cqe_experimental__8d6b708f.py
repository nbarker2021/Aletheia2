#!/usr/bin/env python3
"""
CQE Quickstart Example

Demonstrates basic CQE usage:
- Embedding text content
- Computing metrics
- Applying transformations
"""

from cqe import CQEClient

def main():
    print("=== CQE Quickstart Example ===\n")

    # Initialize client
    print("1. Initializing CQE client...")
    client = CQEClient()

    # Embed some text
    print("\n2. Embedding text content...")
    text = "Quantum entanglement demonstrates non-local correlations between particles."
    overlay = client.embed(text, domain="text", optimize=True)

    print(f"   Created overlay: {overlay.hash_id[:8]}")
    print(f"   Active slots: {len(overlay.active_slots)}/248")
    print(f"   Cartan active: {overlay.cartan_active}/8")

    # Get metrics
    print("\n3. Computing Φ metrics...")
    metrics = client.get_phi_metrics(overlay)
    for key, value in metrics.items():
        print(f"   {key}: {value:.3f}")

    # Apply transformation
    print("\n4. Applying midpoint operator...")
    transformed = client.apply_operator("midpoint", overlay)

    new_metrics = client.get_phi_metrics(transformed)
    delta = new_metrics['phi_total'] - metrics['phi_total']
    print(f"   Φ change: {delta:.3f}")

    # Embed more content
    print("\n5. Embedding multiple texts...")
    texts = [
        "Machine learning enables pattern recognition in data.",
        "Neural networks approximate complex functions.",
        "Deep learning uses multiple layers for hierarchical features."
    ]

    overlays = []
    for i, text in enumerate(texts):
        ov = client.embed(text, optimize=True)
        overlays.append(ov)
        print(f"   Text {i+1}: {ov.hash_id[:8]} (Φ={client.get_phi_metrics(ov)['phi_total']:.2f})")

    # Find similar
    print("\n6. Finding similar overlays...")
    query = overlays[0]
    similar = client.find_similar(query, top_k=3)

    for i, (ov, distance) in enumerate(similar):
        print(f"   {i+1}. {ov.hash_id[:8]} (distance={distance:.3f})")

    print("\n✓ Quickstart complete!")
    print("\nNext steps:")
    print("  - Try different domains (code, scientific)")
    print("  - Experiment with other operators")
    print("  - Explore MORSR handshake logs")


if __name__ == "__main__":
    main()
