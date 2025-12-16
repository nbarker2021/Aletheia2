#!/usr/bin/env python3
"""
CQE Advanced Usage Example

Demonstrates:
- Custom operator sequences
- MORSR handshake analysis
- Cross-domain embedding
- Metric tracking over transformations
"""

from cqe import CQEClient
from cqe.core.phi import PhiComputer
from cqe.core.canonicalization import Canonicalizer
from cqe.morsr.protocol import MORSRProtocol
import numpy as np


def analyze_morsr_handshakes(client, text):
    """Analyze MORSR optimization handshakes"""
    print("\n=== MORSR Handshake Analysis ===\n")

    # Embed with optimization
    overlay = client.embed(text, optimize=True)

    # Get handshake log
    handshakes = client.morsr.get_handshake_log()

    print(f"Total handshakes: {len(handshakes)}")

    # Analyze acceptance
    accepted = [h for h in handshakes if h.accepted]
    rejected = [h for h in handshakes if not h.accepted]

    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Acceptance rate: {len(accepted)/len(handshakes):.1%}")

    # Show Φ trajectory
    print("\nΦ trajectory:")
    for i, h in enumerate(accepted[:5]):  # First 5 accepted
        print(f"  {i+1}. {h.operator_name:20s} ΔΦ={h.delta_phi:+.3f} → Φ={h.phi_after:.3f}")

    return overlay


def compare_operators(client, text):
    """Compare effects of different operators"""
    print("\n=== Operator Comparison ===\n")

    # Base overlay
    overlay = client.embed(text, optimize=False)
    base_metrics = client.get_phi_metrics(overlay)

    print(f"Base Φ: {base_metrics['phi_total']:.3f}")
    print("\nOperator effects:")

    operators = ["rotation", "midpoint", "parity"]

    for op_name in operators:
        transformed = client.apply_operator(op_name, overlay)
        new_metrics = client.get_phi_metrics(transformed)
        delta = new_metrics['phi_total'] - base_metrics['phi_total']

        print(f"  {op_name:12s} → ΔΦ={delta:+.3f}, Φ={new_metrics['phi_total']:.3f}")


def track_metric_evolution(client, text):
    """Track how metrics evolve through transformations"""
    print("\n=== Metric Evolution ===\n")

    overlay = client.embed(text, optimize=False)

    operators = ["rotation", "midpoint", "rotation"]

    print("Transformation sequence:")
    print(f"  Initial: Φ={client.get_phi_metrics(overlay)['phi_total']:.3f}")

    for i, op_name in enumerate(operators, 1):
        overlay = client.apply_operator(op_name, overlay)
        metrics = client.get_phi_metrics(overlay)

        print(f"  {i}. After {op_name}:")
        print(f"     Φ_total={metrics['phi_total']:.3f}")
        print(f"     Φ_geom={metrics['phi_geom']:.3f}, Φ_parity={metrics['phi_parity']:.1f}")


def cross_domain_analysis(client):
    """Analyze overlays across different content types"""
    print("\n=== Cross-Domain Analysis ===\n")

    contents = {
        'scientific': "Quantum entanglement demonstrates non-local correlations.",
        'code': "def fibonacci(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
        'prose': "The sun set slowly over the distant mountains."
    }

    overlays = {}

    for domain, text in contents.items():
        overlay = client.embed(text, optimize=True)
        metrics = client.get_phi_metrics(overlay)
        overlays[domain] = overlay

        print(f"{domain:12s}: Φ={metrics['phi_total']:.2f}, "
              f"Cartan={overlay.cartan_active}/8, "
              f"Active={len(overlay.active_slots)}/248")

    # Cross-domain similarity
    print("\nCross-domain similarities:")
    domains = list(overlays.keys())
    for i in range(len(domains)):
        for j in range(i+1, len(domains)):
            d1, d2 = domains[i], domains[j]

            similar = client.find_similar(overlays[d1], top_k=5)

            # Check if d2's overlay is in results
            d2_hash = overlays[d2].hash_id
            match = next((s for s in similar if s[0].hash_id == d2_hash), None)

            if match:
                distance = match[1]
                print(f"  {d1} ↔ {d2}: distance={distance:.3f}")


def main():
    print("=== CQE Advanced Usage Examples ===")

    client = CQEClient()

    test_text = "Neural networks approximate complex non-linear functions through hierarchical feature learning."

    # Run analyses
    analyze_morsr_handshakes(client, test_text)
    compare_operators(client, test_text)
    track_metric_evolution(client, test_text)
    cross_domain_analysis(client)

    print("\n✓ Advanced examples complete!")


if __name__ == "__main__":
    main()
