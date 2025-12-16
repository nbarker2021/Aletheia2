#!/usr/bin/env python3
"""
CQE Installation Verification Script

Verifies that all components are properly installed and functional.
"""

import sys
from pathlib import Path


def check_imports():
    """Verify all core imports work"""
    print("Checking imports...")

    try:
        from cqe import CQEClient, __version__
        from cqe.core.lattice import E8Lattice
        from cqe.core.overlay import CQEOverlay
        from cqe.morsr.protocol import MORSRProtocol
        from cqe.operators.rotation import RotationOperator
        print(f"  ✓ All imports successful (CQE v{__version__})")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def check_client():
    """Verify client initialization"""
    print("Checking client initialization...")

    try:
        from cqe import CQEClient
        client = CQEClient()
        print("  ✓ Client initialized successfully")
        return True, client
    except Exception as e:
        print(f"  ✗ Client initialization failed: {e}")
        return False, None


def check_embedding(client):
    """Verify embedding functionality"""
    print("Checking embedding...")

    try:
        overlay = client.embed("Test content", optimize=False)
        assert overlay.hash_id is not None
        assert len(overlay.active_slots) > 0
        print(f"  ✓ Embedding successful (hash: {overlay.hash_id[:8]})")
        return True
    except Exception as e:
        print(f"  ✗ Embedding failed: {e}")
        return False


def check_optimization(client):
    """Verify MORSR optimization"""
    print("Checking MORSR optimization...")

    try:
        overlay = client.embed("Optimization test", optimize=True)
        assert overlay.hash_id is not None
        print(f"  ✓ Optimization successful")
        return True
    except Exception as e:
        print(f"  ✗ Optimization failed: {e}")
        return False


def check_metrics(client):
    """Verify Φ metrics computation"""
    print("Checking Φ metrics...")

    try:
        overlay = client.embed("Metrics test", optimize=False)
        metrics = client.get_phi_metrics(overlay)

        required_keys = {'phi_total', 'phi_geom', 'phi_parity', 'phi_sparsity', 'phi_kissing'}
        assert required_keys.issubset(metrics.keys())

        print(f"  ✓ Metrics computed (Φ={metrics['phi_total']:.2f})")
        return True
    except Exception as e:
        print(f"  ✗ Metrics computation failed: {e}")
        return False


def check_operators(client):
    """Verify operator application"""
    print("Checking operators...")

    try:
        overlay = client.embed("Operator test", optimize=False)
        transformed = client.apply_operator("midpoint", overlay)

        assert transformed.hash_id is not None
        assert len(transformed.provenance) > len(overlay.provenance)

        print("  ✓ Operators working")
        return True
    except Exception as e:
        print(f"  ✗ Operator application failed: {e}")
        return False


def check_data_dirs():
    """Verify data directories exist"""
    print("Checking data directories...")

    dirs = [
        "data/overlays",
        "data/rag",
        "data/checkpoints",
        "data/golden"
    ]

    all_exist = True
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} missing")
            all_exist = False

    return all_exist


def main():
    """Run all verification checks"""
    print("="*60)
    print("CQE Installation Verification")
    print("="*60)
    print()

    results = []

    # Run checks
    results.append(("Imports", check_imports()))

    success, client = check_client()
    results.append(("Client", success))

    if client:
        results.append(("Embedding", check_embedding(client)))
        results.append(("Optimization", check_optimization(client)))
        results.append(("Metrics", check_metrics(client)))
        results.append(("Operators", check_operators(client)))

    results.append(("Data directories", check_data_dirs()))

    # Summary
    print()
    print("="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for check_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{check_name:20s} {status}")

    print()
    print(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! CQE is properly installed.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
