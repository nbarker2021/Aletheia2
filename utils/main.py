def main():
    """Run all enhanced examples."""
    
    print("Enhanced CQE System - Legacy Integration Examples")
    print("=" * 60)
    
    try:
        # Run enhanced examples
        example_tqf_governance()
        example_uvibs_extension()
        example_hybrid_governance()
        example_scene_debugging()
        example_performance_comparison()
        
        print("\n" + "=" * 60)
        print("ALL ENHANCED EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ TQF Governance with quaternary encoding")
        print("✓ UVIBS 80D extensions with Monster governance")
        print("✓ Hybrid governance combining multiple approaches")
        print("✓ Scene-based debugging with 8×8 viewers")
        print("✓ Multi-window validation (W4/W80/TQF/Mirror)")
        print("✓ Performance comparison across governance types")
        
    except Exception as e:
        print(f"\nError running enhanced examples: {e}")
        print("This may be due to missing dependencies or configuration issues.")

if __name__ == "__main__":
    main()
"""
Comprehensive test suite for CQE System

Tests all major components and integration scenarios.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from cqe import CQESystem
from cqe.core import E8Lattice, MORSRExplorer, CQEObjectiveFunction
from cqe.core.parity_channels import ParityChannels
from cqe.domains import DomainAdapter
from cqe.validation import ValidationFramework
