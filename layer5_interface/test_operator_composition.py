def test_operator_composition(sample_overlay):
    """Test applying multiple operators in sequence"""
    from canonicalizer import Canonicalizer
    from cqe.core.lattice import E8Lattice

    canonicalizer = Canonicalizer(E8Lattice())
    sample_overlay = canonicalizer.canonicalize(sample_overlay)

    op1 = RotationOperator()
    op2 = MidpointOperator()

    result = op2.apply(op1.apply(sample_overlay))

    assert len(result.provenance) >= 2
"""
Unit tests for E8 Lattice
"""

import pytest
import numpy as np
from cqe.core.lattice import E8Lattice

