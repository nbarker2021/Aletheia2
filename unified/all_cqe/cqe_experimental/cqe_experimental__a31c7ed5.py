"""
PyTest configuration and fixtures
"""

import pytest
import numpy as np
from cqe.core.lattice import E8Lattice
from cqe.core.embedding import BabaiEmbedder
from cqe.core.phi import PhiComputer
from cqe.core.canonicalization import Canonicalizer
from cqe.core.overlay import CQEOverlay
from cqe.morsr.protocol import MORSRProtocol
from cqe.adapters.text import TextAdapter


@pytest.fixture
def e8_lattice():
    """E8 lattice instance"""
    return E8Lattice()


@pytest.fixture
def embedder(e8_lattice):
    """Babai embedder instance"""
    return BabaiEmbedder(e8_lattice)


@pytest.fixture
def phi_computer():
    """Phi computer instance"""
    return PhiComputer()


@pytest.fixture
def canonicalizer(e8_lattice):
    """Canonicalizer instance"""
    return Canonicalizer(e8_lattice)


@pytest.fixture
def morsr(phi_computer, canonicalizer):
    """MORSR protocol instance"""
    return MORSRProtocol(phi_computer, canonicalizer)


@pytest.fixture
def text_adapter():
    """Text adapter instance"""
    return TextAdapter()


@pytest.fixture
def sample_overlay():
    """Sample overlay for testing"""
    present = np.zeros(248, dtype=bool)
    present[0] = True  # Activate root
    present[240] = True  # Activate Cartan lane 0
    present[241] = True  # Activate Cartan lane 1

    w = np.zeros(248)
    w[0] = 1.0
    w[240] = 0.5
    w[241] = 0.3

    phi = np.zeros(248)
    phi[0] = 0.0
    phi[240] = np.pi/4
    phi[241] = -np.pi/4

    return CQEOverlay(
        present=present,
        w=w,
        phi=phi,
        pose={'domain_type': 'test'}
    )


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return "Quantum entanglement demonstrates non-local correlations between particles."
