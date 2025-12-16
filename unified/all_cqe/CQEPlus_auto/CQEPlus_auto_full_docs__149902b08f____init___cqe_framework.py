"""
CQE Framework - Complete Universal Computational Framework
=========================================================

The definitive implementation of the Cartan Quadratic Equivalence (CQE) system
integrating all discoveries, enhancements, and validations into a unified framework.

This framework combines:
- CQE manipulation through E₈ lattice operations
- Sacred Geometry guidance through digital root and frequency analysis
- Mandelbrot fractal storage with bit-level precision
- Toroidal geometry for force field analysis
- Universal atomic operations for any data type

Author: CQE Development Team
Version: 1.0.0 Master
License: Universal Framework License
"""

__version__ = "1.0.0"
__author__ = "CQE Development Team"
__license__ = "Universal Framework License"

# Core imports
from .core.e8_lattice import E8LatticeSystem
from .core.sacred_geometry import SacredGeometryEngine
from .core.mandelbrot_processor import MandelbrotFractalProcessor
from .core.toroidal_geometry import ToroidalGeometryModule
from .core.universal_atoms import UniversalAtomFactory, UniversalAtom
from .core.combination_engine import AtomicCombinationEngine
from .core.cqe_system import CQESystem

# Enhanced systems
from .enhanced.unified_system import UnifiedCQESystem
from .enhanced.tqf_governance import TQFGovernanceSystem
from .enhanced.uvibs_extension import UVIBSExtension

# Ultimate system
from .ultimate.complete_system import UltimateCQESystem

# Validation framework
from .validation.framework import CQEValidationFramework
from .validation.test_harness import CQETestHarness

# Domain adapters
from .domains.adapter import UniversalDomainAdapter
from .domains.mathematical import MathematicalDomainAdapter
from .domains.creative import CreativeDomainAdapter

# Interfaces
from .interfaces.cli import CQECommandLineInterface
from .interfaces.api import CQEAPIInterface
from .interfaces.web import CQEWebInterface

# Main system classes for easy access
__all__ = [
    # Core systems
    'CQESystem',
    'E8LatticeSystem', 
    'SacredGeometryEngine',
    'MandelbrotFractalProcessor',
    'ToroidalGeometryModule',
    'UniversalAtomFactory',
    'UniversalAtom',
    'AtomicCombinationEngine',
    
    # Enhanced systems
    'UnifiedCQESystem',
    'TQFGovernanceSystem', 
    'UVIBSExtension',
    
    # Ultimate system
    'UltimateCQESystem',
    
    # Validation
    'CQEValidationFramework',
    'CQETestHarness',
    
    # Domain adapters
    'UniversalDomainAdapter',
    'MathematicalDomainAdapter',
    'CreativeDomainAdapter',
    
    # Interfaces
    'CQECommandLineInterface',
    'CQEAPIInterface', 
    'CQEWebInterface'
]

# Framework constants
FRAMEWORK_NAME = "CQE Master Suite"
FRAMEWORK_DESCRIPTION = "Complete Universal Computational Framework"
SUPPORTED_PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]

# Quick access functions
def create_cqe_system(system_type="ultimate", **kwargs):
    """Create CQE system instance of specified type"""
    if system_type == "basic":
        return CQESystem(**kwargs)
    elif system_type == "unified":
        return UnifiedCQESystem(**kwargs)
    elif system_type == "ultimate":
        return UltimateCQESystem(**kwargs)
    else:
        raise ValueError(f"Unknown system type: {system_type}")

def get_framework_info():
    """Get comprehensive framework information"""
    return {
        'name': FRAMEWORK_NAME,
        'version': __version__,
        'description': FRAMEWORK_DESCRIPTION,
        'author': __author__,
        'license': __license__,
        'supported_python': SUPPORTED_PYTHON_VERSIONS,
        'core_systems': [
            'E₈ Lattice Operations',
            'Sacred Geometry Engine', 
            'Mandelbrot Fractal Processing',
            'Toroidal Geometry Module',
            'Universal Atomic Operations',
            'Combination Engine',
            'Validation Framework'
        ],
        'capabilities': [
            'Universal data processing',
            'Sacred geometry guidance',
            'Fractal storage optimization',
            'Atomic combination operations',
            'Mathematical validation',
            'Performance benchmarking',
            'Multi-domain adaptation'
        ]
    }

def validate_installation():
    """Validate CQE framework installation"""
    try:
        # Test core imports
        from .core.cqe_system import CQESystem
        from .ultimate.complete_system import UltimateCQESystem
        from .validation.framework import CQEValidationFramework
        
        # Test basic functionality
        system = CQESystem()
        validator = CQEValidationFramework()
        
        return {
            'status': 'SUCCESS',
            'message': 'CQE Framework installation validated successfully',
            'version': __version__,
            'core_systems_available': True,
            'validation_framework_available': True
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'CQE Framework installation validation failed: {str(e)}',
            'error': str(e)
        }

# Initialize framework logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
