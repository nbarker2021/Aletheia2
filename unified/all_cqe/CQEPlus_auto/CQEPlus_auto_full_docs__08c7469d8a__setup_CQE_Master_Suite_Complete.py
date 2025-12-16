#!/usr/bin/env python3
"""
CQE Master Suite - Setup Configuration
=====================================

Complete setup configuration for the CQE (Cartan Quadratic Equivalence) Master Suite.
This is the definitive universal computational framework that integrates:

- E₈ lattice mathematics for geometric processing
- Sacred geometry principles for binary guidance  
- Mandelbrot fractal storage with bit-level precision
- Toroidal geometry for force field analysis
- Universal atomic operations for any data type

Author: CQE Development Team
Version: 1.0.0 Master
License: Universal Framework License
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure Python 3.8+
if sys.version_info < (3, 8):
    raise RuntimeError("CQE Master Suite requires Python 3.8 or higher")

# Read long description from README
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'documentation', 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "CQE Master Suite - The definitive universal computational framework"

# Read version from package
def read_version():
    version_file = os.path.join(os.path.dirname(__file__), 'cqe_framework', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return "1.0.0"

# Core dependencies
CORE_DEPENDENCIES = [
    'numpy>=1.20.0',
    'scipy>=1.7.0',
    'matplotlib>=3.3.0',
    'networkx>=2.6.0',
    'psutil>=5.8.0',
    'pillow>=8.0.0',
    'requests>=2.25.0',
    'pandas>=1.3.0',
    'sympy>=1.8.0'
]

# Optional dependencies for enhanced features
OPTIONAL_DEPENDENCIES = {
    'visualization': [
        'plotly>=5.0.0',
        'seaborn>=0.11.0',
        'bokeh>=2.3.0'
    ],
    'performance': [
        'numba>=0.53.0',
        'cython>=0.29.0',
        'joblib>=1.0.0'
    ],
    'scientific': [
        'scikit-learn>=0.24.0',
        'scikit-image>=0.18.0',
        'astropy>=4.2.0'
    ],
    'quantum': [
        'qiskit>=0.25.0',
        'cirq>=0.10.0'
    ],
    'web': [
        'flask>=2.0.0',
        'fastapi>=0.65.0',
        'uvicorn>=0.13.0',
        'websockets>=9.0.0'
    ],
    'database': [
        'sqlalchemy>=1.4.0',
        'pymongo>=3.11.0',
        'redis>=3.5.0'
    ]
}

# Development dependencies
DEV_DEPENDENCIES = [
    'pytest>=6.2.0',
    'pytest-cov>=2.12.0',
    'pytest-benchmark>=3.4.0',
    'black>=21.0.0',
    'flake8>=3.9.0',
    'mypy>=0.812',
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=0.5.0',
    'jupyter>=1.0.0',
    'notebook>=6.4.0'
]

# All optional dependencies combined
ALL_OPTIONAL = []
for deps in OPTIONAL_DEPENDENCIES.values():
    ALL_OPTIONAL.extend(deps)

setup(
    # Basic package information
    name="cqe-master-suite",
    version=read_version(),
    description="The definitive universal computational framework integrating E₈ mathematics, sacred geometry, and fractal storage",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    
    # Author and contact information
    author="CQE Development Team",
    author_email="cqe-dev@universalframework.org",
    url="https://github.com/cqe-framework/cqe-master-suite",
    project_urls={
        "Documentation": "https://cqe-master-suite.readthedocs.io/",
        "Source Code": "https://github.com/cqe-framework/cqe-master-suite",
        "Bug Tracker": "https://github.com/cqe-framework/cqe-master-suite/issues",
        "Discussions": "https://github.com/cqe-framework/cqe-master-suite/discussions"
    },
    
    # Package configuration
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    package_data={
        'cqe_framework': [
            'data/*.json',
            'data/*.yaml',
            'config/*.json',
            'config/*.yaml'
        ]
    },
    include_package_data=True,
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=CORE_DEPENDENCIES,
    extras_require={
        **OPTIONAL_DEPENDENCIES,
        'all': ALL_OPTIONAL,
        'dev': DEV_DEPENDENCIES,
        'complete': ALL_OPTIONAL + DEV_DEPENDENCIES
    },
    
    # Entry points for command-line tools
    entry_points={
        'console_scripts': [
            'cqe=cqe_framework.interfaces.cli:main',
            'cqe-bootstrap=bootstrap:main',
            'cqe-test=tests.golden_suite.golden_test_suite:main',
            'cqe-validate=cqe_framework.validation.framework:main',
            'cqe-analyze=tools.analyzers.system_analyzer:main',
            'cqe-visualize=tools.visualizers.geometric_visualizer:main'
        ]
    },
    
    # Classification
    classifiers=[
        # Development Status
        "Development Status :: 5 - Production/Stable",
        
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        
        # Topic
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Operating System
        "Operating System :: OS Independent",
        
        # Natural Language
        "Natural Language :: English"
    ],
    
    # Keywords for discovery
    keywords=[
        "cqe", "cartan", "quadratic", "equivalence",
        "e8", "lattice", "sacred", "geometry", "mandelbrot", "fractal",
        "universal", "computation", "mathematics", "physics",
        "quantum", "consciousness", "frequency", "resonance",
        "toroidal", "geometry", "atomic", "operations"
    ],
    
    # License
    license="MIT",
    
    # Zip safety
    zip_safe=False,
    
    # Additional metadata
    platforms=["any"],
    
    # Test suite
    test_suite="tests",
    tests_require=DEV_DEPENDENCIES,
    
    # Options for different installation methods
    options={
        'build_scripts': {
            'executable': '/usr/bin/env python3'
        }
    }
)

# Post-installation validation
def post_install_validation():
    """Validate installation after setup"""
    try:
        import cqe_framework
        validation_result = cqe_framework.validate_installation()
        
        if validation_result['status'] == 'SUCCESS':
            print("\n" + "="*60)
            print("CQE MASTER SUITE INSTALLATION SUCCESSFUL")
            print("="*60)
            print(f"Version: {validation_result['version']}")
            print("Core systems: ✓ Available")
            print("Validation framework: ✓ Available")
            print("\nNext steps:")
            print("1. Run bootstrap: python3 -m cqe_framework.bootstrap")
            print("2. Run tests: cqe-test")
            print("3. Explore examples: cd examples/")
            print("4. Read documentation: cd documentation/")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("CQE MASTER SUITE INSTALLATION FAILED")
            print("="*60)
            print(f"Error: {validation_result['message']}")
            print("Please check dependencies and try again.")
            print("="*60)
            
    except Exception as e:
        print(f"\nInstallation validation failed: {e}")
        print("Please check dependencies and installation.")

if __name__ == "__main__":
    # Run setup
    setup()
    
    # Validate installation
    post_install_validation()
