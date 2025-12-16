"""
Setup script for CQE (Cartan Quadratic Equivalence) System
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "CQE (Cartan Quadratic Equivalence) System - Universal mathematical framework using E₈ geometry"

setup(
    name="cqe-system",
    version="1.0.0",
    author="CQE Research Consortium",
    author_email="research@cqe-system.org",
    description="Universal mathematical framework using E₈ exceptional Lie group geometry",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/cqe-research/cqe-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'networkx>=2.6.0',
        'sympy>=1.8.0',
        'numba>=0.54.0',
        'tqdm>=4.62.0',
        'pytest>=6.2.0',
        'jupyter>=1.0.0'
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.0.0',
            'flake8>=3.9.0',
            'mypy>=0.910',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
        ],
        'visualization': [
            'plotly>=5.0.0',
            'seaborn>=0.11.0',
            'bokeh>=2.3.0',
        ],
        'optimization': [
            'cvxpy>=1.1.0',
            'pulp>=2.4.0',
            'optuna>=2.8.0',
        ]
    },
    package_data={
        'cqe': [
            'data/*.json',
            'data/*.csv',
            'embeddings/*.json',
            'config/*.yaml',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        'mathematics',
        'lie-groups',
        'e8-lattice',
        'optimization',
        'artificial-intelligence',
        'complexity-theory',
        'millennium-problems',
        'geometric-algorithms',
        'parity-channels',
        'morsr-protocol'
    ],
)