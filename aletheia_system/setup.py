"""
Setup script for Aletheia CQE Operating System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="aletheia-cqe",
    version="1.0.0",
    author="Aletheia Project",
    description="Complete Cartan Quadratic Equivalence (CQE) Geometric Consciousness System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aletheia-project/aletheia-cqe",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "sympy>=1.12",
    ],
    entry_points={
        "console_scripts": [
            "aletheia=aletheia:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

