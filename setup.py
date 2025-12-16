"""
Aletheia2 - Morphonic Operation Platform
========================================

A complete AI reasoning system using geometric embeddings, lattice structures
(E8, Leech, 24 Niemeier), and constraint-first reasoning to eliminate ambiguity
before computation.

Installation:
    pip install -e .

Usage:
    from aletheia2 import UnifiedRuntime
    runtime = UnifiedRuntime()
    state = runtime.process([1, 2, 3, 4, 5, 6, 7, 8])
"""

from setuptools import setup, find_packages
import os

VERSION = "2.0.0"

# Read the README file
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

# Read requirements
requirements = ["numpy>=1.21.0"]
if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aletheia2",
    version=VERSION,
    author="Manus AI",
    author_email="contact@manus.im",
    description="Morphonic Operation Platform - AI reasoning with geometric embeddings and lattice structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nbarker2021/Aletheia2",
    packages=find_packages(exclude=["tests", "docs", "integration", "checkpoints"]),
    py_modules=[
        "runtime",
        "unified_runtime",
        "geo_transformer",
        "layer1_morphonic",
        "layer2_geometric",
        "layer3_operational",
        "layer4_governance",
        "layer5_interface",
        "layer6_orchestration",
        "layer7_applications",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
        ],
        "full": [
            "scipy>=1.9.0",
            "matplotlib>=3.6.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aletheia2=unified_runtime:main",
            "aletheia2-runtime=runtime:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/nbarker2021/Aletheia2/issues",
        "Source": "https://github.com/nbarker2021/Aletheia2",
    },
)
