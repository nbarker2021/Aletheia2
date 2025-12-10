"""
CQE Unified Runtime - Setup Configuration
A comprehensive implementation of the CQE (Consciousness Quantum Encoding) framework
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cqe-unified-runtime",
    version="4.0.0-beta",
    author="CQE Research Team",
    author_email="research@cqe.dev",
    description="A comprehensive implementation of the CQE framework with 90% completion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cqe-research/cqe-unified-runtime",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cqe=cqe_cli:main",
            "cqe-server=cqe_server:main",
            "cqe-explorer=cqe_explorer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml"],
    },
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/cqe-research/cqe-unified-runtime/issues",
        "Source": "https://github.com/cqe-research/cqe-unified-runtime",
        "Documentation": "https://cqe-unified-runtime.readthedocs.io/",
    },
)
