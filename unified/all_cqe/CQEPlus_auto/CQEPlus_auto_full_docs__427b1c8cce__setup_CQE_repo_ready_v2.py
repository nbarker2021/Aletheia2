from setuptools import setup, find_packages

setup(
    name="cqe",
    version="0.1.0",
    description="CQE — Cartan Quadratic Equivalence",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[req.strip() for req in open("requirements.txt").read().splitlines() if req.strip()],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "cqe-harness=cqe.cli.harness_cli:main",
            "cqe-bootstrap=cqe.cli.bootstrap:main",
        ]
    },
)
