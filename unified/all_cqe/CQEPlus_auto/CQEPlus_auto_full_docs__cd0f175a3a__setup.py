from setuptools import setup, find_packages

setup(
    name="scene8",
    version="1.0.0-MVP",
    description="Geometry-First Generative Video AI",
    author="CQE Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=9.0.0",
    ],
    entry_points={
        "console_scripts": [
            "scene8=cli.scene8:main",
        ],
    },
    python_requires=">=3.8",
)
