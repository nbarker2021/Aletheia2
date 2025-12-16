"""
CQE-MORSR System

Cartan-Quadratic Equivalence with Multi-Objective Random Search and Repair
for geometric complexity analysis and Millennium Prize Problem exploration.
"""

__version__ = "1.0.0"
__author__ = "CQE Build Space"

from .domain_adapter import DomainAdapter
from .e8_lattice import E8Lattice  
from .parity_channels import ParityChannels
from .objective_function import CQEObjectiveFunction
from .morsr_explorer import MORSRExplorer
from .chamber_board import ChamberBoard, ConstructionType, PolicyChannel
from .cqe_runner import CQERunner

__all__ = [
    "DomainAdapter",
    "E8Lattice", 
    "ParityChannels",
    "CQEObjectiveFunction",
    "MORSRExplorer", 
    "ChamberBoard",
    "ConstructionType",
    "PolicyChannel",
    "CQERunner"
]
