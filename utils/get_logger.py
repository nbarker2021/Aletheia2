def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger
    """
    return logging.getLogger(name)
"""
High-level CQE client API
"""

from typing import List, Optional, Dict, Any
import numpy as np
from cqe.core.lattice import E8Lattice
from cqe.core.embedding import BabaiEmbedder
from cqe.core.phi import PhiComputer
from cqe.core.canonicalization import Canonicalizer
from cqe.core.overlay import CQEOverlay
from cqe.morsr.protocol import MORSRProtocol
from cqe.adapters.text import TextAdapter
from cqe.operators.base import CQEOperator

