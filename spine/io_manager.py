"""
I/O Manager - Atomization and Format Conversion

Converts external data to/from CQE Atoms.
Handles tokenization and E8 embedding.
"""

import json
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np

from spine.kernel import CQEAtom
from spine.speedlight import get_speedlight, receipted


class GeoTokenizer:
    """
    Geometric Tokenizer.
    
    Converts text and other data into geometric tokens.
    Each token is a position in geometric space.
    """
    
    def __init__(self, dim: int = 8):
        self.dim = dim
        self._vocab: Dict[str, np.ndarray] = {}
    
    def tokenize(self, text: str) -> List[np.ndarray]:
        """Convert text to geometric tokens."""
        tokens = []
        for word in text.lower().split():
            if word not in self._vocab:
                # Generate deterministic embedding from word hash
                h = hashlib.sha256(word.encode()).digest()
                vec = np.array([float(b) / 255.0 - 0.5 for b in h[:self.dim]])
                self._vocab[word] = vec
            tokens.append(self._vocab[word])
        return tokens
    
    def embed(self, tokens: List[np.ndarray]) -> np.ndarray:
        """Combine tokens into a single E8 embedding."""
        if not tokens:
            return np.zeros(self.dim)
        # Average pooling with position weighting
        weights = np.exp(-np.arange(len(tokens)) / len(tokens))
        weighted = np.sum([t * w for t, w in zip(tokens, weights)], axis=0)
        return weighted / np.sum(weights)


class GeoTransformer:
    """
    Geometric Transformer.
    
    Transforms data between formats while preserving geometric properties.
    """
    
    def __init__(self):
        self.tokenizer = GeoTokenizer()
    
    def transform(self, data: Any, from_format: str, to_format: str) -> Any:
        """Transform data between formats."""
        # First convert to internal representation
        internal = self._to_internal(data, from_format)
        # Then convert to target format
        return self._from_internal(internal, to_format)
    
    def _to_internal(self, data: Any, fmt: str) -> np.ndarray:
        """Convert any format to internal E8 vector."""
        if fmt == "text":
            tokens = self.tokenizer.tokenize(str(data))
            return self.tokenizer.embed(tokens)
        elif fmt == "json":
            # Flatten JSON to string, then tokenize
            flat = json.dumps(data, sort_keys=True)
            tokens = self.tokenizer.tokenize(flat)
            return self.tokenizer.embed(tokens)
        elif fmt == "vector":
            vec = np.array(data)
            if len(vec) < 8:
                vec = np.pad(vec, (0, 8 - len(vec)))
            return vec[:8]
        elif fmt == "binary":
            # Convert bytes to vector
            if isinstance(data, bytes):
                vec = np.array([float(b) / 255.0 - 0.5 for b in data[:8]])
            else:
                vec = np.zeros(8)
            return vec
        else:
            # Default: try to convert to vector
            return np.array(data)[:8] if hasattr(data, '__iter__') else np.array([float(data)] + [0]*7)
    
    def _from_internal(self, vec: np.ndarray, fmt: str) -> Any:
        """Convert internal E8 vector to target format."""
        if fmt == "vector":
            return vec.tolist()
        elif fmt == "json":
            return {"e8_vector": vec.tolist(), "norm": float(np.linalg.norm(vec))}
        elif fmt == "text":
            # Generate descriptive text
            return f"E8({', '.join(f'{x:.3f}' for x in vec)})"
        elif fmt == "binary":
            return bytes([int((x + 0.5) * 255) % 256 for x in vec])
        else:
            return vec


class IOManager:
    """
    I/O Manager - Main interface for data ingestion and export.
    
    All external data enters the system through here.
    All output leaves through here.
    """
    
    def __init__(self):
        self.transformer = GeoTransformer()
        self.tokenizer = self.transformer.tokenizer
        self.speedlight = get_speedlight()
    
    @receipted("ingest")
    def ingest(self, data: Any, format: str = "auto") -> CQEAtom:
        """
        Atomize any input data into a CQE Atom.
        
        This is the main entry point for all external data.
        """
        # Auto-detect format if needed
        if format == "auto":
            format = self._detect_format(data)
        
        # Transform to E8 vector
        vec = self.transformer._to_internal(data, format)
        
        # Create atom
        atom = CQEAtom.from_vector(vec.tolist(), provenance=f"ingest:{format}")
        
        return atom
    
    @receipted("export")
    def export(self, atom: CQEAtom, format: str = "json") -> Any:
        """
        Convert a CQE Atom to an output format.
        """
        vec = atom.lanes
        return self.transformer._from_internal(vec, format)
    
    def _detect_format(self, data: Any) -> str:
        """Auto-detect the format of input data."""
        if isinstance(data, str):
            # Try to parse as JSON
            try:
                json.loads(data)
                return "json"
            except:
                return "text"
        elif isinstance(data, bytes):
            return "binary"
        elif isinstance(data, dict):
            return "json"
        elif isinstance(data, (list, tuple, np.ndarray)):
            return "vector"
        else:
            return "text"
    
    def batch_ingest(self, items: List[Any], format: str = "auto") -> List[CQEAtom]:
        """Ingest multiple items at once."""
        return [self.ingest(item, format) for item in items]
