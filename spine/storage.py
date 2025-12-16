"""
Storage Manager - Atom Persistence

Persists atoms with geometric indexing.
Supports MDHG hierarchical clustering for efficient retrieval.
"""

import json
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from spine.kernel import CQEAtom
from spine.speedlight import get_speedlight, receipted


def atom_id(atom: CQEAtom) -> str:
    """Generate a unique ID for an atom."""
    data = json.dumps(atom.to_dict(), sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class MDHGNode:
    """Node in the MDHG hierarchical clustering tree."""
    id: str
    centroid: np.ndarray
    children: List['MDHGNode']
    atoms: List[str]  # Atom IDs at this node
    level: int
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class StorageManager:
    """
    Storage Manager - Atom persistence with geometric indexing.
    
    Features:
    - In-memory atom storage
    - Geometric similarity search
    - MDHG hierarchical clustering
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        self.atoms: Dict[str, CQEAtom] = {}
        self.persist_path = persist_path
        self.speedlight = get_speedlight()
        self._mdhg_root: Optional[MDHGNode] = None
    
    @receipted("store")
    def store(self, atom: CQEAtom) -> str:
        """
        Persist an atom and return its ID.
        """
        aid = atom_id(atom)
        self.atoms[aid] = atom
        
        # Invalidate MDHG tree (needs rebuild)
        self._mdhg_root = None
        
        return aid
    
    @receipted("retrieve")
    def retrieve(self, aid: str) -> Optional[CQEAtom]:
        """
        Get an atom by ID.
        """
        return self.atoms.get(aid)
    
    def query(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find k nearest atoms to the query vector.
        
        Returns list of (atom_id, distance) tuples.
        """
        if not self.atoms:
            return []
        
        distances = []
        for aid, atom in self.atoms.items():
            dist = np.linalg.norm(atom.lanes - query_vector)
            distances.append((aid, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def query_by_atom(self, query_atom: CQEAtom, k: int = 5) -> List[Tuple[str, float]]:
        """Find k nearest atoms to a query atom."""
        return self.query(query_atom.lanes, k)
    
    def cluster(self, n_clusters: int = 8) -> MDHGNode:
        """
        Build MDHG hierarchical clustering tree.
        
        Uses a simple k-means-like approach for demonstration.
        """
        if not self.atoms:
            return MDHGNode(
                id="root",
                centroid=np.zeros(8),
                children=[],
                atoms=[],
                level=0
            )
        
        # Get all atom vectors
        atom_ids = list(self.atoms.keys())
        vectors = np.array([self.atoms[aid].lanes for aid in atom_ids])
        
        # Simple clustering: divide by sign of first component
        positive = [aid for aid, vec in zip(atom_ids, vectors) if vec[0] >= 0]
        negative = [aid for aid, vec in zip(atom_ids, vectors) if vec[0] < 0]
        
        # Create child nodes
        children = []
        if positive:
            pos_vecs = np.array([self.atoms[aid].lanes for aid in positive])
            children.append(MDHGNode(
                id="positive",
                centroid=np.mean(pos_vecs, axis=0),
                children=[],
                atoms=positive,
                level=1
            ))
        if negative:
            neg_vecs = np.array([self.atoms[aid].lanes for aid in negative])
            children.append(MDHGNode(
                id="negative",
                centroid=np.mean(neg_vecs, axis=0),
                children=[],
                atoms=negative,
                level=1
            ))
        
        # Create root
        self._mdhg_root = MDHGNode(
            id="root",
            centroid=np.mean(vectors, axis=0),
            children=children,
            atoms=atom_ids,
            level=0
        )
        
        return self._mdhg_root
    
    def save(self, path: Optional[str] = None):
        """Save all atoms to disk."""
        path = path or self.persist_path
        if not path:
            return
        
        data = {
            aid: atom.to_dict() for aid, atom in self.atoms.items()
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: Optional[str] = None):
        """Load atoms from disk."""
        path = path or self.persist_path
        if not path:
            return
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            for aid, atom_dict in data.items():
                self.atoms[aid] = CQEAtom(
                    lanes=np.array(atom_dict['lanes']),
                    parity=np.array(atom_dict['parity']),
                    metadata=atom_dict.get('metadata', {}),
                    provenance=atom_dict.get('provenance', '')
                )
        except FileNotFoundError:
            pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get storage manager status."""
        return {
            "atom_count": len(self.atoms),
            "has_mdhg_tree": self._mdhg_root is not None,
            "persist_path": self.persist_path
        }
