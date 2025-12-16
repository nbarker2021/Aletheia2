"""
AGRM/MDHG - Adaptive Geometric Retrieval Module / Multi-Dimensional Hierarchical Grouping

Provides hierarchical clustering and retrieval for CQE atoms.
Used for efficient geometric search and organization.
"""

import numpy as np
import hashlib
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

from spine.kernel import CQEAtom
from spine.speedlight import get_speedlight, receipted


@dataclass
class MDHGNode:
    """Node in the MDHG hierarchical tree."""
    id: str
    centroid: np.ndarray
    radius: float
    children: List['MDHGNode'] = field(default_factory=list)
    atom_ids: List[str] = field(default_factory=list)
    level: int = 0
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def size(self) -> int:
        if self.is_leaf():
            return len(self.atom_ids)
        return sum(c.size() for c in self.children)


@dataclass
class RetrievalResult:
    """Result of a retrieval query."""
    atom_id: str
    distance: float
    path: List[str]  # Path through MDHG tree


class AGRMEngine:
    """
    AGRM Engine - Adaptive Geometric Retrieval.
    
    Provides:
    - Geometric similarity search
    - Adaptive radius queries
    - K-nearest neighbor retrieval
    """
    
    def __init__(self):
        self.speedlight = get_speedlight()
        self.atoms: Dict[str, CQEAtom] = {}
    
    def add(self, atom: CQEAtom) -> str:
        """Add an atom to the retrieval index."""
        aid = hashlib.sha256(json.dumps(atom.to_dict(), sort_keys=True).encode()).hexdigest()[:16]
        self.atoms[aid] = atom
        return aid
    
    @receipted("agrm_query")
    def query(self, query_atom: CQEAtom, k: int = 5) -> List[RetrievalResult]:
        """Find k nearest atoms to query."""
        if not self.atoms:
            return []
        
        results = []
        for aid, atom in self.atoms.items():
            dist = float(np.linalg.norm(query_atom.lanes - atom.lanes))
            results.append(RetrievalResult(
                atom_id=aid,
                distance=dist,
                path=["root"]
            ))
        
        results.sort(key=lambda r: r.distance)
        return results[:k]
    
    @receipted("agrm_radius_query")
    def radius_query(self, query_atom: CQEAtom, radius: float) -> List[RetrievalResult]:
        """Find all atoms within radius of query."""
        results = []
        for aid, atom in self.atoms.items():
            dist = float(np.linalg.norm(query_atom.lanes - atom.lanes))
            if dist <= radius:
                results.append(RetrievalResult(
                    atom_id=aid,
                    distance=dist,
                    path=["root"]
                ))
        
        results.sort(key=lambda r: r.distance)
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get AGRM engine status."""
        return {
            "atom_count": len(self.atoms),
            "active": True
        }


class MDHGEngine:
    """
    MDHG Engine - Multi-Dimensional Hierarchical Grouping.
    
    Provides:
    - Hierarchical clustering of atoms
    - Efficient tree-based retrieval
    - Cluster management
    """
    
    def __init__(self, max_leaf_size: int = 10, branching_factor: int = 4):
        self.speedlight = get_speedlight()
        self.max_leaf_size = max_leaf_size
        self.branching_factor = branching_factor
        self.root: Optional[MDHGNode] = None
        self.atoms: Dict[str, CQEAtom] = {}
    
    def add(self, atom: CQEAtom) -> str:
        """Add an atom to the MDHG tree."""
        aid = hashlib.sha256(json.dumps(atom.to_dict(), sort_keys=True).encode()).hexdigest()[:16]
        self.atoms[aid] = atom
        
        # Invalidate tree (needs rebuild)
        self.root = None
        
        return aid
    
    @receipted("mdhg_build")
    def build(self) -> MDHGNode:
        """Build the MDHG tree from all atoms."""
        if not self.atoms:
            self.root = MDHGNode(
                id="root",
                centroid=np.zeros(8),
                radius=0.0,
                level=0
            )
            return self.root
        
        # Get all atom IDs and vectors
        atom_ids = list(self.atoms.keys())
        vectors = np.array([self.atoms[aid].lanes for aid in atom_ids])
        
        # Build tree recursively
        self.root = self._build_node(atom_ids, vectors, level=0)
        return self.root
    
    def _build_node(self, atom_ids: List[str], vectors: np.ndarray, level: int) -> MDHGNode:
        """Recursively build a tree node."""
        centroid = np.mean(vectors, axis=0)
        radius = float(np.max(np.linalg.norm(vectors - centroid, axis=1)))
        
        node = MDHGNode(
            id=f"node_L{level}_{len(atom_ids)}",
            centroid=centroid,
            radius=radius,
            level=level
        )
        
        if len(atom_ids) <= self.max_leaf_size:
            # Leaf node
            node.atom_ids = atom_ids
        else:
            # Split into children using k-means-like approach
            # Simple split by sign of first principal component
            pca_dir = vectors[0] - centroid
            if np.linalg.norm(pca_dir) < 1e-10:
                pca_dir = np.ones(8)
            pca_dir = pca_dir / np.linalg.norm(pca_dir)
            
            projections = vectors @ pca_dir
            median = np.median(projections)
            
            left_mask = projections <= median
            right_mask = ~left_mask
            
            if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                left_ids = [aid for aid, m in zip(atom_ids, left_mask) if m]
                right_ids = [aid for aid, m in zip(atom_ids, right_mask) if m]
                
                left_vecs = vectors[left_mask]
                right_vecs = vectors[right_mask]
                
                node.children = [
                    self._build_node(left_ids, left_vecs, level + 1),
                    self._build_node(right_ids, right_vecs, level + 1)
                ]
            else:
                # Can't split, make leaf
                node.atom_ids = atom_ids
        
        return node
    
    @receipted("mdhg_query")
    def query(self, query_atom: CQEAtom, k: int = 5) -> List[RetrievalResult]:
        """Query the MDHG tree for k nearest atoms."""
        if self.root is None:
            self.build()
        
        # Simple traversal (could be optimized with priority queue)
        results = []
        self._traverse(self.root, query_atom, results, [])
        
        results.sort(key=lambda r: r.distance)
        return results[:k]
    
    def _traverse(self, node: MDHGNode, query: CQEAtom, results: List[RetrievalResult], path: List[str]):
        """Traverse tree to find nearest atoms."""
        current_path = path + [node.id]
        
        if node.is_leaf():
            for aid in node.atom_ids:
                atom = self.atoms[aid]
                dist = float(np.linalg.norm(query.lanes - atom.lanes))
                results.append(RetrievalResult(
                    atom_id=aid,
                    distance=dist,
                    path=current_path
                ))
        else:
            # Sort children by distance to centroid
            child_dists = [(c, np.linalg.norm(query.lanes - c.centroid)) for c in node.children]
            child_dists.sort(key=lambda x: x[1])
            
            for child, _ in child_dists:
                self._traverse(child, query, results, current_path)
    
    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the MDHG tree."""
        if self.root is None:
            return {"built": False}
        
        def count_nodes(node):
            if node.is_leaf():
                return 1, 1, node.level
            total = 1
            leaves = 0
            max_depth = node.level
            for c in node.children:
                t, l, d = count_nodes(c)
                total += t
                leaves += l
                max_depth = max(max_depth, d)
            return total, leaves, max_depth
        
        total, leaves, depth = count_nodes(self.root)
        return {
            "built": True,
            "total_nodes": total,
            "leaf_nodes": leaves,
            "max_depth": depth,
            "atom_count": len(self.atoms)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get MDHG engine status."""
        return {
            "atom_count": len(self.atoms),
            "tree_built": self.root is not None,
            "active": True
        }


# Global instances
_agrm: AGRMEngine = None
_mdhg: MDHGEngine = None

def get_agrm() -> AGRMEngine:
    """Get the global AGRM engine."""
    global _agrm
    if _agrm is None:
        _agrm = AGRMEngine()
    return _agrm

def get_mdhg() -> MDHGEngine:
    """Get the global MDHG engine."""
    global _mdhg
    if _mdhg is None:
        _mdhg = MDHGEngine()
    return _mdhg
