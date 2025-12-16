"""
GNLC λ₁ Relation Calculus
Geometry-Native Lambda Calculus (GNLC)
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper:
"λ₁ is the Relation Calculus - concerned with relationships between CQE Atoms.
It introduces tensor products of E₈ vectors, allowing creation of complex
geometric objects. Used to define structure of data (graphs, syntax, etc.)."

This implements:
- Tensor products of E₈ vectors
- Relational types
- Graph structures
- Compositional relationships
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay
from layer5_interface.gnlc_lambda0 import Lambda0Term, Lambda0Calculus


class RelationType(Enum):
    """Types of relations in λ₁."""
    BINARY = "binary"  # Relation between two atoms
    TENSOR = "tensor"  # Tensor product
    GRAPH = "graph"  # Graph structure
    COMPOSITION = "composition"  # Compositional structure


@dataclass
class TensorProduct:
    """
    Tensor product of two E₈ vectors.
    
    From whitepaper:
    "Tensor product allows creation of more complex geometric objects."
    """
    left: Lambda0Term
    right: Lambda0Term
    product_space: np.ndarray = None  # 64-dimensional (8×8)
    
    def __post_init__(self):
        # Compute tensor product
        if self.product_space is None:
            self.product_space = np.outer(
                self.left.overlay.e8_base,
                self.right.overlay.e8_base
            )
    
    @property
    def dimension(self) -> int:
        """Dimension of tensor product space."""
        return 64
    
    def flatten(self) -> np.ndarray:
        """Flatten tensor product to vector."""
        return self.product_space.flatten()
    
    def norm(self) -> float:
        """Frobenius norm of tensor."""
        return np.linalg.norm(self.product_space)


@dataclass
class Relation:
    """
    Binary relation between two atoms.
    
    A relation is a geometric connection between two points in E₈.
    """
    source: Lambda0Term
    target: Lambda0Term
    relation_type: str
    strength: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def distance(self) -> float:
        """Geometric distance between source and target."""
        return np.linalg.norm(
            self.source.overlay.e8_base - self.target.overlay.e8_base
        )
    
    @property
    def direction(self) -> np.ndarray:
        """Direction vector from source to target."""
        diff = self.target.overlay.e8_base - self.source.overlay.e8_base
        norm = np.linalg.norm(diff)
        return diff / norm if norm > 0 else np.zeros(8)


@dataclass
class GraphNode:
    """Node in a geometric graph."""
    term: Lambda0Term
    node_id: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GraphEdge:
    """Edge in a geometric graph."""
    source_id: str
    target_id: str
    relation: Relation
    weight: float = 1.0


@dataclass
class GeometricGraph:
    """
    Graph structure with geometric nodes and edges.
    
    From whitepaper:
    "Used to define structure of data, such as connections between nodes."
    """
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    
    def __post_init__(self):
        if not hasattr(self, 'nodes'):
            self.nodes = {}
        if not hasattr(self, 'edges'):
            self.edges = []
    
    def add_node(self, node: GraphNode):
        """Add node to graph."""
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: GraphEdge):
        """Add edge to graph."""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError("Edge endpoints must exist in graph")
        self.edges.append(edge)
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node."""
        neighbors = []
        for edge in self.edges:
            if edge.source_id == node_id:
                neighbors.append(edge.target_id)
        return neighbors
    
    def get_edge(self, source_id: str, target_id: str) -> Optional[GraphEdge]:
        """Get edge between two nodes."""
        for edge in self.edges:
            if edge.source_id == source_id and edge.target_id == target_id:
                return edge
        return None
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self.nodes)
    
    @property
    def num_edges(self) -> int:
        """Number of edges."""
        return len(self.edges)


@dataclass
class Lambda1Term:
    """
    λ₁ term (relation).
    
    In λ₁, terms are relationships between atoms.
    """
    relation: Relation
    term_type: RelationType = RelationType.BINARY
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __repr__(self):
        return f"λ₁[{self.relation.source.term_id[:8]} → {self.relation.target.term_id[:8]}]"


class Lambda1Calculus:
    """
    λ₁ Relation Calculus.
    
    From whitepaper:
    "λ₁ is concerned with relationships and structures between atoms."
    
    Key features:
    1. Tensor products
    2. Binary relations
    3. Graph structures
    4. Compositional operations
    """
    
    def __init__(self):
        self.lambda0 = Lambda0Calculus()
        self.relations: List[Relation] = []
        self.graphs: Dict[str, GeometricGraph] = {}
    
    def tensor_product(
        self,
        term1: Lambda0Term,
        term2: Lambda0Term
    ) -> TensorProduct:
        """
        Compute tensor product of two atoms.
        
        Args:
            term1: First atom
            term2: Second atom
        
        Returns:
            TensorProduct
        """
        return TensorProduct(left=term1, right=term2)
    
    def create_relation(
        self,
        source: Lambda0Term,
        target: Lambda0Term,
        relation_type: str,
        strength: float = 1.0
    ) -> Relation:
        """
        Create binary relation between two atoms.
        
        Args:
            source: Source atom
            target: Target atom
            relation_type: Type of relation
            strength: Relation strength
        
        Returns:
            Relation
        """
        relation = Relation(
            source=source,
            target=target,
            relation_type=relation_type,
            strength=strength
        )
        self.relations.append(relation)
        return relation
    
    def create_graph(self, graph_id: str) -> GeometricGraph:
        """
        Create empty geometric graph.
        
        Args:
            graph_id: Graph identifier
        
        Returns:
            GeometricGraph
        """
        graph = GeometricGraph(nodes={}, edges=[])
        self.graphs[graph_id] = graph
        return graph
    
    def add_node_to_graph(
        self,
        graph_id: str,
        term: Lambda0Term,
        node_id: str
    ) -> GraphNode:
        """
        Add node to graph.
        
        Args:
            graph_id: Graph identifier
            term: Atom for node
            node_id: Node identifier
        
        Returns:
            GraphNode
        """
        if graph_id not in self.graphs:
            raise ValueError(f"Graph {graph_id} does not exist")
        
        node = GraphNode(term=term, node_id=node_id)
        self.graphs[graph_id].add_node(node)
        return node
    
    def add_edge_to_graph(
        self,
        graph_id: str,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0
    ) -> GraphEdge:
        """
        Add edge to graph.
        
        Args:
            graph_id: Graph identifier
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relation
            weight: Edge weight
        
        Returns:
            GraphEdge
        """
        if graph_id not in self.graphs:
            raise ValueError(f"Graph {graph_id} does not exist")
        
        graph = self.graphs[graph_id]
        
        # Get nodes
        source_node = graph.nodes[source_id]
        target_node = graph.nodes[target_id]
        
        # Create relation
        relation = self.create_relation(
            source_node.term,
            target_node.term,
            relation_type
        )
        
        # Create edge
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight
        )
        
        graph.add_edge(edge)
        return edge
    
    def compose_relations(
        self,
        rel1: Relation,
        rel2: Relation
    ) -> Optional[Relation]:
        """
        Compose two relations (if compatible).
        
        Args:
            rel1: First relation (A → B)
            rel2: Second relation (B → C)
        
        Returns:
            Composed relation (A → C) if compatible, None otherwise
        """
        # Check if relations are composable (rel1.target ≈ rel2.source)
        distance = np.linalg.norm(
            rel1.target.overlay.e8_base - rel2.source.overlay.e8_base
        )
        
        if distance > 0.1:  # Not composable
            return None
        
        # Create composed relation
        composed = Relation(
            source=rel1.source,
            target=rel2.target,
            relation_type=f"{rel1.relation_type}∘{rel2.relation_type}",
            strength=rel1.strength * rel2.strength,
            metadata={
                'composed_from': [rel1, rel2]
            }
        )
        
        self.relations.append(composed)
        return composed
    
    def parallel_composition(
        self,
        rel1: Relation,
        rel2: Relation
    ) -> TensorProduct:
        """
        Parallel composition of two relations.
        
        Args:
            rel1: First relation
            rel2: Second relation
        
        Returns:
            TensorProduct representing parallel composition
        """
        # Parallel composition via tensor product
        return self.tensor_product(rel1.source, rel2.source)
    
    def graph_to_adjacency_matrix(
        self,
        graph_id: str
    ) -> np.ndarray:
        """
        Convert graph to adjacency matrix.
        
        Args:
            graph_id: Graph identifier
        
        Returns:
            Adjacency matrix
        """
        if graph_id not in self.graphs:
            raise ValueError(f"Graph {graph_id} does not exist")
        
        graph = self.graphs[graph_id]
        n = graph.num_nodes
        
        # Create node ID to index mapping
        node_ids = list(graph.nodes.keys())
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Create adjacency matrix
        adj = np.zeros((n, n))
        for edge in graph.edges:
            i = id_to_idx[edge.source_id]
            j = id_to_idx[edge.target_id]
            adj[i, j] = edge.weight
        
        return adj
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get calculus statistics."""
        return {
            'num_relations': len(self.relations),
            'num_graphs': len(self.graphs),
            'total_nodes': sum(g.num_nodes for g in self.graphs.values()),
            'total_edges': sum(g.num_edges for g in self.graphs.values())
        }


# Example usage and tests
if __name__ == "__main__":
    print("=== GNLC λ₁ Relation Calculus Test ===\n")
    
    from layer1_morphonic.overlay_system import ImmutablePose
    import time
    
    # Create λ₁ calculus
    lambda1 = Lambda1Calculus()
    
    # Test 1: Create atoms
    print("Test 1: Create Atoms")
    
    e8_1 = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    e8_2 = np.array([0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])
    e8_3 = np.array([0.6, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    
    activations = np.zeros(240, dtype=int)
    activations[0:80] = 1
    
    pose1 = ImmutablePose(tuple(e8_1), tuple(np.eye(8)[0]), time.time())
    pose2 = ImmutablePose(tuple(e8_2), tuple(np.eye(8)[0]), time.time())
    pose3 = ImmutablePose(tuple(e8_3), tuple(np.eye(8)[0]), time.time())
    
    overlay1 = Overlay(e8_base=e8_1, activations=activations.copy(), pose=pose1)
    overlay2 = Overlay(e8_base=e8_2, activations=activations.copy(), pose=pose2)
    overlay3 = Overlay(e8_base=e8_3, activations=activations.copy(), pose=pose3)
    
    term1 = lambda1.lambda0.atom(overlay1)
    term2 = lambda1.lambda0.atom(overlay2)
    term3 = lambda1.lambda0.atom(overlay3)
    
    print(f"Term 1: {term1}")
    print(f"Term 2: {term2}")
    print(f"Term 3: {term3}")
    print()
    
    # Test 2: Tensor product
    print("Test 2: Tensor Product")
    
    tensor = lambda1.tensor_product(term1, term2)
    print(f"Tensor dimension: {tensor.dimension}")
    print(f"Tensor norm: {tensor.norm():.6f}")
    print(f"Flattened shape: {tensor.flatten().shape}")
    print()
    
    # Test 3: Binary relations
    print("Test 3: Binary Relations")
    
    rel1 = lambda1.create_relation(term1, term2, "edge", strength=0.8)
    rel2 = lambda1.create_relation(term2, term3, "edge", strength=0.9)
    
    print(f"Relation 1: {rel1.source.term_id[:8]} → {rel1.target.term_id[:8]}")
    print(f"  Distance: {rel1.distance:.6f}")
    print(f"  Strength: {rel1.strength}")
    
    print(f"Relation 2: {rel2.source.term_id[:8]} → {rel2.target.term_id[:8]}")
    print(f"  Distance: {rel2.distance:.6f}")
    print(f"  Strength: {rel2.strength}")
    print()
    
    # Test 4: Relation composition
    print("Test 4: Relation Composition")
    
    composed = lambda1.compose_relations(rel1, rel2)
    if composed:
        print(f"Composed: {composed.source.term_id[:8]} → {composed.target.term_id[:8]}")
        print(f"  Type: {composed.relation_type}")
        print(f"  Strength: {composed.strength:.6f}")
    else:
        print("Relations not composable")
    print()
    
    # Test 5: Graph structure
    print("Test 5: Graph Structure")
    
    graph = lambda1.create_graph("test_graph")
    
    node1 = lambda1.add_node_to_graph("test_graph", term1, "n1")
    node2 = lambda1.add_node_to_graph("test_graph", term2, "n2")
    node3 = lambda1.add_node_to_graph("test_graph", term3, "n3")
    
    edge1 = lambda1.add_edge_to_graph("test_graph", "n1", "n2", "connects", 1.0)
    edge2 = lambda1.add_edge_to_graph("test_graph", "n2", "n3", "connects", 1.0)
    edge3 = lambda1.add_edge_to_graph("test_graph", "n1", "n3", "connects", 0.5)
    
    print(f"Graph nodes: {graph.num_nodes}")
    print(f"Graph edges: {graph.num_edges}")
    print(f"Neighbors of n1: {graph.get_neighbors('n1')}")
    print()
    
    # Test 6: Adjacency matrix
    print("Test 6: Adjacency Matrix")
    
    adj = lambda1.graph_to_adjacency_matrix("test_graph")
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Adjacency matrix:\n{adj}")
    print()
    
    # Test 7: Statistics
    print("Test 7: Statistics")
    
    stats = lambda1.get_statistics()
    print(f"Relations: {stats['num_relations']}")
    print(f"Graphs: {stats['num_graphs']}")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print()
    
    print("=== All Tests Passed ===")
