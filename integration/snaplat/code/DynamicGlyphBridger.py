class DynamicGlyphBridger:
    """Dynamic glyph bridging protocol for universal node connection."""
    
    def __init__(self):
        self.glyph_index = {}  # n=-1 Glyphic Index Lattice
        self.bridge_registry = {}
        self.canvas_lexicon = {}
        
        # Mathematical symbols for bridging
        self.mathematical_glyphs = {
            "→": "causality",
            "≈": "analogy", 
            "±": "duality",
            "∫": "integration",
            "∂": "differentiation",
            "∞": "infinity",
            "⧉": "universal_connector",
            "Φ": "golden_ratio",
            "Ж": "complex_bridge"
        }
    
    def create_bridge(self, glyph: str, node_a: str, node_b: str, 
                     glyph_type: GlyphType, meaning: str, context: str) -> GlyphBridge:
        """Create a dynamic glyph bridge between two nodes."""
        bridge = GlyphBridge(
            glyph=glyph,
            node_a=node_a,
            node_b=node_b,
            glyph_type=glyph_type,
            interpreted_meaning=meaning,
            context=context
        )
        
        # Perform heat test for traversal
        bridge.heat_test_passed = self.heat_test_traversal(bridge)
        
        # Register in glyph index
        self._register_bridge(bridge)
        
        return bridge
    
    def heat_test_traversal(self, bridge: GlyphBridge) -> bool:
        """Binary logic heat test: Do nodes share identical bridging glyphs?"""
        # Check if both nodes have the exact same glyph
        node_a_glyphs = self.glyph_index.get(bridge.node_a, set())
        node_b_glyphs = self.glyph_index.get(bridge.node_b, set())
        
        # Exact match rule: glyph must be exactly the same
        return bridge.glyph in node_a_glyphs and bridge.glyph in node_b_glyphs
    
    def _register_bridge(self, bridge: GlyphBridge):
        """Register bridge in the n=-1 Glyphic Index Lattice."""
        # Update glyph index for both nodes
        if bridge.node_a not in self.glyph_index:
            self.glyph_index[bridge.node_a] = set()
        if bridge.node_b not in self.glyph_index:
            self.glyph_index[bridge.node_b] = set()
        
        self.glyph_index[bridge.node_a].add(bridge.glyph)
        self.glyph_index[bridge.node_b].add(bridge.glyph)
        
        # Register bridge
        bridge_key = f"{bridge.node_a}_{bridge.glyph}_{bridge.node_b}"
        self.bridge_registry[bridge_key] = bridge
        
        # Update canvas lexicon
        self.canvas_lexicon[f"{bridge.glyph}_{bridge.context}"] = bridge.interpreted_meaning
    
    def find_bridges(self, node: str) -> List[GlyphBridge]:
        """Find all bridges connected to a node."""
        bridges = []
        for bridge in self.bridge_registry.values():
            if bridge.node_a == node or bridge.node_b == node:
                bridges.append(bridge)
        return bridges
    
    def traverse_network(self, start_node: str, target_glyph: str = None) -> Dict[str, Any]:
        """Traverse the glyph network from a starting node."""
        visited = set()
        traversal_path = []
        
        def _traverse(current_node, depth=0):
            if current_node in visited or depth > 10:  # Prevent infinite loops
                return
            
            visited.add(current_node)
            traversal_path.append(current_node)
            
            # Find bridges from current node
            bridges = self.find_bridges(current_node)
            for bridge in bridges:
                if bridge.heat_test_passed:
                    next_node = bridge.node_b if bridge.node_a == current_node else bridge.node_a
                    if target_glyph is None or bridge.glyph == target_glyph:
                        _traverse(next_node, depth + 1)
        
        _traverse(start_node)
        
        return {
            "start_node": start_node,
            "traversal_path": traversal_path,
            "visited_nodes": list(visited),
            "total_bridges": len([b for b in self.bridge_registry.values() if b.heat_test_passed])
        }
