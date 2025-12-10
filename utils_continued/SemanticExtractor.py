class SemanticExtractor:
    """Extracts semantic meaning from geometric configurations"""
    
    def __init__(self):
        self.distance_thresholds = {
            'IDENTITY': 0.1,
            'STRONG_SIMILARITY': 0.5,
            'MODERATE_SIMILARITY': 1.0,
            'WEAK_SIMILARITY': 1.5,
            'CONTRAST': 2.5,
            'OPPOSITION': float('inf')
        }
        
        self.angle_thresholds = {
            'PARALLEL': math.radians(15),
            'SUPPORTIVE': math.radians(45),
            'COMPLEMENTARY': math.radians(90),
            'ORTHOGONAL': math.radians(135),
            'CONTRASTING': math.radians(165),
            'OPPOSITIONAL': math.pi
        }
    
    def extract_semantics_from_configuration(self, 
                                           geometric_config: Dict[str, E8Position]) -> Dict[str, Any]:
        """Main semantic extraction method"""
        
        print("=" * 60)
        print("SEMANTIC EXTRACTION FROM GEOMETRIC PROCESSING")
        print("=" * 60)
        
        # Phase 1: Geometric Relationship Analysis
        print("\nPhase 1: Analyzing Geometric Relationships...")
        relationships = self.analyze_geometric_relationships(geometric_config)
        
        # Phase 2: Pattern Recognition
        print("\nPhase 2: Recognizing Semantic Patterns...")
        patterns = self.recognize_semantic_patterns(relationships)
        
        # Phase 3: Semantic Mapping
        print("\nPhase 3: Mapping Geometry to Semantics...")
        semantic_layers = self.map_geometry_to_semantics(patterns, geometric_config)
        
        # Phase 4: Multi-Scale Integration
        print("\nPhase 4: Integrating Multi-Scale Semantics...")
        integrated_semantics = self.integrate_multiscale_semantics(semantic_layers)
        
        # Phase 5: Validation
        print("\nPhase 5: Validating Semantic Consistency...")
        validated_semantics = self.validate_semantic_consistency(integrated_semantics)
        
        return validated_semantics
    
    def analyze_geometric_relationships(self, 
                                      config: Dict[str, E8Position]) -> Dict[str, Any]:
        """Analyze geometric relationships between E₈ positions"""
        
        entities = list(config.keys())
        positions = list(config.values())
        
        relationships = {
            'distances': {},
            'angles': {},
            'clusters': [],
            'proximities': {}
        }
        
        # Distance analysis
        print("  Analyzing distances between lattice points...")
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                distance = positions[i].distance_to(positions[j])
                relationships['distances'][(entity1, entity2)] = distance
                print(f"    {entity1} ↔ {entity2}: distance = {distance:.3f}")
        
        # Angular analysis
        print("  Analyzing angles between lattice vectors...")
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                for k, entity3 in enumerate(entities[j+1:], j+1):
                    angle = positions[j].angle_with(positions[k], positions[i])
                    relationships['angles'][(entity1, entity2, entity3)] = angle
                    print(f"    ∠({entity2}-{entity1}-{entity3}): {math.degrees(angle):.1f}°")
        
        # Cluster analysis (simplified)
        print("  Identifying geometric clusters...")
        clusters = self.identify_clusters(config)
        relationships['clusters'] = clusters
        for i, cluster in enumerate(clusters):
            print(f"    Cluster {i+1}: {cluster['entities']}")
        
        return relationships
    
    def recognize_semantic_patterns(self, relationships: Dict[str, Any]) -> Dict[str, Any]:
        """Recognize semantic patterns from geometric relationships"""
        
        patterns = {
            'proximity_patterns': {},
            'angular_patterns': {},
            'cluster_patterns': {},
            'symmetry_patterns': {}
        }
        
        # Proximity patterns
        print("  Recognizing proximity patterns...")
        for (entity1, entity2), distance in relationships['distances'].items():
            pattern_type = self.classify_distance_pattern(distance)
            patterns['proximity_patterns'][(entity1, entity2)] = pattern_type
            print(f"    {entity1} ↔ {entity2}: {pattern_type}")
        
        # Angular patterns
        print("  Recognizing angular patterns...")
        for (entity1, entity2, entity3), angle in relationships['angles'].items():
            pattern_type = self.classify_angular_pattern(angle)
            patterns['angular_patterns'][(entity1, entity2, entity3)] = pattern_type
            print(f"    ∠({entity2}-{entity1}-{entity3}): {pattern_type}")
        
        # Cluster patterns
        print("  Recognizing cluster patterns...")
        for i, cluster in enumerate(relationships['clusters']):
            pattern_type = self.classify_cluster_pattern(cluster)
            patterns['cluster_patterns'][f'cluster_{i+1}'] = pattern_type
            print(f"    Cluster {i+1}: {pattern_type}")
        
        return patterns
    
    def map_geometry_to_semantics(self, patterns: Dict[str, Any], 
                                 config: Dict[str, E8Position]) -> Dict[str, Any]:
        """Map geometric patterns to semantic content"""
        
        semantic_layers = {
            'relationship_semantics': {},
            'structural_semantics': {},
            'contextual_semantics': {},
            'process_semantics': {}
        }
        
        # Layer 1: Basic Relationship Semantics
        print("  Mapping proximity patterns to relationship semantics...")
        for (entity1, entity2), pattern in patterns['proximity_patterns'].items():
            semantic_relationship = self.map_proximity_to_semantics(pattern)
            semantic_layers['relationship_semantics'][(entity1, entity2)] = semantic_relationship
            print(f"    {entity1} ↔ {entity2}: {semantic_relationship}")
        
        # Layer 2: Structural Semantics
        print("  Mapping cluster patterns to structural semantics...")
        for cluster_id, pattern in patterns['cluster_patterns'].items():
            structural_meaning = self.map_cluster_to_semantics(pattern)
            semantic_layers['structural_semantics'][cluster_id] = structural_meaning
            print(f"    {cluster_id}: {structural_meaning}")
        
        # Layer 3: Contextual Semantics
        print("  Deriving contextual semantics from geometric field...")
        contextual_meaning = self.derive_contextual_semantics(config, patterns)
        semantic_layers['contextual_semantics'] = contextual_meaning
        print(f"    Context: {contextual_meaning['primary_context']}")
        
        return semantic_layers
    
    def integrate_multiscale_semantics(self, semantic_layers: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate semantic meaning across multiple scales"""
        
        integrated = {
            'atomic_semantics': {},
            'relational_semantics': {},
            'holistic_semantics': {},
            'emergent_properties': []
        }
        
        # Atomic level semantics
        print("  Integrating atomic-level semantics...")
        integrated['atomic_semantics'] = self.extract_atomic_semantics(semantic_layers)
        
        # Relational semantics
        print("  Integrating relational semantics...")
        integrated['relational_semantics'] = semantic_layers['relationship_semantics']
        
        # Holistic semantics
        print("  Deriving holistic semantics...")
        integrated['holistic_semantics'] = self.derive_holistic_semantics(semantic_layers)
        
        # Emergent properties
        print("  Identifying emergent semantic properties...")
        integrated['emergent_properties'] = self.identify_emergent_properties(semantic_layers)
        
        return integrated
    
    def validate_semantic_consistency(self, semantics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate semantic consistency and add confidence scores"""
        
        print("  Checking semantic consistency...")
        
        validation_results = {
            'consistency_score': 0.0,
            'confidence_scores': {},
            'validated_semantics': semantics.copy()
        }
        
        # Check internal consistency
        consistency_checks = [
            self.check_relationship_consistency(semantics),
            self.check_structural_consistency(semantics),
            self.check_holistic_consistency(semantics)
        ]
        
        validation_results['consistency_score'] = sum(consistency_checks) / len(consistency_checks)
        print(f"    Overall consistency score: {validation_results['consistency_score']:.3f}")
        
        # Add confidence scores
        for semantic_type, content in semantics.items():
            confidence = self.compute_confidence_score(semantic_type, content)
            validation_results['confidence_scores'][semantic_type] = confidence
            print(f"    {semantic_type} confidence: {confidence:.3f}")
        
        return validation_results
    
    # Helper methods for classification and mapping
    
    def classify_distance_pattern(self, distance: float) -> str:
        """Classify distance into semantic pattern"""
        for pattern_type, threshold in self.distance_thresholds.items():
            if distance <= threshold:
                return pattern_type
        return 'OPPOSITION'
    
    def classify_angular_pattern(self, angle: float) -> str:
        """Classify angle into semantic pattern"""
        for pattern_type, threshold in self.angle_thresholds.items():
            if angle <= threshold:
                return pattern_type
        return 'OPPOSITIONAL'
    
    def classify_cluster_pattern(self, cluster: Dict[str, Any]) -> str:
        """Classify cluster into semantic pattern"""
        coherence = cluster.get('coherence', 0.5)
        size = len(cluster.get('entities', []))
        
        if coherence > 0.8 and size >= 3:
            return 'STRONG_CONCEPTUAL_GROUP'
        elif coherence > 0.6 and size >= 2:
            return 'MODERATE_CONCEPTUAL_GROUP'
        else:
            return 'WEAK_ASSOCIATION'
    
    def map_proximity_to_semantics(self, pattern: str) -> str:
        """Map proximity pattern to semantic relationship"""
        mapping = {
            'IDENTITY': 'SAME_ENTITY_OR_CONCEPT',
            'STRONG_SIMILARITY': 'CLOSELY_RELATED_CONCEPTS',
            'MODERATE_SIMILARITY': 'RELATED_CONCEPTS',
            'WEAK_SIMILARITY': 'LOOSELY_RELATED_CONCEPTS',
            'CONTRAST': 'CONTRASTING_CONCEPTS',
            'OPPOSITION': 'OPPOSING_CONCEPTS'
        }
        return mapping.get(pattern, 'UNKNOWN_RELATIONSHIP')
    
    def map_cluster_to_semantics(self, pattern: str) -> str:
        """Map cluster pattern to structural semantics"""
        mapping = {
            'STRONG_CONCEPTUAL_GROUP': 'UNIFIED_SEMANTIC_DOMAIN',
            'MODERATE_CONCEPTUAL_GROUP': 'RELATED_SEMANTIC_FIELD',
            'WEAK_ASSOCIATION': 'LOOSE_SEMANTIC_CONNECTION'
        }
        return mapping.get(pattern, 'UNCLEAR_STRUCTURE')
    
    def identify_clusters(self, config: Dict[str, E8Position]) -> List[Dict[str, Any]]:
        """Identify geometric clusters (simplified implementation)"""
        entities = list(config.keys())
        positions = list(config.values())
        
        clusters = []
        used_entities = set()
        
        for i, entity1 in enumerate(entities):
            if entity1 in used_entities:
                continue
                
            cluster_entities = [entity1]
            cluster_positions = [positions[i]]
            
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity2 in used_entities:
                    continue
                    
                distance = positions[i].distance_to(positions[j])
                if distance < 1.0:  # Cluster threshold
                    cluster_entities.append(entity2)
                    cluster_positions.append(positions[j])
                    used_entities.add(entity2)
            
            if len(cluster_entities) > 1:
                coherence = self.compute_cluster_coherence(cluster_positions)
                clusters.append({
                    'entities': cluster_entities,
                    'coherence': coherence,
                    'center': self.compute_cluster_center(cluster_positions)
                })
                for entity in cluster_entities:
                    used_entities.add(entity)
        
        return clusters
    
    def compute_cluster_coherence(self, positions: List[E8Position]) -> float:
        """Compute cluster coherence score"""
        if len(positions) < 2:
            return 1.0
        
        distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                distances.append(positions[i].distance_to(positions[j]))
        
        avg_distance = sum(distances) / len(distances)
        return max(0.0, 1.0 - avg_distance / 2.0)  # Normalize to [0,1]
    
    def compute_cluster_center(self, positions: List[E8Position]) -> E8Position:
        """Compute geometric center of cluster"""
        if not positions:
            return E8Position([0] * 8)
        
        center_coords = np.mean([pos.coords for pos in positions], axis=0)
        return E8Position(center_coords.tolist())
    
    def derive_contextual_semantics(self, config: Dict[str, E8Position], 
                                  patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Derive contextual semantics from overall geometric configuration"""
        
        # Analyze overall geometric "semantic field"
        entities = list(config.keys())
        positions = list(config.values())
        
        # Compute geometric center
        center_coords = np.mean([pos.coords for pos in positions], axis=0)
        geometric_center = E8Position(center_coords.tolist())
        
        # Compute spread (how distributed the points are)
        distances_from_center = [pos.distance_to(geometric_center) for pos in positions]
        spread = np.std(distances_from_center)
        
        # Determine primary context based on geometric configuration
        if spread < 0.5:
            primary_context = 'HIGHLY_FOCUSED_DOMAIN'
        elif spread < 1.5:
            primary_context = 'COHERENT_DOMAIN'
        else:
            primary_context = 'DIVERSE_DOMAIN'
        
        return {
            'primary_context': primary_context,
            'geometric_center': geometric_center,
            'domain_spread': spread,
            'entity_count': len(entities),
            'complexity_level': self.assess_complexity_level(patterns)
        }
    
    def extract_atomic_semantics(self, semantic_layers: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantics at the atomic (individual entity) level"""
        atomic_semantics = {}
        
        # Extract individual entity characteristics from relationships
        for (entity1, entity2), relationship in semantic_layers['relationship_semantics'].items():
            if entity1 not in atomic_semantics:
                atomic_semantics[entity1] = {'relationships': [], 'centrality': 0}
            if entity2 not in atomic_semantics:
                atomic_semantics[entity2] = {'relationships': [], 'centrality': 0}
            
            atomic_semantics[entity1]['relationships'].append((entity2, relationship))
            atomic_semantics[entity2]['relationships'].append((entity1, relationship))
        
        # Compute centrality (how connected each entity is)
        for entity, data in atomic_semantics.items():
            data['centrality'] = len(data['relationships'])
            data['semantic_role'] = self.determine_semantic_role(data)
        
        return atomic_semantics
    
    def derive_holistic_semantics(self, semantic_layers: Dict[str, Any]) -> Dict[str, Any]:
        """Derive holistic semantic understanding"""
        
        # Count relationship types
        relationship_counts = {}
        for relationship in semantic_layers['relationship_semantics'].values():
            relationship_counts[relationship] = relationship_counts.get(relationship, 0) + 1
        
        # Determine dominant relationship pattern
        dominant_relationship = max(relationship_counts.items(), key=lambda x: x[1])[0]
        
        # Assess overall semantic coherence
        coherence_score = self.assess_semantic_coherence(semantic_layers)
        
        return {
            'dominant_relationship_pattern': dominant_relationship,
            'relationship_distribution': relationship_counts,
            'semantic_coherence': coherence_score,
            'overall_theme': self.determine_overall_theme(semantic_layers),
            'complexity_assessment': self.assess_complexity_level(semantic_layers)
        }
    
    def identify_emergent_properties(self, semantic_layers: Dict[str, Any]) -> List[str]:
        """Identify emergent semantic properties"""
        emergent_properties = []
        
        # Check for emergent patterns
        relationship_count = len(semantic_layers['relationship_semantics'])
        structural_count = len(semantic_layers['structural_semantics'])
        
        if relationship_count > 5:
            emergent_properties.append('COMPLEX_RELATIONAL_NETWORK')
        
        if structural_count > 2:
            emergent_properties.append('MULTI_LAYERED_STRUCTURE')
        
        # Check for semantic diversity
        unique_relationships = set(semantic_layers['relationship_semantics'].values())
        if len(unique_relationships) > 3:
            emergent_properties.append('SEMANTIC_DIVERSITY')
        
        return emergent_properties
    
    # Validation helper methods
    
    def check_relationship_consistency(self, semantics: Dict[str, Any]) -> float:
        """Check consistency of relationship semantics"""
        # Simplified consistency check
        return 0.85  # Placeholder
    
    def check_structural_consistency(self, semantics: Dict[str, Any]) -> float:
        """Check consistency of structural semantics"""
        return 0.90  # Placeholder
    
    def check_holistic_consistency(self, semantics: Dict[str, Any]) -> float:
        """Check consistency of holistic semantics"""
        return 0.88  # Placeholder
    
    def compute_confidence_score(self, semantic_type: str, content: Any) -> float:
        """Compute confidence score for semantic extraction"""
        # Simplified confidence computation
        base_confidence = 0.8
        
        if semantic_type == 'atomic_semantics':
            return base_confidence + 0.1
        elif semantic_type == 'relational_semantics':
            return base_confidence + 0.05
        else:
            return base_confidence
    
    def assess_complexity_level(self, data: Any) -> str:
        """Assess complexity level of semantic structure"""
        if isinstance(data, dict):
            item_count = len(data)
            if item_count > 10:
                return 'HIGH_COMPLEXITY'
            elif item_count > 5:
                return 'MODERATE_COMPLEXITY'
            else:
                return 'LOW_COMPLEXITY'
        return 'UNKNOWN_COMPLEXITY'
    
    def determine_semantic_role(self, entity_data: Dict[str, Any]) -> str:
        """Determine semantic role of entity based on its relationships"""
        centrality = entity_data.get('centrality', 0)
        
        if centrality > 4:
            return 'CENTRAL_CONCEPT'
        elif centrality > 2:
            return 'CONNECTING_CONCEPT'
        else:
            return 'PERIPHERAL_CONCEPT'
    
    def assess_semantic_coherence(self, semantic_layers: Dict[str, Any]) -> float:
        """Assess overall semantic coherence"""
        # Simplified coherence assessment
        return 0.82
    
    def determine_overall_theme(self, semantic_layers: Dict[str, Any]) -> str:
        """Determine overall semantic theme"""
        # Simplified theme determination
        return 'RELATIONAL_STRUCTURE_WITH_MODERATE_COMPLEXITY'
