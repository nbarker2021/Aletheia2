from pathlib import Path
class CQEUniverseAnalyzer:
    """Comprehensive analyzer for the entire CQE data universe."""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path)
        self.documents = {}
        self.patterns = defaultdict(list)
        self.connections = defaultdict(set)
        self.concept_graph = defaultdict(dict)
        self.e8_embeddings = {}
        self.orbital_relationships = defaultdict(list)
        
        # Core CQE concepts for pattern recognition
        self.core_concepts = {
            'mathematical': [
                'e8', 'lattice', 'quadratic', 'palindrome', 'invariant', 'symmetry',
                'modular', 'residue', 'crt', 'golay', 'weyl', 'chamber', 'root'
            ],
            'algorithmic': [
                'morsr', 'alena', 'optimization', 'convergence', 'validation',
                'governance', 'constraint', 'objective', 'exploration', 'search'
            ],
            'structural': [
                'quad', 'triad', 'sequence', 'braid', 'helix', 'strand', 'interleave',
                'lawful', 'canonical', 'normal', 'form', 'embedding'
            ],
            'thermodynamic': [
                'entropy', 'energy', 'information', 'temperature', 'equilibrium',
                'conservation', 'thermodynamic', 'boltzmann', 'planck'
            ],
            'governance': [
                'tqf', 'uvibs', 'policy', 'channel', 'enforcement', 'compliance',
                'validation', 'certification', 'lawfulness', 'governance'
            ]
        }
        
        # Pattern templates for recognition
        self.pattern_templates = {
            'mathematical_formula': r'[A-Za-z_]+\s*=\s*[^=\n]+',
            'dimensional_reference': r'n\s*=\s*\d+|dimension\s*\d+|\d+d\s',
            'optimization_metric': r'score|objective|fitness|quality|performance',
            'validation_claim': r'validated|verified|proven|demonstrated|confirmed',
            'connection_indicator': r'connects?|links?|relates?|corresponds?|maps?',
            'emergence_pattern': r'emerges?|arises?|appears?|manifests?|develops?'
        }
    
    def load_universe(self):
        """Load all documents in the CQE universe."""
        print("Loading CQE universe documents...")
        
        # Recursively find all text files
        for file_path in self.base_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.md', '.txt', '.py']:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    doc_id = str(file_path.relative_to(self.base_path))
                    self.documents[doc_id] = {
                        'path': file_path,
                        'content': content,
                        'size': len(content),
                        'concepts': self._extract_concepts(content),
                        'patterns': self._extract_patterns(content),
                        'formulas': self._extract_formulas(content),
                        'connections': self._extract_connections(content)
                    }
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(self.documents)} documents")
        return self.documents
    
    def _extract_concepts(self, content: str) -> Dict[str, List[str]]:
        """Extract core CQE concepts from document content."""
        concepts = defaultdict(list)
        content_lower = content.lower()
        
        for category, concept_list in self.core_concepts.items():
            for concept in concept_list:
                # Find all occurrences with context
                pattern = rf'\b{re.escape(concept)}\b'
                matches = list(re.finditer(pattern, content_lower))
                
                for match in matches:
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end].strip()
                    concepts[category].append({
                        'concept': concept,
                        'position': match.start(),
                        'context': context
                    })
        
        return concepts
    
    def _extract_patterns(self, content: str) -> Dict[str, List[str]]:
        """Extract pattern instances from document content."""
        patterns = {}
        
        for pattern_name, pattern_regex in self.pattern_templates.items():
            matches = re.findall(pattern_regex, content, re.IGNORECASE | re.MULTILINE)
            patterns[pattern_name] = matches
        
        return patterns
    
    def _extract_formulas(self, content: str) -> List[str]:
        """Extract mathematical formulas and equations."""
        # Look for mathematical expressions
        formula_patterns = [
            r'[A-Za-z_]+\s*=\s*[^=\n]+',  # Basic equations
            r'\$[^$]+\$',  # LaTeX inline math
            r'\$\$[^$]+\$\$',  # LaTeX display math
            r'```math[^`]+```',  # Markdown math blocks
        ]
        
        formulas = []
        for pattern in formula_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            formulas.extend(matches)
        
        return formulas
    
    def _extract_connections(self, content: str) -> List[Dict[str, str]]:
        """Extract explicit connections mentioned in the content."""
        connections = []
        
        # Look for connection phrases
        connection_patterns = [
            r'(\w+)\s+connects?\s+to\s+(\w+)',
            r'(\w+)\s+links?\s+to\s+(\w+)',
            r'(\w+)\s+relates?\s+to\s+(\w+)',
            r'(\w+)\s+corresponds?\s+to\s+(\w+)',
            r'(\w+)\s+maps?\s+to\s+(\w+)'
        ]
        
        for pattern in connection_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                connections.append({
                    'source': match[0].lower(),
                    'target': match[1].lower(),
                    'type': 'explicit'
                })
        
        return connections
    
    def analyze_cross_document_patterns(self) -> Dict[str, Any]:
        """Analyze patterns that span across multiple documents."""
        print("Analyzing cross-document patterns...")
        
        # Concept co-occurrence analysis
        concept_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for doc_id, doc_data in self.documents.items():
            doc_concepts = set()
            for category, concept_list in doc_data['concepts'].items():
                for concept_data in concept_list:
                    doc_concepts.add(concept_data['concept'])
            
            # Count co-occurrences
            for concept1 in doc_concepts:
                for concept2 in doc_concepts:
                    if concept1 != concept2:
                        concept_cooccurrence[concept1][concept2] += 1
        
        # Pattern evolution analysis
        pattern_evolution = self._analyze_pattern_evolution()
        
        # Concept clustering
        concept_clusters = self._cluster_concepts(concept_cooccurrence)
        
        # Connection strength analysis
        connection_strengths = self._analyze_connection_strengths()
        
        return {
            'concept_cooccurrence': dict(concept_cooccurrence),
            'pattern_evolution': pattern_evolution,
            'concept_clusters': concept_clusters,
            'connection_strengths': connection_strengths
        }
    
    def _analyze_pattern_evolution(self) -> Dict[str, List[str]]:
        """Analyze how patterns evolve across documents."""
        evolution = defaultdict(list)
        
        # Sort documents by creation time (approximated by path structure)
        sorted_docs = sorted(self.documents.items(), 
                           key=lambda x: x[0])  # Simple alphabetical sort as proxy
        
        for doc_id, doc_data in sorted_docs:
            for pattern_type, patterns in doc_data['patterns'].items():
                if patterns:
                    evolution[pattern_type].extend(patterns)
        
        return dict(evolution)
    
    def _cluster_concepts(self, cooccurrence: Dict[str, Dict[str, int]]) -> Dict[str, List[str]]:
        """Cluster concepts based on co-occurrence patterns."""
        # Simple clustering based on co-occurrence strength
        clusters = defaultdict(list)
        processed = set()
        
        for concept1, connections in cooccurrence.items():
            if concept1 in processed:
                continue
            
            cluster = [concept1]
            processed.add(concept1)
            
            # Find strongly connected concepts
            for concept2, strength in connections.items():
                if concept2 not in processed and strength >= 3:  # Threshold
                    cluster.append(concept2)
                    processed.add(concept2)
            
            if len(cluster) > 1:
                cluster_name = f"cluster_{len(clusters)}"
                clusters[cluster_name] = cluster
        
        return dict(clusters)
    
    def _analyze_connection_strengths(self) -> Dict[str, float]:
        """Analyze the strength of connections between concepts."""
        strengths = defaultdict(float)
        
        for doc_id, doc_data in self.documents.items():
            for connection in doc_data['connections']:
                source = connection['source']
                target = connection['target']
                conn_key = f"{source}->{target}"
                strengths[conn_key] += 1.0
        
        # Normalize by document count
        total_docs = len(self.documents)
        for key in strengths:
            strengths[key] /= total_docs
        
        return dict(strengths)
    
    def discover_hidden_patterns(self) -> Dict[str, Any]:
        """Discover hidden patterns not explicitly mentioned."""
        print("Discovering hidden patterns...")
        
        hidden_patterns = {}
        
        # Numerical pattern analysis
        hidden_patterns['numerical'] = self._find_numerical_patterns()
        
        # Structural pattern analysis
        hidden_patterns['structural'] = self._find_structural_patterns()
        
        # Semantic pattern analysis
        hidden_patterns['semantic'] = self._find_semantic_patterns()
        
        # Emergence pattern analysis
        hidden_patterns['emergence'] = self._find_emergence_patterns()
        
        return hidden_patterns
    
    def _find_numerical_patterns(self) -> Dict[str, Any]:
        """Find hidden numerical patterns across documents."""
        numbers = []
        
        # Extract all numbers from documents
        for doc_data in self.documents.values():
            content = doc_data['content']
            found_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', content)
            numbers.extend([float(n) for n in found_numbers])
        
        # Analyze number distributions
        number_counter = Counter(numbers)
        most_common = number_counter.most_common(20)
        
        # Look for mathematical relationships
        relationships = []
        for i, (num1, count1) in enumerate(most_common[:10]):
            for j, (num2, count2) in enumerate(most_common[i+1:10]):
                ratio = num1 / num2 if num2 != 0 else 0
                if abs(ratio - round(ratio)) < 0.01:  # Near integer ratio
                    relationships.append({
                        'num1': num1,
                        'num2': num2,
                        'ratio': round(ratio),
                        'significance': count1 + count2
                    })
        
        return {
            'most_common_numbers': most_common,
            'mathematical_relationships': relationships,
            'total_numbers': len(numbers)
        }
    
    def _find_structural_patterns(self) -> Dict[str, Any]:
        """Find hidden structural patterns in the documents."""
        structures = defaultdict(int)
        
        for doc_data in self.documents.values():
            content = doc_data['content']
            
            # Count structural elements
            structures['bullet_points'] += len(re.findall(r'^\s*[-*+]\s', content, re.MULTILINE))
            structures['numbered_lists'] += len(re.findall(r'^\s*\d+\.\s', content, re.MULTILINE))
            structures['headers'] += len(re.findall(r'^#+\s', content, re.MULTILINE))
            structures['code_blocks'] += len(re.findall(r'```', content))
            structures['emphasis'] += len(re.findall(r'\*\*[^*]+\*\*', content))
            structures['links'] += len(re.findall(r'\[([^\]]+)\]\([^)]+\)', content))
        
        return dict(structures)
    
    def _find_semantic_patterns(self) -> Dict[str, Any]:
        """Find hidden semantic patterns across documents."""
        semantic_patterns = {}
        
        # Analyze word frequency patterns
        all_words = []
        for doc_data in self.documents.values():
            content = doc_data['content'].lower()
            words = re.findall(r'\b[a-z]+\b', content)
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        
        # Find domain-specific terminology
        domain_terms = {}
        for category, concepts in self.core_concepts.items():
            category_words = [word for word, freq in word_freq.most_common(100) 
                            if any(concept in word for concept in concepts)]
            domain_terms[category] = category_words[:10]
        
        semantic_patterns['word_frequency'] = word_freq.most_common(50)
        semantic_patterns['domain_terminology'] = domain_terms
        
        return semantic_patterns
    
    def _find_emergence_patterns(self) -> Dict[str, Any]:
        """Find patterns of emergence and development."""
        emergence = {}
        
        # Track concept introduction and development
        concept_timeline = defaultdict(list)
        
        for doc_id, doc_data in self.documents.items():
            for category, concept_list in doc_data['concepts'].items():
                for concept_data in concept_list:
                    concept_timeline[concept_data['concept']].append(doc_id)
        
        # Identify concepts that emerge later
        late_emerging = {}
        for concept, appearances in concept_timeline.items():
            if len(appearances) >= 3:  # Appears in multiple documents
                late_emerging[concept] = len(appearances)
        
        emergence['concept_timeline'] = dict(concept_timeline)
        emergence['late_emerging_concepts'] = late_emerging
        
        return emergence
    
    def create_24d_lattice_embedding(self) -> Dict[str, np.ndarray]:
        """Create 24D lattice embeddings for all concepts."""
        print("Creating 24D lattice embeddings...")
        
        # Define the 24 dimensions as specified in the universe mapping
        dimensions = [
            # Mathematical dimensions (8D)
            'algebraic_structures', 'geometric_relationships', 'topological_properties',
            'analytical_functions', 'symmetry_operations', 'modular_arithmetic',
            'information_theory', 'thermodynamic_principles',
            
            # Implementation dimensions (8D)
            'algorithmic_structures', 'data_representations', 'computational_complexity',
            'validation_mechanisms', 'interface_designs', 'performance_optimization',
            'error_handling', 'extensibility_patterns',
            
            # Application dimensions (8D)
            'problem_domains', 'solution_patterns', 'use_case_scenarios',
            'performance_metrics', 'user_interactions', 'integration_contexts',
            'scalability_factors', 'impact_measurements'
        ]
        
        embeddings = {}
        
        for doc_id, doc_data in self.documents.items():
            # Create 24D vector for this document
            vector = np.zeros(24)
            
            # Mathematical dimensions (0-7)
            math_concepts = doc_data['concepts'].get('mathematical', [])
            vector[0] = len(math_concepts) / 10.0  # Normalize
            vector[1] = len([c for c in math_concepts if 'e8' in c['concept']]) / 5.0
            vector[2] = len([c for c in math_concepts if 'braid' in c['concept']]) / 5.0
            vector[3] = len(doc_data['formulas']) / 10.0
            vector[4] = len([c for c in math_concepts if 'symmetry' in c['concept']]) / 5.0
            vector[5] = len([c for c in math_concepts if 'modular' in c['concept']]) / 5.0
            vector[6] = len([c for c in math_concepts if 'entropy' in c['concept']]) / 5.0
            vector[7] = len([c for c in math_concepts if 'energy' in c['concept']]) / 5.0
            
            # Implementation dimensions (8-15)
            algo_concepts = doc_data['concepts'].get('algorithmic', [])
            vector[8] = len(algo_concepts) / 10.0
            vector[9] = len([c for c in algo_concepts if 'data' in c['concept']]) / 5.0
            vector[10] = len([c for c in algo_concepts if 'complex' in c['concept']]) / 5.0
            vector[11] = len([c for c in algo_concepts if 'valid' in c['concept']]) / 5.0
            vector[12] = len([c for c in algo_concepts if 'interface' in c['concept']]) / 5.0
            vector[13] = len([c for c in algo_concepts if 'optim' in c['concept']]) / 5.0
            vector[14] = len([c for c in algo_concepts if 'error' in c['concept']]) / 5.0
            vector[15] = len([c for c in algo_concepts if 'extend' in c['concept']]) / 5.0
            
            # Application dimensions (16-23)
            struct_concepts = doc_data['concepts'].get('structural', [])
            vector[16] = len(struct_concepts) / 10.0
            vector[17] = len([c for c in struct_concepts if 'pattern' in c['concept']]) / 5.0
            vector[18] = len([c for c in struct_concepts if 'case' in c['concept']]) / 5.0
            vector[19] = len([c for c in struct_concepts if 'performance' in c['concept']]) / 5.0
            vector[20] = len([c for c in struct_concepts if 'user' in c['concept']]) / 5.0
            vector[21] = len([c for c in struct_concepts if 'integration' in c['concept']]) / 5.0
            vector[22] = len([c for c in struct_concepts if 'scale' in c['concept']]) / 5.0
            vector[23] = len([c for c in struct_concepts if 'impact' in c['concept']]) / 5.0
            
            embeddings[doc_id] = vector
        
        return embeddings
    
    def find_e8_connection_paths(self, source_doc: str, target_doc: str) -> List[str]:
        """Find E₈ connection paths between two documents."""
        if source_doc not in self.e8_embeddings or target_doc not in self.e8_embeddings:
            return []
        
        source_vector = self.e8_embeddings[source_doc][:8]  # Use first 8 dimensions for E₈
        target_vector = self.e8_embeddings[target_doc][:8]
        
        # Simple path finding through intermediate documents
        all_docs = list(self.e8_embeddings.keys())
        
        # Find documents that are geometrically between source and target
        intermediate_docs = []
        for doc_id in all_docs:
            if doc_id == source_doc or doc_id == target_doc:
                continue
            
            doc_vector = self.e8_embeddings[doc_id][:8]
            
            # Check if this document is on the path (simplified geometric test)
            source_dist = np.linalg.norm(doc_vector - source_vector)
            target_dist = np.linalg.norm(doc_vector - target_vector)
            direct_dist = np.linalg.norm(target_vector - source_vector)
            
            # If the sum of distances is close to direct distance, it's on the path
            if abs((source_dist + target_dist) - direct_dist) < 0.1:
                intermediate_docs.append((doc_id, source_dist))
        
        # Sort by distance from source
        intermediate_docs.sort(key=lambda x: x[1])
        
        # Return path
        path = [source_doc]
        path.extend([doc[0] for doc in intermediate_docs[:3]])  # Limit to 3 intermediate
        path.append(target_doc)
        
        return path
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        print("Generating comprehensive analysis report...")
        
        # Load universe if not already loaded
        if not self.documents:
            self.load_universe()
        
        # Create embeddings
        self.e8_embeddings = self.create_24d_lattice_embedding()
        
        # Perform all analyses
        cross_doc_patterns = self.analyze_cross_document_patterns()
        hidden_patterns = self.discover_hidden_patterns()
        
        # Generate summary statistics
        summary_stats = {
            'total_documents': len(self.documents),
            'total_concepts': sum(len(doc['concepts']) for doc in self.documents.values()),
            'total_formulas': sum(len(doc['formulas']) for doc in self.documents.values()),
            'total_connections': sum(len(doc['connections']) for doc in self.documents.values()),
            'average_doc_size': np.mean([doc['size'] for doc in self.documents.values()]),
            'concept_diversity': len(set().union(*[
                [c['concept'] for cat in doc['concepts'].values() for c in cat]
                for doc in self.documents.values()
            ]))
        }
        
        # Find strongest connections
        strongest_connections = self._find_strongest_connections()
        
        # Identify key documents
        key_documents = self._identify_key_documents()
        
        return {
            'summary_statistics': summary_stats,
            'cross_document_patterns': cross_doc_patterns,
            'hidden_patterns': hidden_patterns,
            'strongest_connections': strongest_connections,
            'key_documents': key_documents,
            'embeddings_created': len(self.e8_embeddings),
            'analysis_timestamp': 'October 9, 2025'
        }
    
    def _find_strongest_connections(self) -> List[Dict[str, Any]]:
        """Find the strongest connections in the universe."""
        connections = []
        
        for doc_id, doc_data in self.documents.items():
            for connection in doc_data['connections']:
                connections.append({
                    'source_doc': doc_id,
                    'source_concept': connection['source'],
                    'target_concept': connection['target'],
                    'type': connection['type']
                })
        
        # Count connection frequencies
        connection_counts = Counter([
            f"{conn['source_concept']}->{conn['target_concept']}"
            for conn in connections
        ])
        
        return [
            {'connection': conn, 'frequency': freq}
            for conn, freq in connection_counts.most_common(20)
        ]
    
    def _identify_key_documents(self) -> List[Dict[str, Any]]:
        """Identify key documents in the universe."""
        doc_scores = []
        
        for doc_id, doc_data in self.documents.items():
            # Score based on multiple factors
            concept_score = sum(len(concepts) for concepts in doc_data['concepts'].values())
            formula_score = len(doc_data['formulas']) * 2
            connection_score = len(doc_data['connections']) * 3
            size_score = min(doc_data['size'] / 1000, 10)  # Cap at 10
            
            total_score = concept_score + formula_score + connection_score + size_score
            
            doc_scores.append({
                'document': doc_id,
                'total_score': total_score,
                'concept_score': concept_score,
                'formula_score': formula_score,
                'connection_score': connection_score,
                'size_score': size_score
            })
        
        # Sort by total score
        doc_scores.sort(key=lambda x: x['total_score'], reverse=True)
        
        return doc_scores[:20]  # Top 20 documents

if __name__ == "__main__":
    analyzer = CQEUniverseAnalyzer()
    report = analyzer.generate_comprehensive_report()
    
    # Save report
    output_path = Path(__file__).parent / "cqe_analysis/universe_exploration/deep_analysis_report.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Deep analysis complete. Report saved to {output_path}")
    print(f"Analyzed {report['summary_statistics']['total_documents']} documents")
    print(f"Found {report['summary_statistics']['concept_diversity']} unique concepts")
    print(f"Created {report['embeddings_created']} 24D embeddings")
"""
Domain Adapter for CQE System

Converts problem instances from various domains (P/NP, optimization, scenes)
into 8-dimensional feature vectors suitable for E₈ lattice embedding.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import hashlib
