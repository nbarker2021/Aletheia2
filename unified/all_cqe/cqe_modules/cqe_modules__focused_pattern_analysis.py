#!/usr/bin/env python3
"""
Focused Pattern Analysis for CQE Universe
Efficient analysis targeting key patterns and connections
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Any

class FocusedCQEAnalyzer:
    """Efficient analyzer focusing on key CQE patterns."""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path)
        self.key_patterns = {}
        self.concept_connections = defaultdict(set)
        self.evidence_chains = defaultdict(list)
        
        # Focus on most important concepts
        self.priority_concepts = {
            'core_mathematical': ['e8', 'lattice', 'quadratic', 'palindrome', 'invariant'],
            'core_algorithmic': ['morsr', 'alena', 'optimization', 'convergence'],
            'core_structural': ['quad', 'triad', 'braid', 'lawful', 'canonical'],
            'core_governance': ['tqf', 'uvibs', 'policy', 'validation', 'enforcement']
        }
        
        # Key pattern indicators
        self.pattern_indicators = {
            'mathematical_breakthrough': [
                'breakthrough', 'discovery', 'proof', 'theorem', 'solution'
            ],
            'evidence_validation': [
                'validated', 'verified', 'confirmed', 'demonstrated', 'proven'
            ],
            'connection_mapping': [
                'connects', 'links', 'relates', 'corresponds', 'maps'
            ],
            'superiority_claims': [
                'better', 'superior', 'improved', 'optimal', 'breakthrough'
            ]
        }
    
    def analyze_key_documents(self) -> Dict[str, Any]:
        """Analyze only the most important documents."""
        print("Analyzing key CQE documents...")
        
        # Focus on specific high-value files
        key_files = [
            'final_integration_analysis.md',
            'COMPLETE_CQE_EVOLUTION_ANALYSIS.md',
            'cqe_unified_conceptual_framework.md',
            'patterns_trends_gaps_analysis.md',
            'system_relationship_mapping.md'
        ]
        
        analysis_results = {}
        
        for filename in key_files:
            file_paths = list(self.base_path.rglob(filename))
            if file_paths:
                file_path = file_paths[0]  # Take first match
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    analysis_results[filename] = {
                        'concepts': self._extract_priority_concepts(content),
                        'patterns': self._extract_key_patterns(content),
                        'evidence': self._extract_evidence_chains(content),
                        'connections': self._extract_concept_connections(content),
                        'insights': self._extract_insights(content)
                    }
                    
                except Exception as e:
                    print(f"Error analyzing {filename}: {e}")
        
        return analysis_results
    
    def _extract_priority_concepts(self, content: str) -> Dict[str, List[str]]:
        """Extract priority concepts with context."""
        concepts = defaultdict(list)
        content_lower = content.lower()
        
        for category, concept_list in self.priority_concepts.items():
            for concept in concept_list:
                pattern = rf'\b{re.escape(concept)}\b'
                matches = list(re.finditer(pattern, content_lower))
                
                for match in matches:
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end].strip()
                    concepts[category].append({
                        'concept': concept,
                        'context': context,
                        'position': match.start()
                    })
        
        return dict(concepts)
    
    def _extract_key_patterns(self, content: str) -> Dict[str, List[str]]:
        """Extract key pattern indicators."""
        patterns = {}
        
        for pattern_type, indicators in self.pattern_indicators.items():
            found_patterns = []
            for indicator in indicators:
                # Find sentences containing the indicator
                sentences = re.split(r'[.!?]+', content)
                for sentence in sentences:
                    if indicator.lower() in sentence.lower():
                        found_patterns.append(sentence.strip())
            
            patterns[pattern_type] = found_patterns[:5]  # Limit to top 5
        
        return patterns
    
    def _extract_evidence_chains(self, content: str) -> List[Dict[str, str]]:
        """Extract evidence chains and validation claims."""
        evidence = []
        
        # Look for evidence patterns
        evidence_patterns = [
            r'evidence[^.]*shows[^.]*',
            r'validated[^.]*through[^.]*',
            r'proven[^.]*by[^.]*',
            r'demonstrated[^.]*via[^.]*',
            r'confirmed[^.]*using[^.]*'
        ]
        
        for pattern in evidence_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                evidence.append({
                    'claim': match.strip(),
                    'type': 'validation'
                })
        
        return evidence[:10]  # Limit to top 10
    
    def _extract_concept_connections(self, content: str) -> List[Dict[str, str]]:
        """Extract explicit concept connections."""
        connections = []
        
        # Enhanced connection patterns
        connection_patterns = [
            r'(\w+)\s+(?:connects?|links?|relates?)\s+(?:to|with)\s+(\w+)',
            r'(\w+)\s+(?:corresponds?|maps?)\s+to\s+(\w+)',
            r'(\w+)\s+and\s+(\w+)\s+are\s+(?:connected|linked|related)',
            r'relationship\s+between\s+(\w+)\s+and\s+(\w+)'
        ]
        
        for pattern in connection_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                connections.append({
                    'source': match[0].lower(),
                    'target': match[1].lower(),
                    'type': 'explicit_connection'
                })
        
        return connections
    
    def _extract_insights(self, content: str) -> List[str]:
        """Extract key insights and discoveries."""
        insights = []
        
        # Look for insight indicators
        insight_patterns = [
            r'key insight[^.]*',
            r'important discovery[^.]*',
            r'breakthrough[^.]*',
            r'novel approach[^.]*',
            r'significant finding[^.]*'
        ]
        
        for pattern in insight_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            insights.extend([match.strip() for match in matches])
        
        return insights[:10]  # Limit to top 10
    
    def analyze_mathematical_superiority(self) -> Dict[str, Any]:
        """Analyze claims of mathematical superiority over existing methods."""
        print("Analyzing mathematical superiority claims...")
        
        superiority_analysis = {
            'optimization_advantages': [],
            'convergence_improvements': [],
            'universality_claims': [],
            'efficiency_gains': [],
            'theoretical_advances': []
        }
        
        # Search for superiority claims in key documents
        search_patterns = {
            'optimization_advantages': [
                r'better.*optimization', r'superior.*convergence', r'improved.*performance'
            ],
            'convergence_improvements': [
                r'\d+.*times.*faster', r'\d+.*improvement', r'exponential.*reduction'
            ],
            'universality_claims': [
                r'universal.*framework', r'domain.*agnostic', r'any.*problem'
            ],
            'efficiency_gains': [
                r'efficiency.*gain', r'computational.*advantage', r'reduced.*complexity'
            ],
            'theoretical_advances': [
                r'theoretical.*breakthrough', r'mathematical.*advance', r'novel.*theory'
            ]
        }
        
        for file_path in self.base_path.rglob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for category, patterns in search_patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            superiority_analysis[category].append({
                                'claim': match,
                                'source': str(file_path.name),
                                'context': self._get_context(content, match)
                            })
            
            except Exception:
                continue
        
        return superiority_analysis
    
    def _get_context(self, content: str, match: str) -> str:
        """Get context around a match."""
        match_pos = content.lower().find(match.lower())
        if match_pos == -1:
            return ""
        
        start = max(0, match_pos - 200)
        end = min(len(content), match_pos + len(match) + 200)
        return content[start:end].strip()
    
    def identify_irl_validation_opportunities(self) -> Dict[str, Any]:
        """Identify real-world validation opportunities."""
        print("Identifying IRL validation opportunities...")
        
        opportunities = {
            'quantum_computing': [],
            'ai_optimization': [],
            'financial_modeling': [],
            'scientific_computing': [],
            'cryptography': [],
            'game_theory': []
        }
        
        # Search for application mentions
        application_patterns = {
            'quantum_computing': [
                'quantum', 'qubit', 'superposition', 'entanglement', 'quantum.*algorithm'
            ],
            'ai_optimization': [
                'neural.*network', 'machine.*learning', 'ai.*optimization', 'deep.*learning'
            ],
            'financial_modeling': [
                'financial', 'market', 'trading', 'portfolio', 'risk.*management'
            ],
            'scientific_computing': [
                'simulation', 'modeling', 'scientific.*computing', 'numerical.*analysis'
            ],
            'cryptography': [
                'cryptography', 'encryption', 'security', 'hash', 'digital.*signature'
            ],
            'game_theory': [
                'game.*theory', 'strategy', 'equilibrium', 'decision.*theory'
            ]
        }
        
        for file_path in self.base_path.rglob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for domain, patterns in application_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            opportunities[domain].append({
                                'source': str(file_path.name),
                                'relevance': self._assess_relevance(content, pattern),
                                'implementation_notes': self._extract_implementation_notes(content, pattern)
                            })
            
            except Exception:
                continue
        
        return opportunities
    
    def _assess_relevance(self, content: str, pattern: str) -> str:
        """Assess relevance of application to CQE."""
        # Simple relevance assessment
        cqe_indicators = ['cqe', 'quadratic', 'e8', 'lattice', 'optimization']
        relevance_count = sum(1 for indicator in cqe_indicators 
                            if indicator in content.lower())
        
        if relevance_count >= 3:
            return "high"
        elif relevance_count >= 2:
            return "medium"
        else:
            return "low"
    
    def _extract_implementation_notes(self, content: str, pattern: str) -> str:
        """Extract implementation notes for the application."""
        # Find sentences near the pattern that mention implementation
        implementation_keywords = ['implement', 'apply', 'use', 'deploy', 'integrate']
        
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            if re.search(pattern, sentence, re.IGNORECASE):
                for keyword in implementation_keywords:
                    if keyword in sentence.lower():
                        return sentence.strip()
        
        return "No specific implementation notes found"
    
    def generate_focused_report(self) -> Dict[str, Any]:
        """Generate focused analysis report."""
        print("Generating focused analysis report...")
        
        # Perform focused analyses
        key_doc_analysis = self.analyze_key_documents()
        superiority_analysis = self.analyze_mathematical_superiority()
        validation_opportunities = self.identify_irl_validation_opportunities()
        
        # Extract top insights
        top_insights = self._extract_top_insights(key_doc_analysis)
        
        # Identify strongest evidence
        strongest_evidence = self._identify_strongest_evidence(key_doc_analysis)
        
        # Find connection patterns
        connection_patterns = self._analyze_connection_patterns(key_doc_analysis)
        
        return {
            'executive_summary': {
                'documents_analyzed': len(key_doc_analysis),
                'total_concepts_found': sum(len(doc['concepts']) for doc in key_doc_analysis.values()),
                'evidence_chains_identified': sum(len(doc['evidence']) for doc in key_doc_analysis.values()),
                'connection_patterns_found': len(connection_patterns)
            },
            'key_document_analysis': key_doc_analysis,
            'mathematical_superiority': superiority_analysis,
            'irl_validation_opportunities': validation_opportunities,
            'top_insights': top_insights,
            'strongest_evidence': strongest_evidence,
            'connection_patterns': connection_patterns,
            'analysis_timestamp': 'October 9, 2025'
        }
    
    def _extract_top_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract top insights from the analysis."""
        all_insights = []
        for doc_data in analysis.values():
            all_insights.extend(doc_data.get('insights', []))
        
        # Remove duplicates and sort by length (longer = more detailed)
        unique_insights = list(set(all_insights))
        unique_insights.sort(key=len, reverse=True)
        
        return unique_insights[:10]
    
    def _identify_strongest_evidence(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify strongest evidence chains."""
        all_evidence = []
        for doc_name, doc_data in analysis.items():
            for evidence in doc_data.get('evidence', []):
                evidence['source_document'] = doc_name
                all_evidence.append(evidence)
        
        # Sort by claim length and validation strength
        all_evidence.sort(key=lambda x: len(x['claim']), reverse=True)
        
        return all_evidence[:15]
    
    def _analyze_connection_patterns(self, analysis: Dict[str, Any]) -> Dict[str, int]:
        """Analyze connection patterns across documents."""
        connection_counts = defaultdict(int)
        
        for doc_data in analysis.values():
            for connection in doc_data.get('connections', []):
                source = connection['source']
                target = connection['target']
                connection_key = f"{source} -> {target}"
                connection_counts[connection_key] += 1
        
        # Return top connections
        return dict(sorted(connection_counts.items(), 
                          key=lambda x: x[1], reverse=True)[:20])

if __name__ == "__main__":
    analyzer = FocusedCQEAnalyzer()
    report = analyzer.generate_focused_report()
    
    # Save report
    output_path = Path(__file__).parent / "cqe_analysis/universe_exploration/focused_analysis_report.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Focused analysis complete. Report saved to {output_path}")
    print(f"Key insights found: {len(report['top_insights'])}")
    print(f"Evidence chains: {len(report['strongest_evidence'])}")
    print(f"Connection patterns: {len(report['connection_patterns'])}")
