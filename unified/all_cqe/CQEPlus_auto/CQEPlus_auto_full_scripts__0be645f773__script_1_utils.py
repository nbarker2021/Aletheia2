"""Module `script_1.py` — part of CQE (geometry-first, receipts-first).""" # Phase 2: Dynamic Thematic Clustering & Pattern Discovery

from collections import defaultdict
import hashlib

# Analyze patterns in the atomized text
def extract_cqe_patterns(atoms):
    """Extract CQE-specific patterns and themes"""
    
    patterns = {
        'e8_lattice': [],
        'morsr_algorithm': [],
        'rag_system': [],
        'worldforge_manifolds': [],
        'trajectory_analysis': [],
        'validation_gates': [],
        'geometric_operations': [],
        'digital_roots': [],
        'falsification_tests': [],
        'controller_logic': []
    }
    
    # Pattern matching keywords for each theme
    theme_keywords = {
        'e8_lattice': ['E8', 'lattice', 'root', '240', 'norm', 'sqrt(2)', 'Weyl', 'chamber'],
        'morsr_algorithm': ['MORSR', 'pulse', 'radius', 'dwell', 'epsilon', 'iteration'],
        'rag_system': ['RAG', 'retrieval', 'embedding', 'knowledge', 'graph'],
        'worldforge_manifolds': ['WorldForge', 'manifold', 'spawn', 'topology', 'Euler'],
        'trajectory_analysis': ['Mars', 'Hohmann', 'transfer', 'delta-v', 'phase', 'angle'],
        'validation_gates': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'falsifier', 'gate'],
        'geometric_operations': ['snap', 'project', 'torus', 'coordinate', 'transform'],
        'digital_roots': ['digital', 'root', 'modulo', '%', 'residue', 'dr'],
        'falsification_tests': ['test', 'verify', 'validate', 'assertion', 'harness'],
        'controller_logic': ['controller', 'receipt', 'JSONL', 'CSV', 'sense', 'plan', 'act']
    }
    
    # Classify atoms by theme
    for atom in atoms:
        text = atom['text'].lower()
        atom_themes = []
        
        for theme, keywords in theme_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in text)
            if score > 0:
                patterns[theme].append({
                    'atom_id': atom['id'],
                    'text': atom['text'][:200],  # Truncate for storage
                    'score': score,
                    'keywords_found': [kw for kw in keywords if kw.lower() in text]
                })
                atom_themes.append(theme)
        
        # Store multi-theme atoms for cross-cluster analysis
        atom['themes'] = atom_themes
    
    return patterns, atoms

# Extract patterns
patterns, enriched_atoms = extract_cqe_patterns(atoms)

# Generate statistics
pattern_stats = {}
for theme, matches in patterns.items():
    pattern_stats[theme] = {
        'count': len(matches),
        'avg_score': np.mean([m['score'] for m in matches]) if matches else 0,
        'top_keywords': Counter([kw for m in matches for kw in m['keywords_found']]).most_common(5)
    }

print("🎯 Thematic Pattern Analysis:")
for theme, stats in pattern_stats.items():
    print(f"  {theme}: {stats['count']} atoms (avg score: {stats['avg_score']:.2f})")
    if stats['top_keywords']:
        top_kw = [f"{kw}({cnt})" for kw, cnt in stats['top_keywords']]
        print(f"    Top keywords: {', '.join(top_kw)}")

# Identify cross-theme connections
cross_theme_connections = defaultdict(int)
for atom in enriched_atoms:
    themes = atom['themes']
    for i, theme1 in enumerate(themes):
        for theme2 in themes[i+1:]:
            pair = tuple(sorted([theme1, theme2]))
            cross_theme_connections[pair] += 1

print(" 🔗 Cross-Theme Connections:")
for (theme1, theme2), count in sorted(cross_theme_connections.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {theme1} ↔ {theme2}: {count} shared atoms")

# Save pattern analysis
pattern_data = {
    'pattern_stats': pattern_stats,
    'cross_connections': dict(cross_theme_connections),
    'total_patterns': sum(len(matches) for matches in patterns.values()),
    'multi_theme_atoms': len([a for a in enriched_atoms if len(a['themes']) > 1])
}

with open('grok_patterns.json', 'w') as f:
    json.dump(pattern_data, f, indent=2)

print(f" ✅ Phase 2 Complete: Pattern data saved to grok_patterns.json")
print(f"📊 Total classified atoms: {pattern_data['total_patterns']}")
print(f"🔄 Multi-theme atoms: {pattern_data['multi_theme_atoms']}")