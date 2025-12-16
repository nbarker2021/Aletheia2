"""Module `script_3.py` — part of CQE (geometry-first, receipts-first).""" # Phase 4: Key Findings Extraction & Validation Results
print("="*60)
print("Phase 4: Key Findings Extraction")
print("="*60)

def extract_key_findings(atoms, patterns):
    """Extract specific findings, test results, and validations from the log"""
    
    findings = {
        'trajectory_calculations': [],
        'geometric_validations': [],
        'falsifier_results': [],
        'performance_metrics': [],
        'mathematical_proofs': [],
        'system_integrations': []
    }
    
    # Look for specific numeric findings and results
    numeric_pattern = r'[-+]? * ? +(?:[eE][-+]? +)?'
    
    for atom in atoms:
        text = atom['text']
        
        # Extract trajectory-related calculations
        if any(kw in text.lower() for kw in ['mars', 'delta', 'phase', 'transfer', 'hohmann']):
            numbers = re.findall(numeric_pattern, text)
            if numbers:
                findings['trajectory_calculations'].append({
                    'text': text[:300],
                    'numbers': numbers[:5],  # Limit numbers
                    'atom_id': atom['id']
                })
        
        # Extract validation results (PASS/FAIL, TRUE/FALSE, etc.)
        if any(kw in text.lower() for kw in ['pass', 'fail', 'validated', 'confirmed', 'verified']):
            findings['geometric_validations'].append({
                'text': text[:300],
                'atom_id': atom['id'],
                'status': 'PASS' if any(kw in text.lower() for kw in ['pass', 'validated', 'confirmed']) else 'UNKNOWN'
            })
        
        # Extract falsifier gate results (F1, F2, etc.)
        falsifier_match = re.search(r'F[1-6].*?(GREEN|RED|PASS|FAIL)', text, re.IGNORECASE)
        if falsifier_match:
            findings['falsifier_results'].append({
                'text': text[:300],
                'result': falsifier_match.group(0),
                'atom_id': atom['id']
            })
        
        # Extract performance metrics
        if any(kw in text.lower() for kw in ['time', 'memory', 'cpu', 'performance', 'speed']):
            numbers = re.findall(numeric_pattern, text)
            if numbers:
                findings['performance_metrics'].append({
                    'text': text[:300],
                    'metrics': numbers[:3],
                    'atom_id': atom['id']
                })
        
        # Extract mathematical statements
        if any(kw in text.lower() for kw in ['theorem', 'proof', 'lemma', 'equation', 'formula']):
            findings['mathematical_proofs'].append({
                'text': text[:300],
                'atom_id': atom['id']
            })
        
        # Extract system integration mentions
        if any(kw in text.lower() for kw in ['integration', 'pipeline', 'workflow', 'architecture']):
            findings['system_integrations'].append({
                'text': text[:300],
                'atom_id': atom['id']
            })
    
    return findings

# Extract findings
key_findings = extract_key_findings(enriched_atoms, patterns)

print("🔍 Key Findings Summary:")
for category, items in key_findings.items():
    print(f"  {category}: {len(items)} findings")
    if items:
        # Show a sample finding
        sample = items[0]['text'][:100] + "..." if len(items[0]['text']) > 100 else items[0]['text']
        print(f"    Sample: {sample}")

# Phase 5: Generate Master Report Structure
print(" " + "="*60)
print("Phase 5: Master Report Generation")
print("="*60)

# Compile comprehensive statistics
master_stats = {
    'document_overview': {
        'total_characters': len(grok_log),
        'total_words': len(grok_log.split()),
        'total_atoms': len(atoms),
        'code_blocks': len(code_blocks),
        'functions_found': len(functions),
        'classes_found': len(classes)
    },
    'theme_analysis': pattern_stats,
    'cross_connections': len(cross_theme_connections),
    'key_findings_count': {cat: len(items) for cat, items in key_findings.items()},
    'technical_depth': {
        'unique_technical_terms': len(set(technical_terms)),
        'multi_theme_atoms': len([a for a in enriched_atoms if len(a['themes']) > 1]),
        'avg_atom_complexity': np.mean([len(a['text'].split()) for a in atoms])
    }
}

print("📊 Master Statistics Generated:")
print(f"  Document size: {master_stats['document_overview']['total_characters']:,} chars")
print(f"  Atomization: {master_stats['document_overview']['total_atoms']:,} micro-statements")
print(f"  Code elements: {master_stats['document_overview']['functions_found']} functions, {master_stats['document_overview']['classes_found']} classes")
print(f"  Thematic depth: {len(pattern_stats)} themes with {master_stats['cross_connections']} cross-connections")

# Save master report data
master_report_data = {
    'statistics': master_stats,
    'key_findings_summary': {cat: len(items) for cat, items in key_findings.items()},
    'top_themes': sorted(pattern_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10],
    'processing_metadata': {
        'analysis_date': '2025-10-12',
        'processing_phases': ['atomization', 'pattern_extraction', 'code_analysis', 'findings_extraction'],
        'total_processing_time': 'simulated'
    }
}

with open('grok_master_report.json', 'w') as f:
    json.dump(master_report_data, f, indent=2)

print(f" ✅ Phase 5 Complete: Master report saved to grok_master_report.json")

# Generate file manifest
manifest = {
    'generated_files': [
        'grok_atomization.json',
        'grok_patterns.json', 
        'grok_code_analysis.json',
        'grok_master_report.json'
    ],
    'file_descriptions': {
        'grok_atomization.json': 'Raw text atomization and micro-statement extraction',
        'grok_patterns.json': 'Thematic pattern analysis and cross-connections',
        'grok_code_analysis.json': 'Code structure and function categorization',
        'grok_master_report.json': 'Comprehensive analysis statistics and summaries'
    }
}

with open('analysis_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"📋 Analysis complete! Generated {len(manifest['generated_files'])} analysis files.")
print("🎯 Ready for detailed findings presentation...")