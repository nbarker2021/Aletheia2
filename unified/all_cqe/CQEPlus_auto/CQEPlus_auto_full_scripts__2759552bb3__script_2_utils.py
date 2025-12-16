"""Module `script_2.py` — part of CQE (geometry-first, receipts-first).""" # Fix the JSON serialization issue and continue with Phase 2

# Convert cross-theme connections to string keys for JSON
cross_connections_str = {f"{theme1}↔{theme2}": count for (theme1, theme2), count in cross_theme_connections.items()}

# Save pattern analysis with corrected data
pattern_data = {
    'pattern_stats': pattern_stats,
    'cross_connections': cross_connections_str,
    'total_patterns': sum(len(matches) for matches in patterns.values()),
    'multi_theme_atoms': len([a for a in enriched_atoms if len(a['themes']) > 1])
}

with open('grok_patterns.json', 'w') as f:
    json.dump(pattern_data, f, indent=2)

print(f"✅ Phase 2 Complete: Pattern data saved to grok_patterns.json")
print(f"📊 Total classified atoms: {pattern_data['total_patterns']}")
print(f"🔄 Multi-theme atoms: {pattern_data['multi_theme_atoms']}")

# Phase 3: Code Extraction & Function Analysis
print(" " + "="*60)
print("Phase 3: Code Extraction & Function Analysis")
print("="*60)

def extract_functions_and_classes(code_blocks):
    """Extract detailed function and class information from code blocks"""
    
    functions = []
    classes = []
    constants = []
    
    for block in code_blocks:
        code = block['code']
        
        # Extract function definitions
        func_pattern = r'def +( +) * [^)]* :'
        func_matches = re.findall(func_pattern, code)
        for func_name in func_matches:
            # Extract the full function signature
            full_sig_pattern = rf'def +{func_name} * [^)]* :[^ ]*'
            sig_match = re.search(full_sig_pattern, code)
            if sig_match:
                functions.append({
                    'name': func_name,
                    'signature': sig_match.group(0),
                    'atom_id': block['atom_id'],
                    'block_type': block['type']
                })
        
        # Extract class definitions
        class_pattern = r'class +( +)(?: [^)]* )?:'
        class_matches = re.findall(class_pattern, code)
        for class_name in class_matches:
            classes.append({
                'name': class_name,
                'atom_id': block['atom_id'],
                'block_type': block['type']
            })
        
        # Extract constants (ALL_CAPS identifiers)
        const_pattern = r' [A-Z_]{3,} '
        const_matches = re.findall(const_pattern, code)
        for const in const_matches:
            if len(const) > 2:  # Filter out short matches
                constants.append({
                    'name': const,
                    'atom_id': block['atom_id'],
                    'block_type': block['type']
                })
    
    return functions, classes, constants

# Extract code elements
functions, classes, constants = extract_functions_and_classes(code_blocks)

print(f"🔧 Functions found: {len(functions)}")
print(f"🏗️  Classes found: {len(classes)}")
print(f"📏 Constants found: {len(set(c['name'] for c in constants))}")

# Categorize by CQE subsystem
function_categories = defaultdict(list)
for func in functions:
    name = func['name'].lower()
    if any(kw in name for kw in ['e8', 'root', 'lattice']):
        function_categories['e8_lattice'].append(func)
    elif any(kw in name for kw in ['morsr', 'pulse']):
        function_categories['morsr'].append(func)
    elif any(kw in name for kw in ['rag', 'retrieval', 'embed']):
        function_categories['rag'].append(func)
    elif any(kw in name for kw in ['snap', 'project', 'transform']):
        function_categories['geometric'].append(func)
    elif any(kw in name for kw in ['validate', 'test', 'check']):
        function_categories['validation'].append(func)
    else:
        function_categories['general'].append(func)

print(" 🎯 Function Categories:")
for category, funcs in function_categories.items():
    print(f"  {category}: {len(funcs)} functions")
    if funcs:
        sample_names = [f['name'] for f in funcs[:3]]
        print(f"    Examples: {', '.join(sample_names)}")

# Save code analysis
code_analysis = {
    'functions': functions[:50],  # Limit for storage
    'classes': classes,
    'unique_constants': list(set(c['name'] for c in constants))[:50],
    'function_categories': {cat: len(funcs) for cat, funcs in function_categories.items()},
    'total_code_blocks': len(code_blocks)
}

with open('grok_code_analysis.json', 'w') as f:
    json.dump(code_analysis, f, indent=2)

print(f" ✅ Phase 3 Complete: Code analysis saved to grok_code_analysis.json")