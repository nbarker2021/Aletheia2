"""Module `script.py` — part of CQE (geometry-first, receipts-first).""" # CQE Lab Mode: Comprehensive Grok Log Processing Engine
# Phase 1: Document Ingestion & Atomization

import re
import json
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Read the full grok log
with open('grok-build.txt', 'r', encoding='utf-8') as f:
    grok_log = f.read()

print(f"📊 Grok Log Size: {len(grok_log):,} chars, {len(grok_log.split()):,} words")

# Phase 1: Atomization - split into micro-statements
def atomize_text(text):
    """Split text into atomic statements for analysis"""
    # Split on sentence boundaries but preserve context
    sentences = re.split(r'(?<=[.!?]) +(?=[A-Z])', text)
    
    atoms = []
    for i, sentence in enumerate(sentences):
        # Further split on logical connectors while preserving meaning
        parts = re.split(r' (and|or|therefore|thus|however|meanwhile) ', sentence, flags=re.IGNORECASE)
        
        for j, part in enumerate(parts):
            if part.strip() and not re.match(r'^(and|or|therefore|thus|however|meanwhile)$', part, re.IGNORECASE):
                atoms.append({
                    'id': f"atom_{i}_{j}",
                    'text': part.strip(),
                    'sentence_id': i,
                    'length': len(part.strip())
                })
    
    return atoms

# Atomize the log
atoms = atomize_text(grok_log)
print(f"🔬 Atomized into {len(atoms)} micro-statements")

# Phase 2: Extract code blocks and technical terms
code_patterns = [
    r'```[  ]*?```',  # Code blocks
    r'`[^`]+`',  # Inline code
    r'def + + [^)]* :', # Function definitions
    r'class + +[  ]*?:', # Class definitions
    r' [A-Z_]{3,} ',  # Constants
]

technical_terms = []
code_blocks = []

for atom in atoms:
    text = atom['text']
    
    # Extract code blocks
    for pattern in code_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match.strip()) > 5:  # Filter out tiny matches
                code_blocks.append({
                    'code': match,
                    'atom_id': atom['id'],
                    'type': 'code_block' if '```' in match else 'inline_code'
                })
    
    # Extract technical terms (camelCase, CONSTANTS, function calls)
    tech_matches = re.findall(r' (?:[a-z]+[A-Z][a-zA-Z]*|[A-Z_]{2,}| +  ) ', text)
    technical_terms.extend(tech_matches)

print(f"💻 Extracted {len(code_blocks)} code blocks")
print(f"🔧 Found {len(set(technical_terms))} unique technical terms")

# Save atomization results
atomization_data = {
    'total_atoms': len(atoms),
    'avg_atom_length': np.mean([a['length'] for a in atoms]),
    'code_blocks_count': len(code_blocks),
    'unique_tech_terms': len(set(technical_terms)),
    'sample_atoms': atoms[:10],
    'sample_code': code_blocks[:5]
}

with open('grok_atomization.json', 'w') as f:
    json.dump(atomization_data, f, indent=2)

print("✅ Phase 1 Complete: Atomization data saved to grok_atomization.json")