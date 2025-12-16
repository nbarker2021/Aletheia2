
#!/usr/bin/env python3
"""CQE System Batch Processor - Populates full master cross-index"""

import pandas as pd
import os
import json
from pathlib import Path

class CQESystemProcessor:
    def __init__(self):
        self.file_types = {
            'Theory': ['WHY_', 'Theory', 'Proof', 'Mathematical', 'Framework'],
            'Code': ['.py', 'Implementation', 'Operator', 'Viewer'],
            'Data': ['.csv', 'Registry', 'Tokens', 'Evidence', 'Ledger'],
            'Roleplay': ['Scene-', 'Roleplay'],
            'Admin': ['README', 'LICENSE', 'NOTICE', 'Runbook', 'Cookbook'],
            'Method': ['Test-set', 'Methods', 'Falsifier', 'Validation'],
            'Paper': ['.pdf', 's41', 'Paper']
        }

    def classify_file(self, filename):
        """Classify file by type based on filename patterns"""
        filename_lower = filename.lower()

        for doc_type, patterns in self.file_types.items():
            for pattern in patterns:
                if pattern.lower() in filename_lower:
                    return doc_type
        return 'Data'  # Default

    def extract_tags(self, filename, doc_type):
        """Extract semantic tags from filename and type"""
        base_tags = []

        # Add type-specific tags
        if doc_type == 'Theory':
            base_tags.extend(['CQE', 'Mathematics', 'Framework'])
        elif doc_type == 'Code':
            base_tags.extend(['Implementation', 'Python'])
        elif doc_type == 'Data':
            base_tags.extend(['Registry', 'CSV'])
        elif doc_type == 'Roleplay':
            base_tags.extend(['Pedagogical', 'Educational'])

        # Extract from filename
        if 'quantum' in filename.lower():
            base_tags.append('Quantum')
        if 'lattice' in filename.lower():
            base_tags.append('Lattice')
        if 'e8' in filename.lower():
            base_tags.append('E8')
        if 'octad' in filename.lower():
            base_tags.append('Octad')

        return ', '.join(set(base_tags))

    def generate_rag_prompt(self, filename, doc_type, title):
        """Generate appropriate RAG retrieval prompt"""
        if doc_type == 'Theory':
            return f"Explain the theoretical concepts in {title}"
        elif doc_type == 'Code':
            return f"Show implementation details from {title}"
        elif doc_type == 'Data':
            return f"List data entries from {title}"
        elif doc_type == 'Roleplay':
            return f"Retrieve pedagogical content from {title}"
        else:
            return f"Summarize contents of {title}"

    def infer_links(self, filename, all_files):
        """Infer semantic links to other files"""
        links = []

        # Scene linking logic
        if filename.startswith('Scene-'):
            scene_num = filename.split('_')[0].replace('Scene-', '')
            try:
                next_scene = f"Scene-{int(scene_num)+1}"
                for f in all_files:
                    if f.startswith(next_scene):
                        links.append(f)
                        break
            except ValueError:
                pass

        # WHY file linking
        if filename.startswith('WHY_'):
            for f in all_files:
                if 'TQF' in f or 'quadratic' in f.lower():
                    links.append(f)
                    break

        # Code to spec linking
        if filename.endswith('.py'):
            spec_name = filename.replace('.py', '.md')
            if spec_name in all_files:
                links.append(spec_name)

        return ', '.join(links[:3])  # Limit to 3 links

    def process_all_files(self, file_list):
        """Process all 328 files into master cross-index"""
        results = []

        for filename in file_list:
            doc_type = self.classify_file(filename)
            title = self.generate_title(filename)
            tags = self.extract_tags(filename, doc_type)
            links = self.infer_links(filename, file_list)
            rag_prompt = self.generate_rag_prompt(filename, doc_type, title)
            summary = self.generate_summary(filename, doc_type)
            created = self.extract_date(filename)

            results.append({
                'file_id': filename,
                'title': title,
                'doc_type': doc_type,
                'tags': tags,
                'linked_to': links,
                'rag_prompt': rag_prompt,
                'summary': summary,
                'created': created
            })

        return pd.DataFrame(results)

    def generate_title(self, filename):
        """Generate human-readable title"""
        title = filename.replace('_', ' ').replace('-', ' ')
        title = title.replace('.pdf', '').replace('.txt', '').replace('.csv', '').replace('.py', '')
        return ' '.join(word.capitalize() for word in title.split())

    def generate_summary(self, filename, doc_type):
        """Generate one-line summary"""
        if doc_type == 'Theory':
            return f"Theoretical framework document for {filename.split('_')[0] if '_' in filename else 'CQE'}"
        elif doc_type == 'Code':
            return f"Implementation of {filename.replace('.py', '').replace('_', ' ')}"
        elif doc_type == 'Data':
            return f"Data registry for {filename.replace('.csv', '').replace('_', ' ')}"
        elif doc_type == 'Roleplay':
            return f"Educational scenario: {filename.split('_')[0] if '_' in filename else 'CQE Introduction'}"
        else:
            return f"Documentation for {filename.split('.')[0]}"

    def extract_date(self, filename):
        """Extract date from filename or use default"""
        import re
        date_match = re.search(r'(\d{6})', filename)
        if date_match:
            date_str = date_match.group(1)
            try:
                year = f"20{date_str[:2]}"
                month = date_str[2:4]
                day = date_str[4:6]
                return f"{year}-{month}-{day}"
            except:
                pass
        return "2025-10-11"  # Default

if __name__ == "__main__":
    processor = CQESystemProcessor()

    # Sample file list (in production, would read all 328 files)
    sample_files = [
        "CQE_Whitepaper_v1.pdf",
        "WHY_1_n4_to_n5_octad.pdf", 
        "Scene-0_250920_091530.txt",
        "cqe_tokens_v2.csv",
        "o8.py"
    ]

    df = processor.process_all_files(sample_files)
    df.to_csv('cqe_master_cross_index_full.csv', index=False)

    print(f"Processed {len(df)} files into master cross-index")
    print("Next steps:")
    print("1. Run with all 328 files")
    print("2. Refine semantic linking")
    print("3. Validate graph connectivity")
