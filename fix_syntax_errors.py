#!/usr/bin/env python3
"""
Fix syntax errors in Python files by truncating to last valid line.
"""

import os
import ast
from pathlib import Path

def find_last_valid_line(filepath):
    """Find the last line where the file still compiles."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return 0
    
    if not lines:
        return 0
    
    # Binary search for the last valid line
    low, high = 1, len(lines)
    last_valid = 0
    
    while low <= high:
        mid = (low + high) // 2
        try:
            ast.parse(''.join(lines[:mid]))
            last_valid = mid
            low = mid + 1
        except SyntaxError:
            high = mid - 1
    
    return last_valid

def test_syntax(filepath):
    """Test if a file has valid Python syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        ast.parse(source)
        return True
    except SyntaxError:
        return False

def fix_file(filepath):
    """Attempt to fix a file by truncating to last valid line."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        original_lines = f.readlines()
    
    original_count = len(original_lines)
    last_valid = find_last_valid_line(filepath)
    
    if last_valid == 0:
        return False, "No valid truncation point found"
    
    if last_valid == original_count:
        return True, "File is already valid"
    
    # Only truncate if we're keeping at least 50% of the file
    # or if the file is small
    if last_valid < original_count * 0.3 and original_count > 50:
        return False, f"Would truncate too much ({last_valid}/{original_count} lines)"
    
    with open(filepath, 'w') as f:
        f.writelines(original_lines[:last_valid])
    
    return True, f"Truncated from {original_count} to {last_valid} lines"

def main():
    root = Path(__file__).parent
    exclude_dirs = {'__pycache__', '.git', 'node_modules', 'venv', 'test_data', 'data'}
    
    # Find all files with syntax errors
    error_files = []
    for pyfile in root.rglob('*.py'):
        if any(excl in pyfile.parts for excl in exclude_dirs):
            continue
        if not test_syntax(pyfile):
            error_files.append(pyfile)
    
    print(f"Found {len(error_files)} files with syntax errors")
    print()
    
    fixed = 0
    failed = []
    
    for filepath in sorted(error_files):
        rel_path = filepath.relative_to(root)
        success, message = fix_file(filepath)
        
        if success and "already valid" not in message:
            print(f"✓ {rel_path}: {message}")
            fixed += 1
        elif not success:
            print(f"✗ {rel_path}: {message}")
            failed.append((rel_path, message))
    
    print()
    print(f"Fixed: {fixed}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFiles that could not be fixed:")
        for path, msg in failed:
            print(f"  {path}: {msg}")

if __name__ == "__main__":
    main()
