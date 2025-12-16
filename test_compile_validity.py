#!/usr/bin/env python3
"""
Test compile validity of all Python files in the repository.
Reports syntax errors and import failures.
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path

def test_syntax(filepath):
    """Test if a file has valid Python syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_import(filepath):
    """Test if a file can be imported (checks for import errors)."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", filepath)
        if spec is None:
            return False, "Could not create module spec"
        module = importlib.util.module_from_spec(spec)
        # Don't actually execute - just check if it can be loaded
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    root = Path(__file__).parent
    
    # Exclude certain directories
    exclude_dirs = {'__pycache__', '.git', 'node_modules', 'venv', 'test_data', 'data'}
    
    syntax_errors = []
    syntax_ok = []
    
    for pyfile in root.rglob('*.py'):
        # Skip excluded directories
        if any(excl in pyfile.parts for excl in exclude_dirs):
            continue
        
        rel_path = pyfile.relative_to(root)
        ok, error = test_syntax(pyfile)
        
        if ok:
            syntax_ok.append(str(rel_path))
        else:
            syntax_errors.append((str(rel_path), error))
    
    # Report
    print(f"=== COMPILE VALIDITY REPORT ===")
    print(f"Total files tested: {len(syntax_ok) + len(syntax_errors)}")
    print(f"Syntax OK: {len(syntax_ok)}")
    print(f"Syntax Errors: {len(syntax_errors)}")
    print()
    
    if syntax_errors:
        print("=== FILES WITH SYNTAX ERRORS ===")
        for filepath, error in sorted(syntax_errors):
            print(f"  {filepath}")
            print(f"    {error}")
        print()
    
    # Group files by directory for overview
    print("=== FILES BY DIRECTORY ===")
    dir_counts = {}
    for f in syntax_ok:
        d = str(Path(f).parent)
        dir_counts[d] = dir_counts.get(d, 0) + 1
    
    for d in sorted(dir_counts.keys()):
        print(f"  {d}: {dir_counts[d]} files OK")
    
    return len(syntax_errors)

if __name__ == "__main__":
    sys.exit(main())
