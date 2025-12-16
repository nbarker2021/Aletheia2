# Create setup script for generating embeddings
setup_script_code = '''#!/usr/bin/env python3
"""
Setup Script for CQE-MORSR Framework

Generates E₈ embedding and prepares system for operation.
Run this script first after installation.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def setup_embeddings():
    """Generate E₈ embeddings."""
    print("Setting up E₈ embeddings...")
    
    try:
        # Import and run E₈ embedding generator
        from embeddings.e8_embedding import save_embedding
        save_embedding()
        print("✓ E₈ embedding generated successfully")
        
    except Exception as e:
        print(f"✗ Failed to generate E₈ embedding: {e}")
        return False
    
    return True

def setup_directories():
    """Create necessary directories."""
    print("Setting up directories...")
    
    directories = [
        "data/generated",
        "data/cache", 
        "logs",
        "embeddings"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def verify_dependencies():
    """Verify required dependencies are installed."""
    print("Verifying dependencies...")
    
    required_packages = [
        "numpy",
        "scipy", 
        "matplotlib",
        "pytest"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} found")
        except ImportError:
            print(f"✗ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\\nPlease install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main setup function."""
    print("CQE-MORSR Framework Setup")
    print("=" * 40)
    
    # Verify dependencies
    if not verify_dependencies():
        print("\\nSetup failed: missing dependencies")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Generate embeddings
    if not setup_embeddings():
        print("\\nSetup failed: could not generate embeddings")
        sys.exit(1)
    
    print("\\n" + "=" * 40)
    print("Setup complete! CQE-MORSR framework is ready.")
    print("\\nNext steps:")
    print("1. Run tests: python -m pytest tests/")
    print("2. Try examples: python examples/golden_test_harness.py")
    print("3. Generate Niemeier lattices (requires SageMath):")
    print("   sage sage_scripts/generate_niemeier_lattices.sage")

if __name__ == "__main__":
    main()
'''

with open("scripts/setup_embeddings.py", 'w') as f:
    f.write(setup_script_code)

print("Created: scripts/setup_embeddings.py")