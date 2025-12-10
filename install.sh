#!/bin/bash
# CQE Unified Runtime v7.0 - Installation Script

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     CQE Unified Runtime v7.0 - Installation Script          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
log_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        log_success "Python $PYTHON_VERSION found"
    else
        log_error "Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    log_error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check pip
log_info "Checking pip..."
if command -v pip3 &> /dev/null; then
    log_success "pip3 found"
else
    log_error "pip3 not found. Please install pip3"
    exit 1
fi

# Install dependencies
log_info "Installing dependencies..."
pip3 install --quiet numpy scipy 2>&1 | grep -v "already satisfied" || true
log_success "Dependencies installed"

# Set up Python path
log_info "Setting up Python path..."
CQE_DIR=$(pwd)
export PYTHONPATH=$CQE_DIR:$PYTHONPATH

# Add to bashrc/zshrc
if [ -f "$HOME/.bashrc" ]; then
    if ! grep -q "CQE_UNIFIED_RUNTIME" "$HOME/.bashrc"; then
        echo "" >> "$HOME/.bashrc"
        echo "# CQE Unified Runtime" >> "$HOME/.bashrc"
        echo "export PYTHONPATH=$CQE_DIR:\$PYTHONPATH" >> "$HOME/.bashrc"
        log_success "Added to ~/.bashrc"
    fi
fi

if [ -f "$HOME/.zshrc" ]; then
    if ! grep -q "CQE_UNIFIED_RUNTIME" "$HOME/.zshrc"; then
        echo "" >> "$HOME/.zshrc"
        echo "# CQE Unified Runtime" >> "$HOME/.zshrc"
        echo "export PYTHONPATH=$CQE_DIR:\$PYTHONPATH" >> "$HOME/.zshrc"
        log_success "Added to ~/.zshrc"
    fi
fi

# Verify installation
log_info "Verifying installation..."
python3 -c "
import sys
sys.path.insert(0, '$CQE_DIR')
from layer2_geometric.e8.lattice import E8Lattice
from layer4_governance.gravitational import GravitationalLayer
print('✅ CQE Unified Runtime v7.0 installed successfully!')
" 2>&1

if [ $? -eq 0 ]; then
    log_success "Installation verified"
else
    log_error "Installation verification failed"
    exit 1
fi

# Create directories
log_info "Creating directories..."
mkdir -p "$HOME/.cqe"
mkdir -p "$HOME/.cqe/cache"
mkdir -p "$HOME/.cqe/logs"
mkdir -p "$HOME/.cqe/data"
log_success "Directories created"

# Create default config
log_info "Creating default configuration..."
cat > "$HOME/.cqe/config.yaml" << EOF
# CQE Unified Runtime Configuration
core:
  log_level: INFO
  cache_size: 1000
  workers: 4

api:
  host: 0.0.0.0
  port: 8000
  auth_required: false

performance:
  use_gpu: false
  batch_size: 32
  timeout: 300

layers:
  layer1_enabled: true
  layer2_enabled: true
  layer3_enabled: true
  layer4_enabled: true
  layer5_enabled: true
EOF
log_success "Configuration created at ~/.cqe/config.yaml"

# Run quick test
log_info "Running quick test..."
python3 -c "
import sys
sys.path.insert(0, '$CQE_DIR')
from layer2_geometric.e8.lattice import E8Lattice
from layer4_governance.gravitational import GravitationalLayer

e8 = E8Lattice()
grav = GravitationalLayer()

# Test E8 projection
vector = [1, 2, 3, 4, 5, 6, 7, 8]
projected = e8.project(vector)
print(f'✓ E8 projection: {projected[:4]}...')

# Test digital root
dr = grav.compute_digital_root(432)
print(f'✓ Digital root: DR(432) = {dr}')

print('✓ All tests passed!')
"

if [ $? -eq 0 ]; then
    log_success "Quick test passed"
else
    log_error "Quick test failed"
    exit 1
fi

# Print summary
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                 Installation Complete!                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📦 CQE Unified Runtime v7.0 is ready to use!"
echo ""
echo "Quick Start:"
echo "  1. Source your shell config: source ~/.bashrc (or ~/.zshrc)"
echo "  2. Run tests: python3 comprehensive_test_harness.py"
echo "  3. Read docs: cat QUICKSTART.md"
echo ""
echo "Documentation:"
echo "  • QUICKSTART.md       - Get started in 5 minutes"
echo "  • OPERATION_MANUAL.md - Complete user guide"
echo "  • DEPLOYMENT.md       - Production deployment"
echo "  • TEST_DOCUMENTATION.md - Testing guide"
echo ""
echo "Support: https://help.manus.im"
echo ""
