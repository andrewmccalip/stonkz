#!/bin/bash

# 🚀 UV-Based FlaMinGo TimesFM Environment Setup Script
# This script uses UV for fast and reliable dependency management

set -e  # Exit on any error

echo "🚀 Starting UV-Based FlaMinGo TimesFM Environment Setup"
echo "====================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_warning "Running as root. This is not recommended but proceeding anyway..."
   export DEBIAN_FRONTEND=noninteractive
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

print_status "Working directory: $SCRIPT_DIR"

# 1. Ensure UV is installed and in PATH
print_status "Step 1/6: Setting up UV package manager..."
export PATH="/root/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    print_status "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
fi

print_success "UV $(uv --version) is ready"

# 2. Set HuggingFace Token
print_status "Step 2/6: Setting up HuggingFace authentication..."
export HF_TOKEN="${HF_TOKEN:-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}"
print_success "HuggingFace token configured"

# 3. Create UV virtual environment with Python 3.11
print_status "Step 3/6: Creating UV virtual environment with Python 3.11..."

# Remove existing environments
rm -rf .venv flamingo_env flamingo_env_311

# Create new UV environment
uv venv --python 3.11 .venv
print_success "UV virtual environment created with Python 3.11"

# 4. Install core dependencies via UV
print_status "Step 4/6: Installing core dependencies with UV..."

# Install PyTorch with CUDA support first
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other core dependencies
uv pip install \
    transformers==4.49.0 \
    scikit-learn==1.7.1 \
    jax==0.7.1 \
    jaxlib==0.7.1 \
    pandas==2.3.1 \
    "numpy>=1.26.4,<2.0.0" \
    pyarrow==20.0.0 \
    matplotlib==3.10.5 \
    seaborn==0.13.2 \
    python-dotenv==1.1.1 \
    pytz==2025.2 \
    tqdm==4.67.1 \
    "huggingface-hub>=0.20.0" \
    "tokenizers>=0.15.0" \
    "safetensors>=0.4.0" \
    "einshape>=1.0.0" \
    "utilsforecast>=0.1.10" \
    "typer>=0.12.3" \
    "absl-py>=1.4.0"

print_success "Core dependencies installed"

# 5. Install TimesFM from modified local version
print_status "Step 5/6: Installing TimesFM..."

# First, ensure we have the modified TimesFM with Python 3.12 support
if [ ! -d "timesfm_local" ]; then
    print_status "Cloning TimesFM repository..."
    git clone https://github.com/google-research/timesfm.git timesfm_local
    
    # Modify Python version constraint
    sed -i 's/python = ">=3.10,<3.12"/python = ">=3.10,<3.13"/g' timesfm_local/pyproject.toml
fi

# Install TimesFM in editable mode
uv pip install -e ./timesfm_local

print_success "TimesFM installed successfully"

# 6. Create required directories and activation script
print_status "Step 6/6: Setting up project structure..."

mkdir -p model_checkpoints
mkdir -p training_plots
mkdir -p dataset_analysis_plots
mkdir -p dataset_cache
mkdir -p databento/ES

# Create UV activation script
cat > activate_uv.sh << 'EOF'
#!/bin/bash
# Activate UV environment
export PATH="/root/.local/bin:$PATH"
source .venv/bin/activate
export HF_TOKEN="${HF_TOKEN:-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}"

echo "🔥 UV FlaMinGo TimesFM environment activated!"
echo "Python version: $(python --version)"
echo "Available commands:"
echo "  python finetuning.py          - Start finetuning"
echo "  python analyze_dataset.py     - Analyze dataset"
echo "  python finetuned_inference.py - Run inference"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "No GPU detected"
EOF

chmod +x activate_uv.sh

print_success "Project structure created"

# 7. Final verification
print_status "Running final verification..."

# Activate environment and test imports
source .venv/bin/activate

python3 -c "
try:
    import torch
    import transformers
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import jax
    import jaxlib
    from timesfm import TimesFmHparams
    print('✅ All critical imports successful')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU count: {torch.cuda.device_count()}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Create environment info file
cat > uv_environment_info.txt << EOF
UV-Based FlaMinGo TimesFM Environment Setup Complete
==================================================

Setup Date: $(date)
UV Version: $(uv --version)
Python Version: $(python3 --version)
PyTorch Version: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
CUDA Available: $(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "Unknown")

Directory Structure:
- .venv/                      # UV virtual environment
- model_checkpoints/          # Saved model checkpoints
- training_plots/            # Training progress plots
- dataset_analysis_plots/    # Dataset analysis visualizations
- dataset_cache/            # Cached processed data
- databento/ES/            # Dataset location
- timesfm_local/           # Modified TimesFM source

Quick Commands:
- source activate_uv.sh        # Activate environment
- python finetuning.py         # Start training
- python analyze_dataset.py    # Analyze data
- python finetuned_inference.py # Run inference

Files Created:
- activate_uv.sh              # Environment activation
- uv_environment_info.txt     # This file
- pyproject.toml             # UV project configuration
EOF

print_success "Environment verification completed"

echo ""
echo "🎉 UV-Based FlaMinGo TimesFM Environment Setup Complete!"
echo "======================================================"
echo ""
echo "📁 Working directory: $SCRIPT_DIR"
echo "🐍 Python environment: .venv (managed by UV)"
echo "🔥 GPU support: $(python3 -c "import torch; print('✅ Enabled' if torch.cuda.is_available() else '❌ Disabled')" 2>/dev/null || echo '❓ Unknown')"
echo ""
echo "🚀 Next Steps:"
echo "1. Activate environment:    source activate_uv.sh"
echo "2. Upload your dataset to:  databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv"
echo "3. Start training:          python finetuning.py"
echo ""
echo "📖 See uv_environment_info.txt for detailed setup information"
echo ""
print_success "Setup script completed successfully!"

echo ""
echo "💡 Tip: Run 'source activate_uv.sh' to activate the environment"
