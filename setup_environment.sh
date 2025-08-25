#!/bin/bash

# 🚀 FlaMinGo TimesFM Stock Finetuning Environment Setup Script
# This script sets up everything needed for stock market finetuning on a new GPU instance

set -e  # Exit on any error

echo "🚀 Starting FlaMinGo TimesFM Stock Finetuning Environment Setup"
echo "================================================================"

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

# 1. System Updates and Dependencies
print_status "Step 1/8: Updating system and installing dependencies..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install essential packages
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    htop \
    tmux \
    vim \
    unzip

print_success "System dependencies installed"

# 2. NVIDIA Drivers and CUDA (if not already installed)
print_status "Step 2/8: Checking NVIDIA GPU and CUDA installation..."

if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA drivers already installed"
    nvidia-smi
else
    print_warning "NVIDIA drivers not found. Installing..."
    
    # Add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    
    # Install NVIDIA drivers and CUDA
    sudo apt-get install -y nvidia-driver-535 cuda-toolkit-12-2
    
    print_warning "NVIDIA drivers installed. You may need to reboot the system."
    print_warning "After reboot, run this script again to continue setup."
fi

# Check CUDA installation
if command -v nvcc &> /dev/null; then
    print_success "CUDA toolkit found: $(nvcc --version | grep release)"
else
    print_error "CUDA toolkit not found. Please install CUDA manually."
    exit 1
fi

# 3. Python Virtual Environment Setup
print_status "Step 3/8: Setting up Python virtual environment..."

VENV_NAME="flamingo_env"

if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment '$VENV_NAME' already exists. Removing..."
    rm -rf "$VENV_NAME"
fi

python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

print_success "Virtual environment '$VENV_NAME' created and activated"

# 4. Install PyTorch with CUDA support
print_status "Step 4/8: Installing PyTorch with CUDA support..."

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA installation
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

print_success "PyTorch with CUDA support installed"

# 5. Install Core Dependencies
print_status "Step 5/8: Installing core Python dependencies..."

# Install from requirements-minimal.txt if it exists, otherwise install manually
if [ -f "requirements-minimal.txt" ]; then
    print_status "Installing from requirements-minimal.txt..."
    pip install -r requirements-minimal.txt
else
    print_status "Installing core dependencies manually..."
    
    # Core ML/AI packages
    pip install transformers==4.49.0
    pip install scikit-learn==1.7.1
    
    # JAX (required for TimesFM backend)
    pip install jax==0.7.1 jaxlib==0.7.1
    
    # Data processing
    pip install pandas==2.3.1
    pip install numpy==1.26.4
    pip install pyarrow==20.0.0
    
    # Plotting and visualization
    pip install matplotlib==3.10.5
    pip install seaborn==0.13.2
    
    # Utilities
    pip install python-dotenv==1.1.1
    pip install pytz==2025.2
    pip install tqdm==4.67.1
    
    # HuggingFace ecosystem
    pip install huggingface-hub>=0.20.0
    pip install tokenizers>=0.15.0
    pip install safetensors>=0.4.0
fi

print_success "Core dependencies installed"

# 6. Install FlaMinGo TimesFM
print_status "Step 6/8: Installing FlaMinGo TimesFM..."

# Install FlaMinGo TimesFM directly from HuggingFace (no local clone needed)
print_status "Installing FlaMinGo TimesFM from HuggingFace..."
pip install -e "git+https://huggingface.co/PartAI/FlaMinGo-timesfm@bc45674bf0c36dc324731151a40af7d82b9e8046#egg=timesfm"

print_success "FlaMinGo TimesFM installed"

# 7. Create Required Directories
print_status "Step 7/8: Creating required directories..."

mkdir -p model_checkpoints
mkdir -p training_plots
mkdir -p dataset_analysis_plots
mkdir -p dataset_cache

print_success "Required directories created"

# 8. Download Sample Data (if not present)
print_status "Step 8/8: Checking for dataset..."

DATASET_PATH="databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv"

if [ ! -f "$DATASET_PATH" ]; then
    print_warning "Large dataset not found at $DATASET_PATH"
    print_status "You'll need to upload your dataset to this location manually."
    
    # Create the directory structure
    mkdir -p "$(dirname "$DATASET_PATH")"
    
    print_status "Created directory structure for dataset"
else
    print_success "Dataset found at $DATASET_PATH"
    
    # Show dataset info
    DATASET_SIZE=$(du -h "$DATASET_PATH" | cut -f1)
    print_status "Dataset size: $DATASET_SIZE"
fi

# 9. Create activation script
print_status "Creating environment activation script..."

cat > activate_flamingo.sh << 'EOF'
#!/bin/bash
# Activate FlaMinGo environment
source flamingo_env/bin/activate
echo "🔥 FlaMinGo TimesFM environment activated!"
echo "Available commands:"
echo "  python finetuning.py          - Start finetuning"
echo "  python analyze_dataset.py     - Analyze dataset"
echo "  python finetuned_inference.py - Run inference"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
EOF

chmod +x activate_flamingo.sh

# 10. Create quick start script
print_status "Creating quick start script..."

cat > quick_start.sh << 'EOF'
#!/bin/bash
# Quick start script for FlaMinGo finetuning

echo "🚀 FlaMinGo TimesFM Quick Start"
echo "==============================="

# Activate environment
source flamingo_env/bin/activate

echo "1. Analyzing dataset..."
python analyze_dataset.py

echo ""
echo "2. Starting finetuning with progress tracking..."
python finetuning.py

echo ""
echo "3. Running inference demo..."
python finetuned_inference.py

echo "✅ Quick start completed!"
EOF

chmod +x quick_start.sh

# 11. Final verification
print_status "Running final verification..."

# Test imports
python3 -c "
try:
    import torch
    import transformers
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from timesfm import TimesFmHparams
    from timesfm.timesfm_torch import TimesFmTorch
    print('✅ All critical imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Create environment info file
cat > environment_info.txt << EOF
FlaMinGo TimesFM Environment Setup Complete
==========================================

Setup Date: $(date)
Python Version: $(python3 --version)
PyTorch Version: $(python3 -c "import torch; print(torch.__version__)")
CUDA Available: $(python3 -c "import torch; print(torch.cuda.is_available())")
GPU Count: $(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

Directory Structure:
- flamingo_env/              # Virtual environment
- model_checkpoints/         # Saved model checkpoints
- training_plots/           # Training progress plots
- dataset_analysis_plots/   # Dataset analysis visualizations
- dataset_cache/           # Cached processed data
- databento/ES/           # Dataset location

Quick Commands:
- source activate_flamingo.sh    # Activate environment
- ./quick_start.sh               # Run full pipeline
- python finetuning.py           # Start training
- python analyze_dataset.py      # Analyze data
- python finetuned_inference.py  # Run inference

Files Created:
- activate_flamingo.sh      # Environment activation
- quick_start.sh           # Quick start pipeline
- environment_info.txt     # This file
EOF

print_success "Environment verification completed"

echo ""
echo "🎉 FlaMinGo TimesFM Environment Setup Complete!"
echo "=============================================="
echo ""
echo "📁 Working directory: $SCRIPT_DIR"
echo "🐍 Virtual environment: $VENV_NAME"
echo "🔥 GPU support: $(python3 -c "import torch; print('✅ Enabled' if torch.cuda.is_available() else '❌ Disabled')")"
echo ""
echo "🚀 Next Steps:"
echo "1. Activate environment:    source activate_flamingo.sh"
echo "2. Upload your dataset to:  databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv"
echo "3. Run quick start:         ./quick_start.sh"
echo "   OR"
echo "4. Start training:          python finetuning.py"
echo ""
echo "📖 See environment_info.txt for detailed setup information"
echo ""
print_success "Setup script completed successfully!"

# Deactivate virtual environment
deactivate 2>/dev/null || true

echo ""
echo "💡 Tip: Run 'source activate_flamingo.sh' to activate the environment"
