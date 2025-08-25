#!/bin/bash

# =============================================================================
# UV Environment Setup Script for TimesFM Stock Market Finetuning
# =============================================================================
# This script recreates the complete development environment using UV
# Author: Andrew McCalip <andrewmccalip@gmail.com>
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.12"
PROJECT_NAME="flamingo-timesfm-finetuning"
GIT_USER_NAME="Andrew McCalip"
GIT_USER_EMAIL="andrewmccalip@gmail.com"

echo -e "${BLUE}🚀 Setting up TimesFM Stock Market Finetuning Environment${NC}"
echo -e "${BLUE}================================================================${NC}"

# Function to print status messages
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    print_error "UV is not installed. Please install UV first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

print_status "UV is installed: $(uv --version)"

# Check if we're in the right directory
if [[ ! -f "pytorch_timesfm_finetune.py" ]]; then
    print_error "This script should be run from the project root directory (where pytorch_timesfm_finetune.py exists)"
    exit 1
fi

print_status "Running from correct project directory"

# Git configuration
print_info "Configuring Git..."
git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"
git config --global init.defaultBranch main
git config --global pull.rebase false
print_status "Git configured for $GIT_USER_NAME <$GIT_USER_EMAIL>"

# Initialize git repository if not already initialized
if [[ ! -d ".git" ]]; then
    print_info "Initializing Git repository..."
    git init
    print_status "Git repository initialized"
else
    print_status "Git repository already exists"
fi

# Create or update .gitignore
print_info "Creating/updating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
venv310/
.uv/

# PyTorch
*.pth
*.pt
*.ckpt

# Data and caches
dataset_cache/
cache/
*.csv
*.parquet
*.h5
*.hdf5

# Model checkpoints and outputs
model_checkpoints/
finetune_checkpoints/
*.log
training_output.log

# Plots and visualizations
plots/
finetune_plots/
stock_plots/
*.png
*.jpg
*.jpeg
*.pdf

# Jupyter Notebook
.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Environment files
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
finetuning.env

# Temporary files
tmp/
temp/
*.tmp
current_requirements.txt

# Databento data (large files)
databento/
*.ohlcv-1m.csv

# HuggingFace cache
.cache/
transformers_cache/
EOF

print_status ".gitignore created/updated"

# Remove existing virtual environment if it exists
if [[ -d "venv310" ]]; then
    print_warning "Removing existing venv310 directory..."
    rm -rf venv310
fi

# Create UV project if pyproject.toml doesn't exist or needs updating
print_info "Setting up UV project..."

# Create comprehensive pyproject.toml
cat > pyproject.toml << 'EOF'
[project]
name = "flamingo-timesfm-finetuning"
version = "0.1.0"
description = "FlaMinGo TimesFM Stock Market Finetuning with PyTorch"
authors = [
    {name = "Andrew McCalip", email = "andrewmccalip@gmail.com"}
]
readme = "README.md"
requires-python = ">=3.12,<3.13"
license = {text = "MIT"}
keywords = ["timesfm", "stock-market", "forecasting", "pytorch", "machine-learning"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    # Core ML/AI packages
    "torch>=2.5.0",
    "transformers==4.49.0",
    "scikit-learn==1.7.1",
    "accelerate>=1.10.0",
    
    # JAX (required for TimesFM backend)
    "jax==0.7.1",
    "jaxlib==0.7.1",
    
    # Data processing
    "pandas==2.3.1",
    "numpy>=1.26.4,<2.0.0",
    "pyarrow==20.0.0",
    
    # Plotting and visualization
    "matplotlib==3.10.5",
    "seaborn==0.13.2",
    
    # Utilities
    "python-dotenv==1.1.1",
    "pytz==2025.2",
    "tqdm==4.67.1",
    "pyyaml>=6.0.0",
    "requests>=2.32.0",
    
    # HuggingFace ecosystem
    "huggingface-hub>=0.33.0",
    "tokenizers>=0.21.0",
    "safetensors>=0.5.0",
    
    # TimesFM dependencies
    "einshape>=1.0.0",
    "utilsforecast>=0.2.10",
    "typer>=0.16.0",
    "absl-py>=2.3.0",
    
    # Additional ML utilities
    "wandb>=0.21.0",
    "protobuf>=6.32.0",
    "sentencepiece>=0.2.0",
    
    # Development and debugging
    "ipython>=9.3.0",
    "ipykernel>=6.29.0",
    "ipywidgets>=8.1.0",
    "rich>=14.1.0",
    "inquirerpy>=0.3.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.2",
    "pytest-cov>=5.0.0",
    "black>=24.0.0",
    "isort>=5.13.0",
    "flake8>=7.0.0",
    "mypy>=1.8.0",
]

gpu = [
    # CUDA-specific packages (will be installed automatically with torch+cu121)
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12", 
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu12",
    "triton>=3.1.0",
]

[project.urls]
Homepage = "https://github.com/andrewmccalip/flamingo-timesfm-finetuning"
Repository = "https://github.com/andrewmccalip/flamingo-timesfm-finetuning"
Issues = "https://github.com/andrewmccalip/flamingo-timesfm-finetuning/issues"

[project.scripts]
finetune = "pytorch_timesfm_finetune:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.3.2",
    "pytest-cov>=5.0.0",
    "black>=24.0.0",
    "isort>=5.13.0",
    "flake8>=7.0.0",
    "mypy>=1.8.0",
]

[tool.black]
line-length = 100
target-version = ['py312']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true

[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
EOF

print_status "pyproject.toml created"

# Initialize UV project
print_info "Initializing UV project with Python $PYTHON_VERSION..."
uv init --python $PYTHON_VERSION --no-readme

# Install dependencies
print_info "Installing dependencies with UV..."
uv sync --all-extras

print_status "Core dependencies installed"

# Install PyTorch with CUDA support explicitly
print_info "Installing PyTorch with CUDA support..."
uv add "torch>=2.5.0" --index-url https://download.pytorch.org/whl/cu121
uv add "torchaudio>=2.5.0" --index-url https://download.pytorch.org/whl/cu121  
uv add "torchvision>=0.20.0" --index-url https://download.pytorch.org/whl/cu121

print_status "PyTorch with CUDA support installed"

# Install TimesFM from git (since it's not on PyPI)
print_info "Installing TimesFM from GitHub..."
uv add "git+https://github.com/google-research/timesfm.git"

print_status "TimesFM installed from GitHub"

# Create activation script
print_info "Creating activation script..."
cat > activate_uv.sh << 'EOF'
#!/bin/bash
# Activation script for UV environment
# Usage: source activate_uv.sh

echo "🚀 Activating UV environment for TimesFM Stock Market Finetuning..."

# Check if we're in the right directory
if [[ ! -f "pytorch_timesfm_finetune.py" ]]; then
    echo "❌ Error: Run this script from the project root directory"
    return 1
fi

# Activate UV environment
source .venv/bin/activate

# Set environment variables
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Display environment info
echo "✅ Environment activated!"
echo "📍 Project: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "🔥 PyTorch: $(python -c 'import torch; print(f"v{torch.__version__} (CUDA: {torch.cuda.is_available()})")')"
echo "🤖 Transformers: $(python -c 'import transformers; print(f"v{transformers.__version__}")')"

# Check GPU availability
if python -c "import torch; print('🎮 GPU Available:', torch.cuda.is_available())"; then
    python -c "import torch; print('🎮 GPU Count:', torch.cuda.device_count())"
    python -c "import torch; print('🎮 GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
fi

echo ""
echo "🎯 Ready to run:"
echo "   python pytorch_timesfm_finetune.py --help"
echo "   python pytorch_timesfm_finetune.py --epochs 100 --batch-size 32"
echo ""
EOF

chmod +x activate_uv.sh
print_status "Activation script created (activate_uv.sh)"

# Create environment file template
print_info "Creating environment configuration template..."
cat > .env.template << 'EOF'
# Environment Configuration for TimesFM Stock Market Finetuning
# Copy this file to .env and customize as needed

# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false

# Model Configuration
TIMESFM_MODEL_REPO=google/timesfm-1.0-200m-pytorch
HUGGINGFACE_HUB_CACHE=.cache/huggingface

# Training Configuration
BATCH_SIZE=64
LEARNING_RATE=1e-5
NUM_EPOCHS=5000

# Data Configuration
DATASET_PATH=databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv
CONTEXT_LENGTH=448
HORIZON_LENGTH=64

# Logging and Monitoring
WANDB_PROJECT=timesfm-stock-finetuning
WANDB_ENTITY=your-wandb-username

# Output Directories
CHECKPOINT_DIR=finetune_checkpoints
PLOT_DIR=finetune_plots
CACHE_DIR=dataset_cache
EOF

print_status "Environment template created (.env.template)"

# Create README if it doesn't exist
if [[ ! -f "README.md" ]]; then
    print_info "Creating README.md..."
    cat > README.md << 'EOF'
# FlaMinGo TimesFM Stock Market Finetuning

A comprehensive PyTorch implementation for fine-tuning Google's TimesFM model on stock market data.

## Quick Start

1. **Setup Environment:**
   ```bash
   ./setup_environment.sh
   source activate_uv.sh
   ```

2. **Run Training:**
   ```bash
   python pytorch_timesfm_finetune.py --epochs 100 --batch-size 32
   ```

## Features

- 🚀 PyTorch implementation of TimesFM fine-tuning
- 📊 Comprehensive visualization and monitoring
- 🎯 Directional accuracy optimization
- 💾 Automatic checkpointing and resuming
- 🔥 CUDA acceleration support
- 📈 Real-time training plots

## Environment

- Python 3.12
- PyTorch 2.5+ with CUDA 12.1
- TimesFM (Google Research)
- UV package manager

## Author

Andrew McCalip <andrewmccalip@gmail.com>
EOF
    print_status "README.md created"
fi

# Create development scripts
print_info "Creating development scripts..."

# Training script
cat > run_training.sh << 'EOF'
#!/bin/bash
# Quick training script with common configurations

set -e

echo "🚀 Starting TimesFM Stock Market Fine-tuning..."

# Activate environment
source activate_uv.sh

# Default parameters (can be overridden)
EPOCHS=${1:-100}
BATCH_SIZE=${2:-32}
PLOTS_PER_EPOCH=${3:-4}

echo "📊 Configuration:"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: $BATCH_SIZE" 
echo "   Plots per Epoch: $PLOTS_PER_EPOCH"
echo ""

# Run training
python pytorch_timesfm_finetune.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --plots-per-epoch $PLOTS_PER_EPOCH \
    --keep-plots

echo "✅ Training completed!"
EOF

chmod +x run_training.sh

# Quick test script
cat > test_environment.sh << 'EOF'
#!/bin/bash
# Test script to verify environment setup

set -e

echo "🧪 Testing TimesFM Environment Setup..."

# Activate environment
source activate_uv.sh

echo ""
echo "📦 Testing package imports..."

python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
except ImportError as e:
    print(f'❌ PyTorch: {e}')

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except ImportError as e:
    print(f'❌ Transformers: {e}')

try:
    import timesfm
    print(f'✅ TimesFM: Available')
except ImportError as e:
    print(f'❌ TimesFM: {e}')

try:
    import pandas as pd
    print(f'✅ Pandas: {pd.__version__}')
except ImportError as e:
    print(f'❌ Pandas: {e}')

try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except ImportError as e:
    print(f'❌ NumPy: {e}')

try:
    import matplotlib
    print(f'✅ Matplotlib: {matplotlib.__version__}')
except ImportError as e:
    print(f'❌ Matplotlib: {e}')

try:
    import jax
    print(f'✅ JAX: {jax.__version__}')
except ImportError as e:
    print(f'❌ JAX: {e}')
"

echo ""
echo "🎮 GPU Information:"
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
else:
    print('No CUDA GPUs available')
"

echo ""
echo "✅ Environment test completed!"
EOF

chmod +x test_environment.sh

print_status "Development scripts created"

# Final verification
print_info "Running environment verification..."
source .venv/bin/activate

# Test critical imports
python -c "
import torch
import transformers  
import timesfm
import pandas
import numpy
import matplotlib
print('✅ All critical packages imported successfully')
print(f'PyTorch CUDA: {torch.cuda.is_available()}')
"

print_status "Environment verification completed"

# Clean up temporary files
rm -f current_requirements.txt

echo ""
echo -e "${GREEN}🎉 Environment setup completed successfully!${NC}"
echo -e "${BLUE}================================================================${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Activate the environment:"
echo "   source activate_uv.sh"
echo ""
echo "2. Test the environment:"
echo "   ./test_environment.sh"
echo ""
echo "3. Start training:"
echo "   ./run_training.sh 100 32 4"
echo "   # Or manually:"
echo "   python pytorch_timesfm_finetune.py --epochs 100 --batch-size 32"
echo ""
echo "4. View training progress:"
echo "   # Plots are saved to finetune_plots/latest.png"
echo ""
echo -e "${GREEN}Environment ready for TimesFM stock market fine-tuning! 🚀${NC}"
