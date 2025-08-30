#!/bin/bash
# ==============================================================================
# TimesFM Stock Prediction Project Setup Script - Linux Version
# ==============================================================================
# This script sets up the entire project environment from scratch on Linux
# Run with: chmod +x setup_project.sh && ./setup_project.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.10"
VENV_NAME="venv310"
GIT_EMAIL="andrewmccalip@gmail.com"
GIT_NAME="Andrew McCalip"

# ==============================================================================
# Function Definitions
# ==============================================================================

print_header() {
    echo -e "${GREEN}🚀 TimesFM Stock Prediction Project Setup${NC}"
    echo -e "${YELLOW}$(printf '%.0s=' {1..60})${NC}"
}

print_step() {
    echo -e "\n${CYAN}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

check_command() {
    command -v "$1" >/dev/null 2>&1
}

get_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif [ -f /etc/debian_version ]; then
        echo "debian"
    elif [ -f /etc/redhat-release ]; then
        echo "rhel"
    else
        echo "unknown"
    fi
}

# ==============================================================================
# Prerequisites Check
# ==============================================================================

print_step "Checking Prerequisites..."

# Check if running as root (warn if yes)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. This is not recommended for development."
fi

# Check Python installation
if check_command python3; then
    PYTHON_CMD="python3"
    PYTHON_VERSION_OUTPUT=$(python3 --version)
    print_success "Python found: $PYTHON_VERSION_OUTPUT"
elif check_command python; then
    PYTHON_CMD="python"
    PYTHON_VERSION_OUTPUT=$(python --version)
    print_success "Python found: $PYTHON_VERSION_OUTPUT"
else
    print_error "Python not found. Please install Python $PYTHON_VERSION first."
    echo "Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "Arch: sudo pacman -S python python-pip"
    exit 1
fi

# Check Python version
PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
    print_success "Python version $PYTHON_MAJOR.$PYTHON_MINOR is compatible"
else
    print_warning "Python version $PYTHON_MAJOR.$PYTHON_MINOR detected. Recommended: Python $PYTHON_VERSION+"
fi

# Check Git installation
if check_command git; then
    GIT_VERSION=$(git --version)
    print_success "Git found: $GIT_VERSION"
else
    print_error "Git not found. Installing Git..."

    DISTRO=$(get_distro)
    case $DISTRO in
        ubuntu|debian)
            sudo apt update && sudo apt install -y git
            ;;
        centos|rhel|fedora)
            sudo yum install -y git || sudo dnf install -y git
            ;;
        arch)
            sudo pacman -S --noconfirm git
            ;;
        *)
            print_error "Unknown distribution. Please install Git manually."
            exit 1
            ;;
    esac

    if check_command git; then
        print_success "Git installed successfully"
    else
        print_error "Failed to install Git. Please install it manually."
        exit 1
    fi
fi

# ==============================================================================
# Vast.ai Configuration (Disable Tmux)
# ==============================================================================

print_step "Configuring Vast.ai settings..."

# Disable automatic tmux session on Vast.ai
if [ ! -f ~/.no_auto_tmux ]; then
    touch ~/.no_auto_tmux
    print_success "Disabled automatic tmux session on Vast.ai"
else
    print_success "Vast.ai tmux already disabled"
fi

# ==============================================================================
# Git Configuration
# ==============================================================================

print_step "Setting up Git configuration..."

git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

# Set some useful Git defaults
git config --global core.autocrlf input
git config --global core.editor "nano"
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global credential.helper store

print_success "Git configuration set for $GIT_NAME <$GIT_EMAIL>"

# ==============================================================================
# Virtual Environment Setup
# ==============================================================================

print_step "Setting up Python virtual environment..."

# Remove existing venv if it exists
if [ -d "$VENV_NAME" ]; then
    print_warning "Removing existing virtual environment..."
    rm -rf "$VENV_NAME"
fi

# Create new virtual environment
print_step "Creating virtual environment: $VENV_NAME"
$PYTHON_CMD -m venv "$VENV_NAME"

if [ $? -ne 0 ]; then
    print_error "Failed to create virtual environment. Installing python3-venv..."
    DISTRO=$(get_distro)
    case $DISTRO in
        ubuntu|debian)
            sudo apt install -y python3-venv
            ;;
        centos|rhel|fedora)
            sudo yum install -y python3-venv || sudo dnf install -y python3-venv
            ;;
        arch)
            sudo pacman -S --noconfirm python-virtualenv
            ;;
    esac
    $PYTHON_CMD -m venv "$VENV_NAME"
fi

print_success "Virtual environment created successfully"

# Activate virtual environment
print_step "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip

# ==============================================================================
# Install System Dependencies
# ==============================================================================

print_step "Installing system dependencies..."

DISTRO=$(get_distro)
case $DISTRO in
    ubuntu|debian)
        print_step "Installing Ubuntu/Debian dependencies..."
        sudo apt update
        sudo apt install -y \
            build-essential \
            cmake \
            git \
            curl \
            wget \
            libssl-dev \
            libffi-dev \
            python3-dev \
            libjpeg-dev \
            zlib1g-dev \
            libpng-dev \
            libfreetype6-dev \
            liblcms2-dev \
            libopenjp2-7 \
            libtiff5 \
            libxcb1 \
            libxcb1-dev \
            libx11-xcb1 \
            libxss1 \
            libasound2 \
            libgtk-3-0 \
            libgbm1 \
            libxcomposite1 \
            libxcursor1 \
            libxdamage1 \
            libxi6 \
            libxtst6 \
            libnss3 \
            libcups2 \
            libxrandr2 \
            libgconf-2-4 \
            libxss1 \
            libappindicator1 \
            libindicator7 \
            fonts-liberation \
            lsb-release \
            xdg-utils
        ;;
    centos|rhel|fedora)
        print_step "Installing CentOS/RHEL/Fedora dependencies..."
        sudo yum groupinstall -y "Development Tools" || sudo dnf groupinstall -y "Development Tools"
        sudo yum install -y python3-devel openssl-devel libffi-devel || \
        sudo dnf install -y python3-devel openssl-devel libffi-devel
        ;;
    arch)
        print_step "Installing Arch Linux dependencies..."
        sudo pacman -S --noconfirm \
            base-devel \
            cmake \
            git \
            curl \
            wget \
            openssl \
            libffi \
            python \
            libjpeg-turbo \
            zlib \
            libpng \
            freetype2 \
            lcms2 \
            openjpeg2 \
            libtiff \
            xcb-util \
            xcb-util-wm \
            xcb-util-keysyms \
            alsa-lib \
            gtk3 \
            libxcomposite \
            libxcursor \
            libxdamage \
            libxi \
            libxtst \
            nss \
            cups \
            libxrandr \
            gconf \
            libxss \
            libappindicator-gtk3 \
            liberation-fonts \
            lsb-release \
            xdg-utils
        ;;
    *)
        print_warning "Unknown distribution. Skipping system dependency installation."
        ;;
esac

print_success "System dependencies installation completed"

# ==============================================================================
# Install Core Dependencies
# ==============================================================================

print_step "Installing core Python dependencies..."

CORE_PACKAGES=(
    "torch>=2.0.0"
    "torchvision"
    "torchaudio"
    "numpy"
    "pandas"
    "matplotlib"
    "seaborn"
    "scikit-learn"
    "scipy"
    "jupyter"
    "notebook"
    "ipykernel"
    "tqdm"
    "requests"
    "python-dotenv"
    "psutil"
    "plotly"
    "kaleido"
    "Pillow"
    "opencv-python"
    "h5py"
    "tables"
    "openpyxl"
    "sqlalchemy"
    "pymysql"
    "psycopg2-binary"
    "redis"
    "fastapi"
    "uvicorn"
    "pydantic"
    "python-multipart"
    "aiofiles"
    "tensorboard"
)

for package in "${CORE_PACKAGES[@]}"; do
    echo "Installing $package..."
    pip install "$package" || print_warning "Failed to install $package"
done

# ==============================================================================
# Install ML/AI Specific Packages
# ==============================================================================

print_step "Installing ML/AI specific packages..."

ML_PACKAGES=(
    "transformers"
    "datasets"
    "tokenizers"
    "accelerate"
    "peft"
    "bitsandbytes"
    "optimum"
    "onnx"
    "onnxruntime"
    "jax"
    "jaxlib"
    "flax"
    "dm-haiku"
    "optax"
    "einops"
    "timm"
    "fairscale"
    "deepspeed"
    "pytorch-lightning"
    "lightning"
    "huggingface-hub"
    "diffusers"
    "invisible-watermark"
    "safetensors"
)

for package in "${ML_PACKAGES[@]}"; do
    echo "Installing $package..."
    pip install "$package" || print_warning "Failed to install $package"
done

# ==============================================================================
# Install TimesFM and Related Packages
# ==============================================================================

print_step "Installing TimesFM and time series packages..."

TIMESERIES_PACKAGES=(
    "timesfm[torch]"
    "yfinance"
    "pandas-ta"
    "ta"
    "pandas-datareader"
    "fredapi"
    "quandl"
    "alpha-vantage"
    "pyportfolioopt"
    "cvxpy"
    "backtrader"
    "zipline"
    "pyfolio"
    "empyrical"
    "ffn"
    "quantstats"
)

for package in "${TIMESERIES_PACKAGES[@]}"; do
    echo "Installing $package..."
    pip install "$package" || print_warning "Failed to install $package (might require special setup)"
done

# Try to install TA-Lib separately (often problematic)
echo "Installing TA-Lib..."
pip install ta-lib || print_warning "TA-Lib installation failed. This is common on some systems."

# ==============================================================================
# Install Development Tools
# ==============================================================================

print_step "Installing development and testing tools..."

DEV_PACKAGES=(
    "black"
    "isort"
    "flake8"
    "mypy"
    "pylint"
    "pytest"
    "pytest-cov"
    "pytest-xdist"
    "pytest-html"
    "pytest-mock"
    "coverage"
    "tox"
    "pre-commit"
    "commitizen"
    "mkdocs"
    "mkdocs-material"
    "mkdocstrings"
    "jupyter-book"
    "sphinx"
    "nbsphinx"
    "sphinx-rtd-theme"
)

for package in "${DEV_PACKAGES[@]}"; do
    echo "Installing $package..."
    pip install "$package" || print_warning "Failed to install $package"
done

# ==============================================================================
# Create Jupyter Kernel
# ==============================================================================

print_step "Setting up Jupyter kernel..."

# Create IPython kernel for the virtual environment
python -m ipykernel install --user --name="$VENV_NAME" --display-name="Python ($VENV_NAME)"

if [ $? -eq 0 ]; then
    print_success "Jupyter kernel created: $VENV_NAME"
else
    print_warning "Failed to create Jupyter kernel"
fi

# ==============================================================================
# Create Environment Activation Script
# ==============================================================================

print_step "Creating environment activation helper..."

cat > activate.sh << 'EOF'
#!/bin/bash
# TimesFM Stock Prediction Environment Activation Script
# Run with: source activate.sh

echo "🚀 Activating TimesFM Stock Prediction Environment"
echo "Environment: venv310"
echo "Git User: Andrew McCalip <andrewmccalip@gmail.com>"
echo ""

# Activate virtual environment
source ./venv310/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    echo "Make sure venv310 exists. Run ./setup_project.sh first."
    return 1 2>/dev/null || exit 1
fi

# Set environment variables
export PYTHONPATH="$PWD:$PWD/timesfm/src"
export CUDA_VISIBLE_DEVICES="0"  # Use GPU 0 by default (change if needed)

echo "✅ Environment activated!"
echo ""
echo "Available commands:"
echo "  python finetune_timesfm2.py               # Start fine-tuning"
echo "  tensorboard --logdir=tensorboard_logs    # Start TensorBoard"
echo "  jupyter notebook                         # Start Jupyter"
echo "  python test_setup.py                     # Test installation"
echo "  pytest                                   # Run tests"
echo ""

# Show current directory contents (excluding venv and cache files)
echo "Current project files:"
ls -la | grep -E '^-' | grep -v -E '\.(pyc|pyo)$' | grep -v __pycache__ | grep -v venv310 | awk '{print "  " $9}'

echo ""
echo "🚀 Ready for stock prediction development!"

# Optional: Add some useful aliases
alias activate="source ./venv310/bin/activate"
alias tb="tensorboard --logdir=tensorboard_logs"
alias jn="jupyter notebook"
EOF

chmod +x activate.sh
print_success "Created activation script: activate.sh"

# ==============================================================================
# Create Project Structure
# ==============================================================================

print_step "Creating project directory structure..."

DIRECTORIES=(
    "data"
    "models"
    "notebooks"
    "scripts"
    "tests"
    "docs"
    "logs"
    "cache"
    "checkpoints"
    "tensorboard_logs"
    "finetune_plots"
    "prediction_plots"
    "datasets/ES"
    "datasets/sequences"
    "timesfm_cache"
    "sundial_cache"
    "dataset_cache"
)

for dir in "${DIRECTORIES[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    fi
done

print_success "Project structure created"

# ==============================================================================
# Create .gitignore if it doesn't exist
# ==============================================================================

if [ ! -f ".gitignore" ]; then
    print_step "Creating .gitignore file..."

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
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
venv310/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/
tensorboard_logs/

# Models and data
models/
*.h5
*.pkl
*.joblib
data/
datasets/
cache/
*.cache
*.pth
*.ckpt
checkpoints/

# Jupyter
.ipynb_checkpoints

# Temporary files
*.tmp
*.temp
.cache/

# Environment variables
.env
.env.local

# Documentation
docs/_build/
*.pdf
*.docx

# Test coverage
.coverage
htmlcov/
.pytest_cache/

# Node modules (if any)
node_modules/

# Database
*.db
*.sqlite
*.sqlite3

# API keys and secrets
secrets/
config/private/
EOF

    print_success ".gitignore file created"
fi

# ==============================================================================
# Create README
# ==============================================================================

if [ ! -f "README.md" ]; then
    print_step "Creating README.md..."

    cat > README.md << 'EOF'
# TimesFM Stock Prediction Project

A comprehensive framework for fine-tuning Google's TimesFM model on intraday stock data for prediction and trading strategies.

## Quick Start

1. **Setup Environment:**
   ```bash
   chmod +x setup_project.sh
   ./setup_project.sh
   source activate.sh
   ```

2. **Start Training:**
   ```bash
   python finetune_timesfm2.py
   ```

3. **Monitor Training:**
   ```bash
   tensorboard --logdir=tensorboard_logs
   ```

## Project Structure

```
├── finetune_timesfm2.py       # Main fine-tuning script
├── prediction_timesfm_v2.py   # TimesFM prediction interface
├── datasets/                  # Training data
├── models/                    # Saved models
├── tensorboard_logs/          # Training logs
├── finetune_plots/            # Training visualization plots
├── notebooks/                 # Jupyter notebooks
├── tests/                     # Unit tests
└── docs/                      # Documentation
```

## Features

- ✅ TimesFM 2.0 fine-tuning on intraday stock data
- ✅ Real-time TensorBoard monitoring
- ✅ Comprehensive plotting and visualization
- ✅ Automated data preprocessing
- ✅ Checkpoint management
- ✅ Early stopping and learning rate scheduling
- ✅ GPU acceleration support

## Configuration

Edit the configuration classes in `finetune_timesfm2.py`:

- `ModelConfig`: TimesFM model parameters
- `TrainingConfig`: Training hyperparameters
- `DataConfig`: Data processing settings
- `CheckpointConfig`: Checkpoint management

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- 16GB+ RAM recommended

## License

MIT License - see LICENSE file for details.
EOF

    print_success "README.md created"
fi

# ==============================================================================
# Final Setup Instructions
# ==============================================================================

echo ""
echo -e "${YELLOW}$(printf '%.0s=' {1..60})${NC}"
echo -e "${GREEN}🎉 SETUP COMPLETE!${NC}"
echo -e "${YELLOW}$(printf '%.0s=' {1..60})${NC}"

echo ""
echo -e "${CYAN}📋 Next Steps:${NC}"
echo -e "${WHITE}1. Activate environment: source activate.sh${NC}"
echo -e "${WHITE}2. Test installation: python -c 'import torch; print(torch.__version__)'${NC}"
echo -e "${WHITE}3. Start training: python finetune_timesfm2.py${NC}"
echo -e "${WHITE}4. Monitor training: tensorboard --logdir=tensorboard_logs${NC}"

echo ""
echo -e "${CYAN}📋 Useful Commands:${NC}"
echo -e "${WHITE}• source activate.sh              # Activate environment${NC}"
echo -e "${WHITE}• jupyter notebook               # Start Jupyter${NC}"
echo -e "${WHITE}• python -m pytest               # Run tests${NC}"
echo -e "${WHITE}• pre-commit install             # Setup pre-commit hooks${NC}"

echo ""
echo -e "${CYAN}🔧 Environment Details:${NC}"
echo -e "${WHITE}• Virtual Environment: $VENV_NAME${NC}"
echo -e "${WHITE}• Git User: $GIT_NAME <$GIT_EMAIL>${NC}"
echo -e "${WHITE}• Python Kernel: $VENV_NAME (for Jupyter)${NC}"

echo ""
echo -e "${YELLOW}⚠️  Important Notes:${NC}"
echo -e "${WHITE}• Some packages may have failed to install due to system dependencies${NC}"
echo -e "${WHITE}• You may need to install CUDA Toolkit for GPU support${NC}"
echo -e "${WHITE}• Check logs above for any installation warnings${NC}"

echo ""
echo -e "${GREEN}🚀 Ready to start building amazing stock prediction models!${NC}"

# ==============================================================================
# Optional: Create a simple test script
# ==============================================================================

cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test to verify the TimesFM setup is working
"""

import sys
import torch

def test_setup():
    print("🧪 Testing TimesFM Stock Prediction Setup...")

    # Test PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")

    # Test basic imports
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        print("✅ Core packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Test TimesFM import (optional)
    try:
        import timesfm
        print("✅ TimesFM imported successfully")
    except ImportError as e:
        print(f"⚠️  TimesFM not available: {e}")
        print("   This is expected if TimesFM installation failed")

    print("🎉 Setup test completed!")
    return True

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
EOF

chmod +x test_setup.py
print_success "Created test script: test_setup.py"

echo ""
echo -e "${CYAN}💡 Tip: Run 'python test_setup.py' to verify everything is working!${NC}"

# Deactivate virtual environment at the end
deactivate 2>/dev/null || true
