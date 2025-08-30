# ==============================================================================
# TimesFM Stock Prediction Project Setup Script
# ==============================================================================
# This script sets up the entire project environment from scratch
# Run with: .\setup_project.ps1

Write-Host "🚀 TimesFM Stock Prediction Project Setup" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Yellow

# Configuration
$PYTHON_VERSION = "3.10"
$VENV_NAME = "venv310"
$GIT_EMAIL = "andrewmccalip@gmail.com"
$GIT_NAME = "Andrew McCalip"

# ==============================================================================
# Function Definitions
# ==============================================================================

function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

function Write-Step {
    param([string]$Message)
    Write-Host "`n📋 $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

# ==============================================================================
# Prerequisites Check
# ==============================================================================

Write-Step "Checking Prerequisites..."

# Check if running as administrator (some installations might need it)
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Warning "Not running as Administrator. Some installations might require admin privileges."
}

# Check Python installation
if (Test-Command "python") {
    $pythonVersion = python --version
    Write-Success "Python found: $pythonVersion"
} else {
    Write-Error "Python not found. Please install Python $PYTHON_VERSION from https://python.org"
    exit 1
}

# Check if Python version is correct
$pythonVersionOutput = python --version 2>&1
if ($pythonVersionOutput -match "Python (\d+)\.(\d+)") {
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    if ($major -eq 3 -and $minor -ge 10) {
        Write-Success "Python version $major.$minor is compatible"
    } else {
        Write-Warning "Python version $major.$minor detected. Recommended: Python $PYTHON_VERSION+"
    }
}

# Check Git installation
if (Test-Command "git") {
    $gitVersion = git --version
    Write-Success "Git found: $gitVersion"
} else {
    Write-Error "Git not found. Installing Git..."

    # Try to install Git using winget
    if (Test-Command "winget") {
        Write-Step "Installing Git via winget..."
        winget install --id Git.Git -e --source winget
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Git installed successfully"
        } else {
            Write-Error "Failed to install Git via winget"
            Write-Host "Please install Git manually from https://git-scm.com"
            exit 1
        }
    } else {
        Write-Error "winget not found. Please install Git manually from https://git-scm.com"
        exit 1
    }
}

# ==============================================================================
# Git Configuration
# ==============================================================================

Write-Step "Setting up Git configuration..."

try {
    # Set Git user name and email
    git config --global user.name $GIT_NAME
    git config --global user.email $GIT_EMAIL

    # Set some useful Git defaults
    git config --global core.autocrlf true
    git config --global core.editor "code --wait"
    git config --global init.defaultBranch main
    git config --global pull.rebase false

    Write-Success "Git configuration set for $GIT_NAME <$GIT_EMAIL>"
} catch {
    Write-Warning "Failed to configure Git: $_"
}

# ==============================================================================
# Virtual Environment Setup
# ==============================================================================

Write-Step "Setting up Python virtual environment..."

# Remove existing venv if it exists
if (Test-Path $VENV_NAME) {
    Write-Warning "Removing existing virtual environment..."
    Remove-Item $VENV_NAME -Recurse -Force
}

# Create new virtual environment
Write-Step "Creating virtual environment: $VENV_NAME"
python -m venv $VENV_NAME

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create virtual environment"
    exit 1
}

Write-Success "Virtual environment created successfully"

# Activate virtual environment
Write-Step "Activating virtual environment..."
& ".\$VENV_NAME\Scripts\Activate.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to activate virtual environment"
    exit 1
}

# Upgrade pip
Write-Step "Upgrading pip..."
python -m pip install --upgrade pip

# ==============================================================================
# Install Core Dependencies
# ==============================================================================

Write-Step "Installing core Python dependencies..."

$requirements = @(
    "torch>=2.0.0",
    "torchvision",
    "torchaudio",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "scipy",
    "jupyter",
    "notebook",
    "ipykernel",
    "tqdm",
    "requests",
    "python-dotenv",
    "psutil",
    "GPUtil",
    "tensorboard",
    "wandb",
    "plotly",
    "kaleido",  # For plotly static image export
    "Pillow",
    "opencv-python",
    "h5py",
    "tables",
    "openpyxl",
    "xlrd",
    "sqlalchemy",
    "pymysql",
    "psycopg2-binary",
    "redis",
    "celery",
    "fastapi",
    "uvicorn",
    "pydantic",
    "python-multipart",
    "aiofiles"
)

foreach ($package in $requirements) {
    Write-Host "Installing $package..." -ForegroundColor Gray
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to install $package"
    }
}

# ==============================================================================
# Install ML/AI Specific Packages
# ==============================================================================

Write-Step "Installing ML/AI specific packages..."

$ml_packages = @(
    "transformers",
    "datasets",
    "tokenizers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "optimum",
    "onnx",
    "onnxruntime",
    "jax",
    "jaxlib",
    "flax",
    "dm-haiku",
    "optax",
    "einops",
    "timm",
    "fairscale",
    "deepspeed",
    "apex",
    "pytorch-lightning",
    "lightning",
    "huggingface-hub",
    "diffusers",
    "invisible-watermark",
    "safetensors"
)

foreach ($package in $ml_packages) {
    Write-Host "Installing $package..." -ForegroundColor Gray
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to install $package"
    }
}

# ==============================================================================
# Install TimesFM and Related Packages
# ==============================================================================

Write-Step "Installing TimesFM and time series packages..."

$timesfm_packages = @(
    "timesfm[torch]",
    "paxml",
    "lingvo",
    "seqio",
    "t5",
    "flaxformer",
    "mesh-tensorflow",
    "tensorflow-text",
    "tensorflow-datasets",
    "yfinance",
    "ta-lib",
    "pandas-ta",
    "ta",
    "pandas-datareader",
    "fredapi",
    "quandl",
    "alpha-vantage",
    "pyportfolioopt",
    "cvxpy",
    "cvxopt",
    "quantlib",
    "quantlib-python",
    "backtrader",
    "zipline",
    "pyfolio",
    "empyrical",
    "ffn",
    "quantstats"
)

foreach ($package in $timesfm_packages) {
    Write-Host "Installing $package..." -ForegroundColor Gray
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to install $package (might require special setup)"
    }
}

# ==============================================================================
# Install Development Tools
# ==============================================================================

Write-Step "Installing development and testing tools..."

$dev_packages = @(
    "black",
    "isort",
    "flake8",
    "mypy",
    "pylint",
    "pytest",
    "pytest-cov",
    "pytest-xdist",
    "pytest-html",
    "pytest-mock",
    "coverage",
    "tox",
    "pre-commit",
    "commitizen",
    "mkdocs",
    "mkdocs-material",
    "mkdocstrings",
    "jupyter-book",
    "sphinx",
    "nbsphinx",
    "sphinx-rtd-theme",
    "sphinx-autobuild",
    "sphinx-gallery",
    "jupyter-sphinx"
)

foreach ($package in $dev_packages) {
    Write-Host "Installing $package..." -ForegroundColor Gray
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to install $package"
    }
}

# ==============================================================================
# Create Jupyter Kernel
# ==============================================================================

Write-Step "Setting up Jupyter kernel..."

# Create IPython kernel for the virtual environment
python -m ipykernel install --user --name=$VENV_NAME --display-name="Python ($VENV_NAME)"

if ($LASTEXITCODE -eq 0) {
    Write-Success "Jupyter kernel created: $VENV_NAME"
} else {
    Write-Warning "Failed to create Jupyter kernel"
}

# ==============================================================================
# Create Environment Activation Script
# ==============================================================================

Write-Step "Creating environment activation helper..."

$activateScript = @"
# TimesFM Stock Prediction Environment
# Run this to activate the environment and start development

Write-Host "🚀 Activating TimesFM Stock Prediction Environment" -ForegroundColor Green
Write-Host "Environment: $VENV_NAME" -ForegroundColor Yellow
Write-Host "Git User: $GIT_NAME <$GIT_EMAIL>" -ForegroundColor Yellow
Write-Host ""

# Activate virtual environment
& ".\$VENV_NAME\Scripts\Activate.ps1"

# Set environment variables
`$env:PYTHONPATH = "`$PWD;\`$PWD/timesfm/src"
`$env:CUDA_VISIBLE_DEVICES = "0"  # Use GPU 0 by default

Write-Host "Environment activated!" -ForegroundColor Green
Write-Host "Available commands:" -ForegroundColor Cyan
Write-Host "  python finetune_timesfm2.py    # Start fine-tuning" -ForegroundColor White
Write-Host "  tensorboard --logdir=tensorboard_logs    # Start TensorBoard" -ForegroundColor White
Write-Host "  jupyter notebook    # Start Jupyter" -ForegroundColor White
Write-Host "  pytest    # Run tests" -ForegroundColor White
Write-Host ""

# Optional: Start TensorBoard in background
# Start-Process -NoNewWindow -FilePath "tensorboard" -ArgumentList "--logdir=tensorboard_logs", "--host=127.0.0.1", "--port=6006"
"@

$activateScript | Out-File -FilePath "activate.ps1" -Encoding UTF8
Write-Success "Created activation script: activate.ps1"

# ==============================================================================
# Create Project Structure
# ==============================================================================

Write-Step "Creating project directory structure..."

$directories = @(
    "data",
    "models",
    "notebooks",
    "scripts",
    "tests",
    "docs",
    "logs",
    "cache",
    "checkpoints",
    "tensorboard_logs",
    "finetune_plots",
    "prediction_plots",
    "datasets/ES",
    "datasets/sequences",
    "timesfm_cache",
    "sundial_cache",
    "dataset_cache"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Gray
    }
}

Write-Success "Project structure created"

# ==============================================================================
# Create .gitignore if it doesn't exist
# ==============================================================================

if (-not (Test-Path ".gitignore")) {
    Write-Step "Creating .gitignore file..."

    $gitignoreContent = @"
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
"@

    $gitignoreContent | Out-File -FilePath ".gitignore" -Encoding UTF8
    Write-Success ".gitignore file created"
}

# ==============================================================================
# Create README
# ==============================================================================

if (-not (Test-Path "README.md")) {
    Write-Step "Creating README.md..."

    $readmeContent = @"
# TimesFM Stock Prediction Project

A comprehensive framework for fine-tuning Google's TimesFM model on intraday stock data for prediction and trading strategies.

## Quick Start

1. **Setup Environment:**
   ```powershell
   .\setup_project.ps1
   .\activate.ps1
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
"@

    $readmeContent | Out-File -FilePath "README.md" -Encoding UTF8
    Write-Success "README.md created"
}

# ==============================================================================
# Final Setup Instructions
# ==============================================================================

Write-Host "`n" + "=" * 60 -ForegroundColor Yellow
Write-Host "🎉 SETUP COMPLETE!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Yellow

Write-Host "`n📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Activate environment: .\activate.ps1" -ForegroundColor White
Write-Host "2. Test installation: python -c 'import torch; print(torch.__version__)'" -ForegroundColor White
Write-Host "3. Start training: python finetune_timesfm2.py" -ForegroundColor White
Write-Host "4. Monitor training: tensorboard --logdir=tensorboard_logs" -ForegroundColor White

Write-Host "`n📋 Useful Commands:" -ForegroundColor Cyan
Write-Host "• .\activate.ps1              # Activate environment" -ForegroundColor White
Write-Host "• jupyter notebook           # Start Jupyter" -ForegroundColor White
Write-Host "• python -m pytest           # Run tests" -ForegroundColor White
Write-Host "• pre-commit install         # Setup pre-commit hooks" -ForegroundColor White

Write-Host "`n🔧 Environment Details:" -ForegroundColor Cyan
Write-Host "• Virtual Environment: $VENV_NAME" -ForegroundColor White
Write-Host "• Git User: $GIT_NAME <$GIT_EMAIL>" -ForegroundColor White
Write-Host "• Python Kernel: $VENV_NAME (for Jupyter)" -ForegroundColor White

Write-Host "`n⚠️  Important Notes:" -ForegroundColor Yellow
Write-Host "• Some packages may have failed to install due to system dependencies" -ForegroundColor White
Write-Host "• You may need to install CUDA Toolkit for GPU support" -ForegroundColor White
Write-Host "• Check logs above for any installation warnings" -ForegroundColor White

Write-Host "`n🚀 Ready to start building amazing stock prediction models!" -ForegroundColor Green

# ==============================================================================
# Optional: Create a simple test script
# ==============================================================================

$testScript = @"
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
"@

$testScript | Out-File -FilePath "test_setup.py" -Encoding UTF8
Write-Success "Created test script: test_setup.py"

Write-Host "`n💡 Tip: Run 'python test_setup.py' to verify everything is working!" -ForegroundColor Cyan
