# Environment Setup Guide

This guide explains how to recreate the complete TimesFM stock market fine-tuning environment using UV.

## Quick Start

### Option 1: Full Setup (Recommended)
```bash
# Run the comprehensive setup script
./setup_environment.sh

# Activate the environment
source activate_uv.sh

# Test the environment
./test_environment.sh
```

### Option 2: Quick Setup (If pyproject.toml exists)
```bash
# Run quick setup
./quick_setup.sh

# Activate manually
source .venv/bin/activate
```

### Option 3: Manual UV Setup
```bash
# Initialize UV project
uv init --python 3.12

# Install dependencies
uv sync --all-extras

# Install PyTorch with CUDA
uv add "torch>=2.5.0" --index-url https://download.pytorch.org/whl/cu121
uv add "torchaudio>=2.5.0" --index-url https://download.pytorch.org/whl/cu121  
uv add "torchvision>=0.20.0" --index-url https://download.pytorch.org/whl/cu121

# Install TimesFM
uv add "git+https://github.com/google-research/timesfm.git"

# Activate
source .venv/bin/activate
```

## What Gets Set Up

### 1. Git Configuration
- User: Andrew McCalip <andrewmccalip@gmail.com>
- Default branch: main
- Pull strategy: merge (not rebase)

### 2. Python Environment
- Python 3.12
- UV package manager
- Virtual environment in `.venv/`

### 3. Core Packages
- **PyTorch 2.5+** with CUDA 12.1 support
- **Transformers 4.49.0** for model handling
- **TimesFM** from Google Research (latest)
- **JAX 0.7.1** for TimesFM backend
- **Pandas, NumPy, Matplotlib** for data processing
- **Scikit-learn** for ML utilities
- **Wandb** for experiment tracking

### 4. Development Tools
- **IPython/Jupyter** for interactive development
- **Rich** for beautiful terminal output
- **TQDM** for progress bars
- **Black, isort, flake8** for code formatting (dev dependencies)

### 5. Project Structure
```
.
├── setup_environment.sh      # Full environment setup
├── quick_setup.sh            # Quick setup for existing projects
├── activate_uv.sh            # Environment activation script
├── test_environment.sh       # Environment verification
├── run_training.sh           # Quick training script
├── pyproject.toml            # UV project configuration
├── .env.template             # Environment variables template
├── .gitignore                # Git ignore rules
├── requirements-full.txt     # Backup pip requirements
└── README.md                 # Project documentation
```

## Usage Examples

### Training
```bash
# Activate environment
source activate_uv.sh

# Quick training run
./run_training.sh 100 32 4 20 5  # epochs, batch_size, plots_per_epoch, val_plots, val_plot_freq

# Manual training with custom parameters
python pytorch_timesfm_finetune.py \
    --epochs 500 \
    --batch-size 64 \
    --plots-per-epoch 2 \
    --val-plots 20 \
    --val-plot-freq 5 \
    --keep-plots

# Disable validation plots for faster training
python pytorch_timesfm_finetune.py \
    --epochs 100 \
    --batch-size 32 \
    --no-val-plots

# Create validation plots every epoch (more frequent monitoring)
python pytorch_timesfm_finetune.py \
    --epochs 50 \
    --val-plots 15 \
    --val-plot-freq 1
```

### Environment Management
```bash
# Add new package
uv add package-name

# Add development dependency
uv add --dev pytest

# Update all packages
uv sync --upgrade

# Remove package
uv remove package-name

# Show installed packages
uv pip list
```

## Troubleshooting

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
uv remove torch torchaudio torchvision
uv add "torch>=2.5.0" --index-url https://download.pytorch.org/whl/cu121
```

### TimesFM Installation Issues
```bash
# Reinstall TimesFM
uv remove timesfm
uv add "git+https://github.com/google-research/timesfm.git"
```

### Environment Corruption
```bash
# Remove and recreate environment
rm -rf .venv
./setup_environment.sh
```

## Environment Variables

Copy `.env.template` to `.env` and customize:

```bash
# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false

# Model Configuration  
TIMESFM_MODEL_REPO=google/timesfm-1.0-200m-pytorch

# Training Configuration
BATCH_SIZE=64
LEARNING_RATE=1e-5
NUM_EPOCHS=5000
```

## Git Configuration

The setup automatically configures git with:
- Name: Andrew McCalip
- Email: andrewmccalip@gmail.com
- Default branch: main

To change these settings:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Requirements Files

Three requirements files are provided:

1. **pyproject.toml** - UV project file (primary)
2. **requirements-full.txt** - Complete pip requirements (backup)
3. **requirements-minimal.txt** - Minimal requirements (if exists)

## Support

For issues with:
- **UV**: https://github.com/astral-sh/uv
- **TimesFM**: https://github.com/google-research/timesfm
- **PyTorch**: https://pytorch.org/get-started/

---

**Author**: Andrew McCalip <andrewmccalip@gmail.com>
