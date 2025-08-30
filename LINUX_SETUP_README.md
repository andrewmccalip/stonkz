# Linux Setup Guide for TimesFM Stock Prediction Project

This guide provides comprehensive instructions for setting up the TimesFM stock prediction project on Linux systems.

## 🚀 Quick Setup (Automated)

### Option 1: One-Command Setup (Recommended)

```bash
# Clone your repository (if not already done)
git clone <your-repo-url>
cd stonkz

# Make setup script executable and run it
chmod +x setup_project.sh
./setup_project.sh

# Activate the environment
source activate.sh

# Test the installation
python test_setup.py
```

### Option 2: Manual Setup

If the automated script fails, follow these manual steps:

## 📋 Prerequisites

### Ubuntu/Debian
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and essential build tools
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Install system dependencies
sudo apt install -y build-essential cmake git curl wget \
    libssl-dev libffi-dev libjpeg-dev zlib1g-dev libpng-dev \
    libfreetype6-dev liblcms2-dev libopenjp2-7 libtiff5 \
    libxcb1 libxcb1-dev libx11-xcb1 libxss1 libasound2 \
    libgtk-3-0 libgbm1 libxcomposite1 libxcursor1 libxdamage1 \
    libxi6 libxtst6 libnss3 libcups2 libxrandr2 fonts-liberation \
    lsb-release xdg-utils pkg-config
```

### CentOS/RHEL/Fedora
```bash
# Install Python and development tools
sudo yum install -y python3 python3-pip python3-devel || \
sudo dnf install -y python3 python3-pip python3-devel

# Install system dependencies
sudo yum groupinstall -y "Development Tools"
sudo yum install -y openssl-devel libffi-devel cmake git curl wget
```

### Arch Linux
```bash
# Install Python and development tools
sudo pacman -S --noconfirm python python-pip base-devel cmake git curl wget

# Install additional dependencies
sudo pacman -S --noconfirm libjpeg-turbo zlib libpng freetype2 lcms2 \
    openjpeg2 libtiff xcb-util xcb-util-wm xcb-util-keysyms alsa-lib \
    gtk3 libxcomposite libxcursor libxdamage libxi libxtst nss cups \
    libxrandr gconf libxss liberation-fonts lsb-release xdg-utils
```

## 🐍 Python Environment Setup

### 1. Create Virtual Environment
```bash
# Create Python 3.10 virtual environment
python3 -m venv venv310

# Activate it
source venv310/bin/activate
```

### 2. Upgrade pip
```bash
pip install --upgrade pip
```

### 3. Install Dependencies

#### Core Dependencies
```bash
pip install torch>=2.0.0 torchvision torchaudio numpy pandas matplotlib seaborn scikit-learn scipy
pip install jupyter notebook ipykernel tqdm requests python-dotenv psutil plotly kaleido Pillow
pip install opencv-python h5py tables openpyxl sqlalchemy pymysql psycopg2-binary redis
pip install fastapi uvicorn pydantic python-multipart aiofiles tensorboard
```

#### ML/AI Dependencies
```bash
pip install transformers datasets tokenizers accelerate peft bitsandbytes optimum
pip install onnx onnxruntime jax jaxlib flax dm-haiku optax einops timm
pip install fairscale deepspeed pytorch-lightning lightning huggingface-hub
pip install diffusers invisible-watermark safetensors
```

#### TimesFM and Financial Dependencies
```bash
pip install timesfm[torch] yfinance pandas-ta ta pandas-datareader
pip install fredapi quandl alpha-vantage pyportfolioopt cvxpy backtrader
pip install zipline pyfolio empyrical ffn quantstats
```

#### TA-Lib (Technical Analysis Library)
```bash
# This might fail on some systems - it's optional
pip install ta-lib
```

#### Development Tools
```bash
pip install black isort flake8 mypy pylint pytest pytest-cov pytest-xdist
pip install pytest-html pytest-mock coverage tox pre-commit commitizen
pip install mkdocs mkdocs-material mkdocstrings jupyter-book
pip install sphinx nbsphinx sphinx-rtd-theme sphinx-autobuild sphinx-gallery jupyter-sphinx
```

## 🔧 Git Configuration

```bash
# Set your Git identity
git config --global user.name "Andrew McCalip"
git config --global user.email "andrewmccalip@gmail.com"

# Set useful defaults
git config --global core.autocrlf input
git config --global core.editor "nano"
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global credential.helper store
```

## 🎯 Jupyter Setup

```bash
# Create Jupyter kernel for the virtual environment
python -m ipykernel install --user --name=venv310 --display-name="Python (venv310)"

# Start Jupyter (after activating environment)
jupyter notebook
```

## 🧪 Testing Installation

Create and run this test script:

```python
#!/usr/bin/env python3
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test_setup():
    print("🧪 Testing TimesFM Stock Prediction Setup...")

    # Test PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")

    # Test basic imports
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import sklearn
        print("✅ Core packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    print("🎉 Setup test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_setup()
    exit(0 if success else 1)
```

## 🏗️ Project Structure

After setup, your project should have this structure:

```
stonkz/
├── venv310/                    # Virtual environment
├── finetune_timesfm2.py       # Main fine-tuning script
├── prediction_timesfm_v2.py   # TimesFM prediction interface
├── setup_project.sh          # Setup script
├── activate.sh               # Environment activation
├── test_setup.py             # Test script
├── requirements-linux.txt    # Linux dependencies
├── datasets/                 # Training data
│   ├── ES/                  # E-mini S&P 500 data
│   └── sequences/           # Processed sequences
├── models/                   # Saved models
├── tensorboard_logs/         # Training logs
├── finetune_plots/           # Training visualization plots
├── notebooks/                # Jupyter notebooks
├── tests/                    # Unit tests
├── docs/                     # Documentation
└── cache/                    # Cache directories
```

## 🚀 Running the Project

### 1. Activate Environment
```bash
source activate.sh
```

### 2. Start Training
```bash
python finetune_timesfm2.py
```

### 3. Monitor Training
```bash
# In another terminal
tensorboard --logdir=tensorboard_logs
```

### 4. Start Jupyter
```bash
jupyter notebook
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Not Available
```bash
# Install CUDA toolkit (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Install cuDNN (download from NVIDIA website)
# Then reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. TimesFM Installation Issues
```bash
# Try installing with specific versions
pip install timesfm[torch] --no-deps
pip install paxml lingvo seqio t5 flaxformer mesh-tensorflow tensorflow-text tensorflow-datasets
```

#### 3. TA-Lib Installation Issues
```bash
# Ubuntu/Debian
sudo apt install build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install ta-lib

# Or use conda (alternative)
conda install -c conda-forge ta-lib
```

#### 4. Permission Issues
```bash
# If you get permission errors
sudo chown -R $USER:$USER ~/your-project-directory
```

#### 5. Memory Issues
```bash
# For systems with limited RAM (<16GB)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# Or reduce batch size in configuration
```

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 4-core processor
- **RAM**: 8GB (16GB recommended)
- **Storage**: 50GB free space
- **OS**: Ubuntu 18.04+, CentOS 7+, or similar

### Recommended Requirements
- **CPU**: 8+ core processor
- **RAM**: 32GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: 100GB+ SSD
- **OS**: Ubuntu 20.04+ or similar

## 🔄 Updating the Environment

To update dependencies:

```bash
# Activate environment
source activate.sh

# Update all packages
pip install --upgrade -r requirements-linux.txt

# Or update specific packages
pip install --upgrade torch torchvision timesfm
```

## 📞 Support

If you encounter issues:

1. Check the test script: `python test_setup.py`
2. Review error logs in `tensorboard_logs/`
3. Check GitHub issues for similar problems
4. Ensure your system meets the requirements

## 🎯 Next Steps

After successful setup:

1. **Explore the data**: Check `datasets/` directory
2. **Run a test training**: Use sample configuration first
3. **Monitor progress**: Use TensorBoard for visualization
4. **Customize configuration**: Edit hyperparameters in `finetune_timesfm2.py`
5. **Start experimenting**: Create your own prediction models

Happy coding! 🚀
