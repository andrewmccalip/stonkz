# 🚀 FlaMinGo TimesFM Environment Setup

One-click setup scripts to get your FlaMinGo TimesFM stock finetuning environment ready on any machine.

## 📋 Prerequisites

### Linux/Ubuntu (Recommended for GPU training)
- Ubuntu 20.04+ or similar Linux distribution
- NVIDIA GPU with compute capability 6.0+ (for GPU training)
- Internet connection for downloading dependencies

### Windows (Development/Testing)
- Windows 10/11
- Python 3.8+ installed
- NVIDIA GPU with drivers (optional, for GPU training)
- PowerShell 5.1+ or PowerShell Core

## 🚀 Quick Start

### Linux/Ubuntu Setup
```bash
# Make script executable and run
chmod +x setup_environment.sh
./setup_environment.sh
```

### Windows Setup
```powershell
# Run PowerShell as Administrator (recommended)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_environment.ps1
```

### Skip CUDA (CPU-only setup)
```powershell
# Windows - CPU only
.\setup_environment.ps1 -SkipCuda

# Linux - CPU only (modify script or install without CUDA)
```

## 📦 What Gets Installed

### System Dependencies
- **Python 3.8+** with pip and venv
- **Git** for repository cloning
- **Build tools** (gcc, make, etc.)
- **NVIDIA drivers and CUDA** (if GPU detected)

### Python Environment
- **Virtual environment** (`flamingo_env/`)
- **PyTorch** with CUDA support
- **Transformers** library
- **FlaMinGo TimesFM** from HuggingFace
- **Data processing** libraries (pandas, numpy, pyarrow)
- **Visualization** libraries (matplotlib, seaborn)
- **Scientific computing** (scikit-learn, JAX)

### Project Structure
```
your-project/
├── flamingo_env/              # Virtual environment
├── model_checkpoints/         # Saved model checkpoints
├── training_plots/           # Training progress plots
├── dataset_analysis_plots/   # Dataset analysis visualizations
├── dataset_cache/           # Cached processed data
├── databento/ES/           # Dataset location

├── activate_flamingo.sh     # Environment activation (Linux)
├── activate_flamingo.ps1    # Environment activation (Windows)
├── quick_start.sh          # Quick start pipeline (Linux)
├── quick_start.ps1         # Quick start pipeline (Windows)
└── environment_info.txt    # Setup summary
```

## 🎯 After Setup

### 1. Activate Environment
```bash
# Linux
source activate_flamingo.sh

# Windows
.\activate_flamingo.ps1
```

### 2. Upload Your Dataset
Place your dataset file at:
```
databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv
```

### 3. Run the Pipeline
```bash
# Quick start (runs everything)
./quick_start.sh          # Linux
.\quick_start.ps1         # Windows

# Or run individual components
python analyze_dataset.py      # Analyze your data
python finetuning.py          # Start training
python finetuned_inference.py # Run inference
```

## 🔧 Troubleshooting

### NVIDIA/CUDA Issues
```bash
# Check GPU status
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Python Environment Issues
```bash
# Recreate environment
rm -rf flamingo_env
python -m venv flamingo_env
source flamingo_env/bin/activate  # Linux
# or
flamingo_env\Scripts\activate     # Windows
```

### Permission Issues (Linux)
```bash
# Make scripts executable
chmod +x *.sh

# Fix ownership if needed
sudo chown -R $USER:$USER .
```

### Windows PowerShell Execution Policy
```powershell
# Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Check current policy
Get-ExecutionPolicy
```

## 📊 Verification

After setup, verify everything works:

```python
# Test critical imports
python -c "
import torch
import transformers
from timesfm import TimesFmHparams
print('✅ All imports successful')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

## 🎮 GPU Training Instances

### Recommended Cloud Instances
- **Google Cloud**: n1-standard-4 + 1x NVIDIA T4/V100
- **AWS**: p3.2xlarge (V100) or g4dn.xlarge (T4)
- **Azure**: NC6s_v3 (V100) or NC4as_T4_v3 (T4)

### Instance Setup Commands
```bash
# SSH into your instance
ssh -i your-key.pem user@instance-ip

# Clone your repository
git clone https://github.com/your-repo/stonkz.git
cd stonkz

# Run setup
./setup_environment.sh

# Upload dataset (use scp, rsync, or cloud storage)
scp local-dataset.csv user@instance-ip:~/stonkz/databento/ES/
```

## 🔄 Updates

To update the environment:
```bash
# Activate environment
source activate_flamingo.sh  # Linux
.\activate_flamingo.ps1      # Windows

# Update packages
pip install --upgrade -r requirements-minimal.txt

# Update FlaMinGo
cd FlaMinGo-timesfm-clean
git pull
pip install -e .
cd ..
```

## 📞 Support

If you encounter issues:

1. **Check logs**: Look at the setup script output for errors
2. **Verify requirements**: Ensure Python 3.8+, CUDA compatibility
3. **Check environment**: Run verification commands above
4. **Clean install**: Remove `flamingo_env/` and run setup again

## 🎯 Next Steps

Once setup is complete:
1. **Analyze your data**: `python analyze_dataset.py`
2. **Start training**: `python finetuning.py`
3. **Monitor progress**: Check `training_plots/` for real-time updates
4. **Run inference**: `python finetuned_inference.py`

Happy training! 🚀
