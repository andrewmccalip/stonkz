#!/bin/bash
# TimesFM Stock Prediction Environment Activation Script - Linux Version
# Run with: source activate.sh

echo "🚀 Activating TimesFM Stock Prediction Environment"
echo "Environment: venv310"
echo "Git User: Andrew McCalip <andrewmccalip@gmail.com>"
echo ""

# Check if virtual environment exists
if [ ! -d "venv310" ]; then
    echo "❌ Virtual environment 'venv310' not found"
    echo "Make sure to run ./setup_project.sh first."
    return 1 2>/dev/null || exit 1
fi

# Activate virtual environment
source ./venv310/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    echo "Make sure venv310 exists and is properly created."
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

# Add some useful aliases for the current session
alias activate="source ./venv310/bin/activate"
alias tb="tensorboard --logdir=tensorboard_logs"
alias jn="jupyter notebook"
alias test="python test_setup.py"
