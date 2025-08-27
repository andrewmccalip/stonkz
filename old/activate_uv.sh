#!/bin/bash
# Activate UV environment
cd "$(dirname "$0")"  # Ensure we're in the project directory
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