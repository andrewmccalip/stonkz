#!/bin/bash

# =============================================================================
# Quick UV Environment Setup Script
# =============================================================================
# A simplified version for quick environment recreation
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Quick TimesFM Environment Setup${NC}"

# Git configuration
echo -e "${BLUE}📝 Configuring Git...${NC}"
git config --global user.name "Andrew McCalip"
git config --global user.email "andrewmccalip@gmail.com"
echo -e "${GREEN}✅ Git configured${NC}"

# UV sync (assuming pyproject.toml exists)
if [[ -f "pyproject.toml" ]]; then
    echo -e "${BLUE}📦 Installing dependencies with UV...${NC}"
    uv sync --all-extras
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  No pyproject.toml found. Run ./setup_environment.sh for full setup${NC}"
fi

# Install PyTorch with CUDA
echo -e "${BLUE}🔥 Installing PyTorch with CUDA...${NC}"
uv add "torch>=2.5.0" --index-url https://download.pytorch.org/whl/cu121
uv add "torchaudio>=2.5.0" --index-url https://download.pytorch.org/whl/cu121  
uv add "torchvision>=0.20.0" --index-url https://download.pytorch.org/whl/cu121

# Install TimesFM
echo -e "${BLUE}🤖 Installing TimesFM...${NC}"
uv add "git+https://github.com/google-research/timesfm.git"

echo -e "${GREEN}✅ Quick setup completed!${NC}"
echo -e "${YELLOW}Activate with: source .venv/bin/activate${NC}"
