#!/bin/bash

# =============================================================================
# Validation Plot Viewer Script
# =============================================================================
# Quick script to view validation plots from training
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

VAL_PLOTS_DIR="finetune_plots/validation"

echo -e "${BLUE}📊 TimesFM Validation Plot Viewer${NC}"
echo "=================================="

# Check if validation plots directory exists
if [[ ! -d "$VAL_PLOTS_DIR" ]]; then
    echo -e "${YELLOW}⚠️  No validation plots directory found: $VAL_PLOTS_DIR${NC}"
    echo "Run training with validation plots enabled first:"
    echo "  python pytorch_timesfm_finetune.py --val-plots 20 --val-plot-freq 5"
    exit 1
fi

# Count plots
TOTAL_PLOTS=$(find "$VAL_PLOTS_DIR" -name "*.png" | wc -l)
if [[ $TOTAL_PLOTS -eq 0 ]]; then
    echo -e "${YELLOW}⚠️  No validation plots found in $VAL_PLOTS_DIR${NC}"
    echo "Run training with validation plots enabled first."
    exit 1
fi

echo -e "${GREEN}✅ Found $TOTAL_PLOTS validation plots${NC}"
echo ""

# List available epochs
EPOCHS=$(find "$VAL_PLOTS_DIR" -name "epoch_*.png" | sed 's/.*epoch_\([0-9]*\)_.*/\1/' | sort -n | uniq)

echo "📈 Available epochs:"
for epoch in $EPOCHS; do
    epoch_plots=$(find "$VAL_PLOTS_DIR" -name "epoch_${epoch}_*.png" | wc -l)
    echo "   Epoch $epoch: $epoch_plots plots"
done

echo ""

# Get latest epoch
LATEST_EPOCH=$(echo "$EPOCHS" | tail -1)
echo -e "${BLUE}📊 Latest epoch: $LATEST_EPOCH${NC}"

# List latest epoch plots
echo ""
echo "🔍 Latest epoch plots:"
find "$VAL_PLOTS_DIR" -name "epoch_${LATEST_EPOCH}_*.png" | sort | while read plot; do
    filename=$(basename "$plot")
    echo "   $filename"
done

echo ""
echo "📁 Validation plots directory: $VAL_PLOTS_DIR"
echo ""

# Usage instructions
echo "💡 Usage:"
echo "   # View all plots in file manager"
echo "   xdg-open $VAL_PLOTS_DIR"
echo ""
echo "   # View specific epoch"
echo "   ls $VAL_PLOTS_DIR/epoch_${LATEST_EPOCH}_*.png"
echo ""
echo "   # Copy latest plots to current directory"
echo "   cp $VAL_PLOTS_DIR/epoch_${LATEST_EPOCH}_*.png ."
echo ""

# Optionally open the directory (if running in a GUI environment)
if command -v xdg-open &> /dev/null && [[ -n "$DISPLAY" ]]; then
    read -p "Open validation plots directory in file manager? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        xdg-open "$VAL_PLOTS_DIR"
    fi
fi
