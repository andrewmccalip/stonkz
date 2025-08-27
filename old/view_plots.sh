#!/bin/bash
# Simple script to view training plots

echo "🖼️  TimesFM Training Plot Viewer"
echo "================================"

PLOT_DIR="/workspace/stonkz/training_plots"

if [ ! -d "$PLOT_DIR" ]; then
    echo "⚠️  Plot directory doesn't exist yet."
    echo "   Plots will be generated every 500 training iterations."
    exit 1
fi

if [ -f "$PLOT_DIR/latest.png" ]; then
    echo "✅ Latest plot available at:"
    echo "   $PLOT_DIR/latest.png"
    echo ""
    echo "📊 All plots:"
    ls -la $PLOT_DIR/training_iter_*.png 2>/dev/null || echo "   No iteration plots yet."
else
    echo "⏳ No plots generated yet."
    echo "   Plots are saved every 500 iterations during training."
fi

echo ""
echo "💡 Tip: You can also run 'python plot_viewer.py' and visit http://localhost:8080"
