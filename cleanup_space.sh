#!/bin/bash
# Vast.ai Space Cleanup Script
# Run this script to free up disk space on your vast.ai instance

echo "🧹 VAST.AI SPACE CLEANUP SCRIPT"
echo "================================"
echo "Starting cleanup process..."

# Check initial disk usage
echo "📊 Initial disk usage:"
df -h | head -2

echo ""
echo "🔄 Cleaning up files..."

# 1. Clean old plot files (keep latest 20)
echo "1. Cleaning old training progress plots (keeping latest 20)..."
find finetune_plots/ -name "training_progress_*.png" 2>/dev/null | sort | head -n -20 | xargs -r rm -v

# 2. Clean old iteration plots (keep latest 10)
echo "2. Cleaning old iteration plots (keeping latest 10)..."
find finetune_plots/ -name "training_iter_*.png" 2>/dev/null | sort | head -n -10 | xargs -r rm -v

# 3. Clean Python cache files
echo "3. Cleaning Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# 4. Clean temporary files
echo "4. Cleaning temporary files..."
rm -f *.tmp *.temp test_*.png 2>/dev/null || true

# 5. Clean old TensorBoard logs (keep latest 3 runs)
echo "5. Cleaning old TensorBoard logs (keeping latest 3 runs)..."
find tensorboard_logs/ -maxdepth 1 -type d -name "timesfm_run_*" 2>/dev/null | sort | head -n -3 | xargs -r rm -rf

# 6. Clean old model cache (keep latest 2)
echo "6. Cleaning old model cache (keeping latest 2)..."
find timesfm_cache/ -maxdepth 1 -type d -name "finetuned_*" 2>/dev/null | sort | head -n -2 | xargs -r rm -rf

# 7. Clean system cache and temporary files
echo "7. Cleaning system cache and temporary files..."
rm -rf /tmp/* 2>/dev/null || true
rm -rf ~/.cache/* 2>/dev/null || true

# 8. Clean pip cache
echo "8. Cleaning pip cache..."
pip cache purge 2>/dev/null || true

# 9. Clean apt cache (if running as root)
if [ "$EUID" -eq 0 ]; then
    echo "9. Cleaning apt cache..."
    apt-get clean 2>/dev/null || true
    apt-get autoremove -y 2>/dev/null || true
else
    echo "9. Skipping apt cleanup (not running as root)"
fi

# 10. Clean Docker if available
if command -v docker &> /dev/null; then
    echo "10. Cleaning Docker cache..."
    docker system prune -f 2>/dev/null || true
else
    echo "10. Docker not available, skipping"
fi

echo ""
echo "🎉 CLEANUP COMPLETE!"
echo "===================="

# Check final disk usage
echo "📊 Final disk usage:"
df -h | head -2

echo ""
echo "📁 Directory sizes after cleanup:"
du -sh * 2>/dev/null | sort -hr | head -10

echo ""
echo "✅ Cleanup script completed successfully!"
echo "💡 You can run this script anytime with: bash cleanup_space.sh"
