# TimesFM Stock Prediction Environment Activation Script
# Run this to activate the environment and start development

Write-Host "🚀 Activating TimesFM Stock Prediction Environment" -ForegroundColor Green
Write-Host "Environment: venv310" -ForegroundColor Yellow
Write-Host "Git User: Andrew McCalip <andrewmccalip@gmail.com>" -ForegroundColor Yellow
Write-Host ""

# Activate virtual environment
& ".\venv310\Scripts\Activate.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "Make sure venv310 exists. Run .\setup_project.ps1 first." -ForegroundColor Yellow
    exit 1
}

# Set environment variables
$env:PYTHONPATH = "$PWD;$PWD/timesfm/src"
$env:CUDA_VISIBLE_DEVICES = "0"  # Use GPU 0 by default (change if needed)

Write-Host "✅ Environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Cyan
Write-Host "  python finetune_timesfm2.py               # Start fine-tuning" -ForegroundColor White
Write-Host "  tensorboard --logdir=tensorboard_logs    # Start TensorBoard" -ForegroundColor White
Write-Host "  jupyter notebook                         # Start Jupyter" -ForegroundColor White
Write-Host "  python test_setup.py                     # Test installation" -ForegroundColor White
Write-Host "  pytest                                   # Run tests" -ForegroundColor White
Write-Host ""

# Optional: Show current directory contents
Write-Host "Current project files:" -ForegroundColor Cyan
Get-ChildItem -Name | Where-Object { $_ -notmatch '^venv310$|^__pycache__$|^.*\.pyc$' } | ForEach-Object {
    Write-Host "  $_" -ForegroundColor White
}

Write-Host ""
Write-Host "🚀 Ready for stock prediction development!" -ForegroundColor Green
