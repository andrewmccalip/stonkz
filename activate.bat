@echo off
REM TimesFM Stock Prediction Environment Activation Script (Batch Version)
echo 🚀 Activating TimesFM Stock Prediction Environment
echo Environment: venv310
echo Git User: Andrew McCalip ^<andrewmccalip@gmail.com^>
echo.

REM Activate virtual environment
call .\venv310\Scripts\activate.bat

if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to activate virtual environment
    echo Make sure venv310 exists. Run .\setup_project.ps1 first.
    pause
    exit /b 1
)

REM Set environment variables
set PYTHONPATH=%CD%;%CD%\timesfm\src
set CUDA_VISIBLE_DEVICES=0

echo ✅ Environment activated!
echo.
echo Available commands:
echo   python finetune_timesfm2.py               # Start fine-tuning
echo   tensorboard --logdir=tensorboard_logs    # Start TensorBoard
echo   jupyter notebook                         # Start Jupyter
echo   python test_setup.py                     # Test installation
echo   pytest                                   # Run tests
echo.

REM Show current directory contents
echo Current project files:
for /f "delims=" %%i in ('dir /b /a-d ^| findstr /v venv310 ^| findstr /v __pycache__ ^| findstr /v "\.pyc$"') do (
    echo   %%i
)

echo.
echo 🚀 Ready for stock prediction development!
echo.
pause
