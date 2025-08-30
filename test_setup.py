#!/usr/bin/env python3
"""
Quick test to verify the TimesFM setup is working
"""

import sys
import platform

def test_setup():
    print("🧪 Testing TimesFM Stock Prediction Setup...")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version}")
    print()

    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device count: {torch.cuda.device_count()}")
            print(f"   Current CUDA device: {torch.cuda.current_device()}")
            if torch.cuda.device_count() > 0:
                print(f"   CUDA device name: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import error: {e}")
        return False

    # Test core packages
    core_packages = ['numpy', 'pandas', 'matplotlib', 'sklearn', 'scipy']
    failed_packages = []

    for package in core_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_packages.append(package)

    if failed_packages:
        print(f"\n⚠️  Some core packages failed: {', '.join(failed_packages)}")
        print("   You may need to reinstall them: pip install " + " ".join(failed_packages))

    # Test TimesFM import (optional)
    try:
        import timesfm
        print("✅ TimesFM imported successfully")
    except ImportError as e:
        print(f"⚠️  TimesFM not available: {e}")
        print("   This is expected if TimesFM installation failed")
        print("   You can still use the framework with other models")

    # Test Jupyter
    try:
        import jupyter
        print("✅ Jupyter available")
    except ImportError:
        print("⚠️  Jupyter not available - optional for development")

    # Test TensorBoard
    try:
        import tensorboard
        print("✅ TensorBoard available")
    except ImportError:
        print("⚠️  TensorBoard not available - optional for monitoring")

    print("\n" + "="*50)
    if not failed_packages:
        print("🎉 Setup test completed successfully!")
        print("   Your environment is ready for stock prediction development!")
        return True
    else:
        print("⚠️  Setup test completed with warnings!")
        print("   Some packages may need reinstallation.")
        return False

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
