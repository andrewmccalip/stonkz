#!/usr/bin/env python3
"""
Quick test script to verify TimesFM imports are working correctly
"""

import sys
from pathlib import Path

# Add TimesFM to path
SCRIPT_DIR = Path(__file__).parent
TIMESFM_DIR = SCRIPT_DIR / "timesfm" / "src"
sys.path.append(str(TIMESFM_DIR))

def test_imports():
    """Test that all required TimesFM imports work"""
    
    print("🧪 Testing TimesFM imports...")
    
    try:
        # Test basic timesfm import
        import timesfm
        print("✅ timesfm imported successfully")
        
        # Test specific imports
        from timesfm import TimesFmCheckpoint, TimesFmHparams
        print("✅ TimesFmCheckpoint and TimesFmHparams imported")
        
        # Test model creation (without loading weights)
        print("🤖 Testing model creation...")
        
        device = "cuda" if __name__ == "__main__" else "cpu"  # Use CPU for testing
        
        hparams = TimesFmHparams(
            backend=device,
            per_core_batch_size=16,
            horizon_len=128,
            num_layers=50,
            use_positional_embedding=False,
            context_len=2048,
        )
        print("✅ TimesFmHparams created successfully")
        
        # Test checkpoint creation (without downloading)
        checkpoint = TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        )
        print("✅ TimesFmCheckpoint created successfully")
        
        print("🎉 All imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
