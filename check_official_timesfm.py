#!/usr/bin/env python3
"""
Test the official TimesFM model to see if it has the same zero-drop issue.
"""
import numpy as np
import torch

try:
    import timesfm
    
    print("🔍 Testing Official TimesFM Model")
    print("=" * 60)
    
    # Configuration
    CONTEXT_LENGTH = 448
    HORIZON_LENGTH = 64
    BACKEND = "gpu" if torch.cuda.is_available() else "cpu"
    
    print(f"Backend: {BACKEND}")
    print(f"Context: {CONTEXT_LENGTH}, Horizon: {HORIZON_LENGTH}")
    
    # Initialize official model
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=BACKEND,
            per_core_batch_size=32,
            horizon_len=128,  # Model's default
            num_layers=20,
            use_positional_embedding=True,
            context_len=512,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )
    
    print("✅ Model loaded successfully")
    
    # Create test data (normalized around 1.0)
    test_context = 1.0 + 0.01 * np.sin(np.arange(CONTEXT_LENGTH) * 0.1)
    
    # Make prediction
    inputs = [test_context.tolist()]
    freq = [0]
    
    forecast, _ = model.forecast(inputs, freq)
    prediction = forecast[0][:HORIZON_LENGTH]
    
    print(f"\n📊 Prediction Analysis:")
    print(f"   Output length: {len(prediction)}")
    print(f"   First 15 values: {prediction[:15]}")
    print(f"   Values at 10-15: {prediction[10:15]}")
    print(f"   Last 10 values: {prediction[-10:]}")
    
    # Check for zeros
    near_zeros = sum(1 for x in prediction if abs(x) < 1e-6)
    if near_zeros > 0:
        print(f"\n⚠️ Found {near_zeros} near-zero values!")
        first_zero = next((i for i, x in enumerate(prediction) if abs(x) < 1e-6), -1)
        print(f"   First zero at index: {first_zero}")
    else:
        print(f"\n✅ No zero-drop issue in official model!")
    
    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(prediction, 'b-', linewidth=2)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=12, color='red', linestyle=':', alpha=0.5, label='Step 12')
        plt.xlabel('Time Steps')
        plt.ylabel('Predicted Value')
        plt.title('Official TimesFM Prediction')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('official_timesfm_test.png')
        print(f"\n📊 Saved plot to: official_timesfm_test.png")
    except:
        pass
        
except ImportError as e:
    print(f"❌ Could not import timesfm: {e}")
    print("\nInstall with: pip install timesfm[torch]")

print("\n✅ Test complete!")
