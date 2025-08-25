#!/usr/bin/env python3
"""
Test script to diagnose why predictions drop to zero after ~12 timesteps.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_simple_prediction():
    """Test predictions with simple synthetic data."""
    
    # Configuration matching the main script
    CONTEXT_LENGTH = 448
    HORIZON_LENGTH = 64
    INPUT_PATCH_LEN = 32
    BATCH_SIZE = 1
    
    print("🔍 Testing Prediction Zero-Drop Issue")
    print("=" * 60)
    print(f"Context: {CONTEXT_LENGTH}, Horizon: {HORIZON_LENGTH}, Patch: {INPUT_PATCH_LEN}")
    print(f"Number of patches: {CONTEXT_LENGTH // INPUT_PATCH_LEN}")
    
    # Create synthetic normalized price data (oscillating around 1.0)
    np.random.seed(42)
    time_steps = CONTEXT_LENGTH + HORIZON_LENGTH
    synthetic_prices = 1.0 + 0.01 * np.sin(np.arange(time_steps) * 0.1) + 0.001 * np.random.randn(time_steps)
    
    # Split into context and target
    context_data = synthetic_prices[:CONTEXT_LENGTH]
    target_data = synthetic_prices[CONTEXT_LENGTH:CONTEXT_LENGTH + HORIZON_LENGTH]
    
    print(f"\n📊 Synthetic data stats:")
    print(f"   Context: min={context_data.min():.6f}, max={context_data.max():.6f}, mean={context_data.mean():.6f}")
    print(f"   Target: min={target_data.min():.6f}, max={target_data.max():.6f}, mean={target_data.mean():.6f}")
    
    # Create patches from context
    num_patches = CONTEXT_LENGTH // INPUT_PATCH_LEN
    patches = context_data.reshape(num_patches, INPUT_PATCH_LEN)
    
    print(f"\n🔍 Examining patches:")
    print(f"   Patches shape: {patches.shape}")
    print(f"   First patch mean: {patches[0].mean():.6f}")
    print(f"   Last patch mean: {patches[-1].mean():.6f}")
    
    # Create a simple linear prediction as baseline
    # Just extend the last value with a small trend
    last_context_value = context_data[-1]
    recent_trend = (context_data[-1] - context_data[-10]) / 10
    
    linear_prediction = np.zeros(HORIZON_LENGTH)
    for i in range(HORIZON_LENGTH):
        linear_prediction[i] = last_context_value + recent_trend * (i + 1)
    
    print(f"\n📊 Linear baseline prediction:")
    print(f"   First 15 values: {linear_prediction[:15]}")
    print(f"   Values around index 12: {linear_prediction[10:15]}")
    print(f"   Last 10 values: {linear_prediction[-10:]}")
    
    # Check for zeros
    near_zeros = np.sum(np.abs(linear_prediction) < 1e-6)
    print(f"   Near-zero values: {near_zeros}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Full sequence
    full_time = np.arange(len(synthetic_prices))
    ax1.plot(full_time[:CONTEXT_LENGTH], context_data, 'gray', label='Context', alpha=0.7)
    ax1.plot(full_time[CONTEXT_LENGTH:], target_data, 'black', label='Ground Truth', linewidth=2)
    ax1.plot(full_time[CONTEXT_LENGTH:], linear_prediction, 'blue', label='Linear Baseline', linestyle='--')
    ax1.axvline(x=CONTEXT_LENGTH, color='red', linestyle=':', alpha=0.5, label='Prediction Start')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Normalized Price')
    ax1.set_title('Full Sequence View')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Focus on prediction horizon
    ax2.plot(target_data, 'black', label='Ground Truth', linewidth=2)
    ax2.plot(linear_prediction, 'blue', label='Linear Baseline', linestyle='--')
    
    # Highlight the problem area
    ax2.axvspan(10, 15, alpha=0.2, color='red', label='Problem Area (steps 10-15)')
    ax2.axvline(x=12, color='red', linestyle=':', alpha=0.7, label='Step 12')
    
    # Mark specific values
    for i in [0, 12, 32, 63]:
        if i < len(target_data):
            ax2.scatter(i, target_data[i], color='black', s=50, zorder=5)
            ax2.text(i, target_data[i] + 0.001, f'{target_data[i]:.4f}', ha='center', fontsize=8)
    
    ax2.set_xlabel('Prediction Time Steps')
    ax2.set_ylabel('Normalized Price')
    ax2.set_title('Prediction Horizon Focus')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, HORIZON_LENGTH + 2)
    
    plt.tight_layout()
    plt.savefig('test_prediction_zeros.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved test plot to: test_prediction_zeros.png")
    
    # Analysis
    print(f"\n🔍 Analysis:")
    print(f"   Step 12 is at {12/HORIZON_LENGTH:.1%} of horizon")
    print(f"   Step 12 is at {12/INPUT_PATCH_LEN:.2f} patches into prediction")
    print(f"   Could be related to:")
    print(f"   - Attention mask cutoff")
    print(f"   - Positional encoding issues")
    print(f"   - Model architecture expecting shorter sequences")
    print(f"   - Numerical underflow in attention scores")

if __name__ == "__main__":
    test_simple_prediction()
    
    # Run the debug analysis too
    print("\n" + "="*60)
    import subprocess
    subprocess.run(["python", "debug_predictions.py"])
