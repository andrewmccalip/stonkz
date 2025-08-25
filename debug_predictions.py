#!/usr/bin/env python3
"""
Debug script to analyze prediction dropping to zero issue.
"""
import numpy as np
import matplotlib.pyplot as plt

def analyze_prediction_pattern():
    """Analyze the pattern of predictions dropping to zero."""
    
    # Simulate a prediction that drops to zero after ~12 steps
    horizon_length = 64
    
    # Different scenarios to test
    scenarios = {
        "gradual_decay": lambda x: np.exp(-x/10) * (1.0 + 0.01 * np.sin(x)),
        "sudden_drop": lambda x: 1.0 if x < 12 else 0.0,
        "oscillating_decay": lambda x: np.cos(x/5) * np.exp(-x/20),
        "normal_pattern": lambda x: 1.0 + 0.001 * x + 0.01 * np.sin(x/3)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (name, func) in enumerate(scenarios.items()):
        ax = axes[idx]
        x = np.arange(horizon_length)
        y = np.array([func(i) for i in x])
        
        # Find where values become near-zero
        near_zero_mask = np.abs(y) < 1e-6
        first_zero = np.where(near_zero_mask)[0][0] if np.any(near_zero_mask) else -1
        
        # Plot
        ax.plot(x, y, 'b-', linewidth=2, label='Prediction')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        
        if first_zero >= 0:
            ax.axvline(x=first_zero, color='red', linestyle=':', alpha=0.7, 
                      label=f'First zero at {first_zero}')
        
        # Highlight the problem area around index 12
        ax.axvspan(10, 15, alpha=0.2, color='orange', label='Problem area (10-15)')
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Predicted Value')
        ax.set_title(f'{name.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text with analysis
        text = f"Min: {y.min():.6f}\nMax: {y.max():.6f}\nAt step 12: {y[12] if len(y) > 12 else 'N/A':.6f}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Prediction Pattern Analysis - Debugging Zero Drop Issue', fontsize=16)
    plt.tight_layout()
    plt.savefig('debug_prediction_patterns.png', dpi=150, bbox_inches='tight')
    print("✅ Saved debug plot to: debug_prediction_patterns.png")

def check_model_output_patterns():
    """Check for common issues that cause predictions to drop to zero."""
    
    print("🔍 Common causes of predictions dropping to zero:\n")
    
    print("1. **Activation Function Saturation**")
    print("   - ReLU dying: negative inputs → all zeros")
    print("   - Sigmoid/Tanh saturation: extreme values → near-zero gradients")
    print("   - Solution: Check activation functions in the model\n")
    
    print("2. **Normalization Issues**")
    print("   - Input data not properly normalized")
    print("   - BatchNorm/LayerNorm with small batch sizes")
    print("   - Solution: Verify normalization in data preprocessing\n")
    
    print("3. **Gradient Vanishing**")
    print("   - Deep networks without skip connections")
    print("   - Poor weight initialization")
    print("   - Solution: Check model architecture and initialization\n")
    
    print("4. **Numerical Instability**")
    print("   - Exponential operations with large negative values")
    print("   - Division by very small numbers")
    print("   - Solution: Add epsilon values, use log-space operations\n")
    
    print("5. **Model Architecture Issues**")
    print("   - Recurrent connections with poor gating")
    print("   - Attention masks cutting off after certain positions")
    print("   - Solution: Check attention masks and sequence lengths\n")
    
    print("6. **Data Preprocessing**")
    print("   - Padding with zeros affecting predictions")
    print("   - Incorrect sequence lengths")
    print("   - Solution: Verify padding and masking strategies\n")

def test_horizon_length_issue():
    """Test if the issue is related to horizon length configuration."""
    
    print("\n🔍 Testing Horizon Length Configuration:")
    print(f"   Expected horizon: 64 steps")
    print(f"   Problem occurs at: ~12 steps")
    print(f"   Ratio: 12/64 = {12/64:.2%}")
    
    # Check if it could be related to patch size
    patch_size = 32
    print(f"\n   Patch size: {patch_size}")
    print(f"   12 steps = {12/patch_size:.2f} patches")
    print(f"   Could be related to patch boundaries?\n")
    
    # Check common model output sizes
    common_sizes = [16, 32, 64, 128, 256, 512]
    for size in common_sizes:
        if size >= 12:
            print(f"   Model layer with size {size}: step 12 is at position {12/size:.2%}")

if __name__ == "__main__":
    print("🐛 Debugging Prediction Zero-Drop Issue")
    print("=" * 60)
    
    # Analyze patterns
    analyze_prediction_pattern()
    
    # Check common issues
    check_model_output_patterns()
    
    # Test horizon configuration
    test_horizon_length_issue()
    
    print("\n📋 Debugging Checklist:")
    print("[ ] Check model output at each layer during forward pass")
    print("[ ] Verify attention mask lengths match horizon length")
    print("[ ] Ensure no ReLU or activation saturation")
    print("[ ] Check for numerical underflow in exponential operations")
    print("[ ] Verify input normalization is correct")
    print("[ ] Test with different horizon lengths (16, 32, 64)")
    print("[ ] Check if issue occurs at same absolute position or relative position")
    
    print("\n✅ Debug analysis complete!")
