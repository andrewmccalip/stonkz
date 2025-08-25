#!/usr/bin/env python3
"""
Simple test for validation plotting functionality without full imports.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configuration (copied from main script)
SCRIPT_DIR = Path(__file__).parent
PLOT_DIR = SCRIPT_DIR / "finetune_plots"
CONTEXT_LENGTH = 448
HORIZON_LENGTH = 64
INPUT_PATCH_LEN = 32

def create_test_validation_plot():
    """Create a single test validation plot to verify the plotting logic."""
    print("🧪 Creating test validation plot...")
    
    # Create validation plots directory
    val_plots_dir = PLOT_DIR / "validation"
    val_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate mock data
    num_patches = CONTEXT_LENGTH // INPUT_PATCH_LEN
    
    # Create mock context (history)
    context = 1.0 + 0.001 * np.cumsum(np.random.randn(CONTEXT_LENGTH))
    
    # Create mock target and predictions
    target = context[-1] + 0.0005 * np.cumsum(np.random.randn(HORIZON_LENGTH))
    finetuned_prediction = target + 0.0002 * np.random.randn(HORIZON_LENGTH)  # Fine-tuned (better)
    official_prediction = target + 0.0004 * np.random.randn(HORIZON_LENGTH)   # Official (worse)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Create x-axis
    context_x = np.arange(len(context))
    prediction_x = np.arange(len(context), len(context) + len(target))
    
    # Plot context (history) in blue
    ax.plot(context_x, context, 'b-', linewidth=2, label='History (Context)', alpha=0.8)
    
    # Plot ground truth in black
    ax.plot(prediction_x, target, 'k-', linewidth=3, label='Ground Truth', alpha=0.9)
    
    # Plot fine-tuned prediction in red
    ax.plot(prediction_x, finetuned_prediction, 'r-', linewidth=2, label='Fine-tuned TimesFM', alpha=0.8)
    
    # Plot official prediction in orange
    ax.plot(prediction_x, official_prediction, 'orange', linewidth=2, label='Official TimesFM', alpha=0.7, linestyle='--')
    
    # Connect context to predictions with thin lines
    ax.plot([context_x[-1], prediction_x[0]], [context[-1], target[0]], 'k-', linewidth=1, alpha=0.3)
    ax.plot([context_x[-1], prediction_x[0]], [context[-1], finetuned_prediction[0]], 'r-', linewidth=1, alpha=0.3)
    ax.plot([context_x[-1], prediction_x[0]], [context[-1], official_prediction[0]], 'orange', linewidth=1, alpha=0.3)
    
    # Add vertical line to separate context from predictions
    ax.axvline(x=len(context), color='gray', linestyle='--', alpha=0.5, label='Prediction Start')
    
    # Calculate metrics for both models
    finetuned_mse = np.mean((finetuned_prediction - target) ** 2)
    finetuned_mae = np.mean(np.abs(finetuned_prediction - target))
    official_mse = np.mean((official_prediction - target) ** 2)
    official_mae = np.mean(np.abs(official_prediction - target))
    
    # Directional accuracy
    target_direction = np.sign(np.diff(target))
    finetuned_pred_direction = np.sign(np.diff(finetuned_prediction))
    official_pred_direction = np.sign(np.diff(official_prediction))
    finetuned_dir_accuracy = np.mean(target_direction == finetuned_pred_direction) * 100
    official_dir_accuracy = np.mean(target_direction == official_pred_direction) * 100
    
    # Correlation
    finetuned_correlation = np.corrcoef(finetuned_prediction, target)[0, 1]
    official_correlation = np.corrcoef(official_prediction, target)[0, 1]
    
    # Price changes
    context_change = (context[-1] - context[0]) / context[0] * 100
    target_change = (target[-1] - target[0]) / target[0] * 100
    finetuned_pred_change = (finetuned_prediction[-1] - finetuned_prediction[0]) / finetuned_prediction[0] * 100
    official_pred_change = (official_prediction[-1] - official_prediction[0]) / official_prediction[0] * 100
    
    # Formatting and labels
    ax.set_xlabel('Time Steps (Minutes)', fontsize=12)
    ax.set_ylabel('Normalized Price', fontsize=12)
    ax.set_title('Test Validation Sample - ESH4 (ID: test_instrument)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add metrics text box with comparison
    mse_improvement = ((official_mse - finetuned_mse) / official_mse * 100) if official_mse > 0 else 0
    dir_improvement = finetuned_dir_accuracy - official_dir_accuracy
    
    metrics_text = (
        f'Fine-tuned vs Official TimesFM:\n'
        f'MSE: {finetuned_mse:.6f} vs {official_mse:.6f}\n'
        f'MAE: {finetuned_mae:.6f} vs {official_mae:.6f}\n'
        f'Dir. Acc: {finetuned_dir_accuracy:.1f}% vs {official_dir_accuracy:.1f}%\n'
        f'Corr: {finetuned_correlation:.3f} vs {official_correlation:.3f}\n\n'
        f'Improvements:\n'
        f'MSE: {mse_improvement:+.1f}%\n'
        f'Dir. Acc: {dir_improvement:+.1f}%\n\n'
        f'Price Changes:\n'
        f'Context: {context_change:.2f}%\n'
        f'Target: {target_change:.2f}%\n'
        f'Fine-tuned: {finetuned_pred_change:.2f}%\n'
        f'Official: {official_pred_change:.2f}%\n\n'
        f'Base Price: $4500.00'
    )
    
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10, fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    # Set reasonable y-axis limits
    all_values = np.concatenate([context, target, finetuned_prediction, official_prediction])
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Save plot
    plot_filename = "test_validation_plot.png"
    plot_path = val_plots_dir / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Test validation plot created: {plot_path}")
    return plot_path

if __name__ == "__main__":
    plot_path = create_test_validation_plot()
    print(f"📊 Test plot saved to: {plot_path}")
    print("🎯 This demonstrates the validation plot format that will be used during training.")
