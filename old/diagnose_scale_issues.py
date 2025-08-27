"""
Diagnostic tool for analyzing scale mismatch in predictions.
Run this to understand why predictions have different scale than targets.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Import your model and dataset
from pytorch_timesfm_model import TimesFMModel
from pytorch_timesfm_finetune import StockDataset, DEVICE, CONTEXT_LENGTH, HORIZON_LENGTH, INPUT_PATCH_LEN


def analyze_scale_issues(model_path=None, dataset_path=None):
    """Analyze scale mismatch between predictions and targets."""
    
    print("🔍 Diagnosing Scale Issues")
    print("=" * 50)
    
    # Load dataset
    if dataset_path:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
            train_dataset = data['train_dataset']
    else:
        # Try to load from cache
        cache_dir = Path("dataset_cache")
        cache_files = list(cache_dir.glob("pytorch_data_*.pkl"))
        if cache_files:
            with open(cache_files[0], 'rb') as f:
                data = pickle.load(f)
                train_dataset = data['train_dataset']
        else:
            print("❌ No dataset found!")
            return
    
    # Initialize model
    model = TimesFMModel(
        context_len=CONTEXT_LENGTH,
        horizon_len=HORIZON_LENGTH,
        input_patch_len=INPUT_PATCH_LEN
    ).to(DEVICE)
    
    # Load weights if provided
    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Analyze samples
    n_samples = min(100, len(train_dataset))
    
    all_predictions = []
    all_targets = []
    all_contexts = []
    
    print(f"\n📊 Analyzing {n_samples} samples...")
    
    with torch.no_grad():
        for i in range(n_samples):
            sample = train_dataset[i]
            
            # Get data
            context = sample['input'].unsqueeze(0).to(DEVICE)
            target = sample['target'].unsqueeze(0).to(DEVICE)
            freq = sample['freq'].unsqueeze(0).to(DEVICE)
            
            # Get prediction
            pred = model(context, freq)
            
            # Store for analysis
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
            # Extract context values (last patch, last values)
            context_last = context[0, -1, -10:].cpu().numpy()
            all_contexts.append(context_last)
    
    # Convert to arrays
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    all_contexts = np.vstack(all_contexts)
    
    # Calculate statistics
    print("\n📈 Scale Analysis:")
    print(f"Predictions - Mean: {all_predictions.mean():.6f}, Std: {all_predictions.std():.6f}")
    print(f"Predictions - Min: {all_predictions.min():.6f}, Max: {all_predictions.max():.6f}")
    print(f"Targets - Mean: {all_targets.mean():.6f}, Std: {all_targets.std():.6f}")
    print(f"Targets - Min: {all_targets.min():.6f}, Max: {all_targets.max():.6f}")
    print(f"Context (last 10) - Mean: {all_contexts.mean():.6f}, Std: {all_contexts.std():.6f}")
    
    # Scale ratio
    scale_ratio = all_predictions.std() / all_targets.std()
    center_diff = all_predictions.mean() - all_targets.mean()
    print(f"\nScale Ratio (pred_std/target_std): {scale_ratio:.2f}")
    print(f"Center Difference (pred_mean - target_mean): {center_diff:.6f}")
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Distribution comparison
    ax1 = axes[0, 0]
    ax1.hist(all_predictions.flatten(), bins=50, alpha=0.5, label='Predictions', density=True)
    ax1.hist(all_targets.flatten(), bins=50, alpha=0.5, label='Targets', density=True)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison')
    ax1.legend()
    
    # Plot 2: Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(all_targets.flatten(), all_predictions.flatten(), alpha=0.1)
    ax2.plot([all_targets.min(), all_targets.max()], 
             [all_targets.min(), all_targets.max()], 'r--', label='Perfect')
    ax2.set_xlabel('Target')
    ax2.set_ylabel('Prediction')
    ax2.set_title('Prediction vs Target')
    ax2.legend()
    
    # Plot 3: Time series examples
    ax3 = axes[0, 2]
    for i in range(min(5, n_samples)):
        ax3.plot(all_targets[i], 'b-', alpha=0.3, label='Target' if i == 0 else '')
        ax3.plot(all_predictions[i], 'r--', alpha=0.3, label='Prediction' if i == 0 else '')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Value')
    ax3.set_title('Sample Predictions')
    ax3.legend()
    
    # Plot 4: Scale evolution
    ax4 = axes[1, 0]
    window = 10
    pred_rolling_std = [all_predictions[i:i+window].std() for i in range(0, len(all_predictions)-window, 5)]
    target_rolling_std = [all_targets[i:i+window].std() for i in range(0, len(all_targets)-window, 5)]
    ax4.plot(pred_rolling_std, label='Prediction Std')
    ax4.plot(target_rolling_std, label='Target Std')
    ax4.set_xlabel('Sample Window')
    ax4.set_ylabel('Std Dev')
    ax4.set_title('Scale Evolution')
    ax4.legend()
    
    # Plot 5: Residual analysis
    ax5 = axes[1, 1]
    residuals = all_predictions - all_targets
    ax5.hist(residuals.flatten(), bins=50)
    ax5.set_xlabel('Residual (Pred - Target)')
    ax5.set_ylabel('Count')
    ax5.set_title(f'Residual Distribution\nMean: {residuals.mean():.6f}, Std: {residuals.std():.6f}')
    
    # Plot 6: Recommendations
    ax6 = axes[1, 2]
    ax6.text(0.1, 0.9, "🔧 Recommendations:", fontsize=14, weight='bold', transform=ax6.transAxes)
    
    recommendations = []
    if scale_ratio > 5:
        recommendations.append("• Predictions have much larger scale\n  → Use output normalization")
    elif scale_ratio < 0.2:
        recommendations.append("• Predictions have much smaller scale\n  → Increase output scale")
    
    if abs(center_diff) > 0.1:
        recommendations.append(f"• Predictions centered at {all_predictions.mean():.3f}\n  → Add bias correction")
    
    if all_predictions.std() < 1e-4:
        recommendations.append("• Predictions have no variation\n  → Check model gradients")
    
    recommendations.append(f"\n• Current scale ratio: {scale_ratio:.2f}")
    recommendations.append(f"• Suggested output scale: {1/scale_ratio:.4f}")
    recommendations.append(f"• Suggested output bias: {all_targets.mean():.4f}")
    
    ax6.text(0.05, 0.8, "\n".join(recommendations), transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('scale_diagnosis.png', dpi=150)
    plt.close()
    
    print("\n✅ Diagnostic plot saved to scale_diagnosis.png")
    
    # Return scale correction parameters
    return {
        'scale_correction': 1 / scale_ratio,
        'bias_correction': all_targets.mean(),
        'target_std': all_targets.std(),
        'pred_std': all_predictions.std()
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    args = parser.parse_args()
    
    corrections = analyze_scale_issues(args.model, args.dataset)
    
    print("\n🎯 Suggested Corrections:")
    print(f"output = predictions * {corrections['scale_correction']:.4f} + {corrections['bias_correction']:.4f}")
