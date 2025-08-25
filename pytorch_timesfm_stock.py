#!/usr/bin/env python3
"""
Test TimesFM stock predictions with proper visualization.
"""
import os
import pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

from pytorch_timesfm_model import TimesFMModel

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONTEXT_LENGTH = 512
HORIZON_LENGTH = 128
INPUT_PATCH_LEN = 32
MODEL_REPO = "google/timesfm-1.0-200m-pytorch"

def load_cached_data():
    """Load cached dataset."""
    # Find the most recent pytorch cache file
    cache_dir = Path("/workspace/stonkz/dataset_cache")
    pytorch_files = list(cache_dir.glob("pytorch_data_*.pkl"))
    if not pytorch_files:
        raise FileNotFoundError("No pytorch cache files found")
    
    # Use the most recent file
    cache_path = max(pytorch_files, key=lambda p: p.stat().st_mtime)
    
    print(f"📦 Loading cached data from {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def get_sample_sequence(data_dict, idx=0):
    """Get a sample sequence from the dataset."""
    dataset = data_dict['train_dataset']
    
    # Get sequence info
    seq_info = dataset.sequence_to_price_idx[idx]
    
    # Get raw prices for this sequence
    start_idx = seq_info['start']
    end_idx = seq_info['end']
    raw_prices = dataset.all_prices[start_idx:end_idx]
    
    # Split into context and target
    context = raw_prices[:CONTEXT_LENGTH]
    target = raw_prices[CONTEXT_LENGTH:CONTEXT_LENGTH + HORIZON_LENGTH]
    
    # Normalize using first price as base
    base_price = raw_prices[0]
    normalized_context = context / base_price
    normalized_target = target / base_price
    
    return {
        'raw_context': context,
        'raw_target': target,
        'normalized_context': normalized_context,
        'normalized_target': normalized_target,
        'base_price': base_price,
        'symbol': seq_info['symbol'],
        'instrument_id': seq_info['instrument_id']
    }

def test_timesfm_prediction():
    """Test TimesFM predictions on stock data."""
    print("\n🚀 Testing TimesFM Stock Predictions")
    print("=" * 50)
    
    # Load model
    print("\n📊 Loading pre-trained TimesFM model...")
    model = TimesFMModel(
        context_len=CONTEXT_LENGTH,
        horizon_len=HORIZON_LENGTH,
        input_patch_len=INPUT_PATCH_LEN
    )
    
    # Load pre-trained weights
    model_path = snapshot_download(repo_id=MODEL_REPO)
    checkpoint_path = os.path.join(model_path, "torch_model.ckpt")
    model.load_pretrained_weights(checkpoint_path)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded on {DEVICE}")
    
    # Load data
    data_dict = load_cached_data()
    
    # Test multiple sequences
    num_tests = 5
    fig, axes = plt.subplots(num_tests, 2, figsize=(20, 4*num_tests))
    if num_tests == 1:
        axes = axes.reshape(1, -1)
    
    for test_idx in range(num_tests):
        # Get a random sequence
        seq_idx = np.random.randint(0, len(data_dict['train_dataset']))
        seq_data = get_sample_sequence(data_dict, seq_idx)
        
        print(f"\n📈 Testing sequence {test_idx+1}/{num_tests}:")
        print(f"   Symbol: {seq_data['symbol']}")
        print(f"   Base price: ${seq_data['base_price']:.2f}")
        
        # Prepare input
        context_tensor = torch.tensor(seq_data['normalized_context'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        freq_tensor = torch.zeros(1, dtype=torch.int32).to(DEVICE)  # Default frequency
        
        # Get prediction
        with torch.no_grad():
            prediction = model(context_tensor, freq_tensor)
        
        prediction_np = prediction.cpu().numpy().squeeze()
        
        # Plot 1: Normalized prices
        ax1 = axes[test_idx, 0]
        
        # Create time axis
        context_time = np.arange(len(seq_data['normalized_context']))
        target_time = np.arange(len(seq_data['normalized_context']), 
                               len(seq_data['normalized_context']) + len(seq_data['normalized_target']))
        
        # Plot context
        ax1.plot(context_time, seq_data['normalized_context'], 'gray', label='History', linewidth=2, alpha=0.7)
        
        # Plot target and prediction
        ax1.plot(target_time, seq_data['normalized_target'], 'black', label='Ground Truth', linewidth=2.5)
        ax1.plot(target_time, prediction_np, 'red', label='TimesFM Prediction', linewidth=2, linestyle='--')
        
        # Add vertical line at prediction start
        ax1.axvline(x=CONTEXT_LENGTH, color='blue', linestyle=':', alpha=0.5)
        
        ax1.set_title(f'Normalized Prices - {seq_data["symbol"]} (Seq {seq_idx})')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Normalized Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculate MSE
        mse = np.mean((prediction_np - seq_data['normalized_target']) ** 2)
        ax1.text(0.02, 0.98, f'MSE: {mse:.6f}', transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Actual dollar prices
        ax2 = axes[test_idx, 1]
        
        # Convert back to dollar prices
        context_dollars = seq_data['raw_context']
        target_dollars = seq_data['raw_target']
        prediction_dollars = prediction_np * seq_data['base_price']
        
        # Plot
        ax2.plot(context_time, context_dollars, 'gray', label='History', linewidth=2, alpha=0.7)
        ax2.plot(target_time, target_dollars, 'black', label='Ground Truth', linewidth=2.5)
        ax2.plot(target_time, prediction_dollars, 'red', label='TimesFM Prediction', linewidth=2, linestyle='--')
        
        # Connect history to predictions
        if len(context_dollars) > 0 and len(target_dollars) > 0:
            ax2.plot([context_time[-1], target_time[0]], 
                    [context_dollars[-1], target_dollars[0]], 
                    'black', linewidth=1, alpha=0.3)
            ax2.plot([context_time[-1], target_time[0]], 
                    [context_dollars[-1], prediction_dollars[0]], 
                    'red', linewidth=1, alpha=0.3)
        
        ax2.axvline(x=CONTEXT_LENGTH, color='blue', linestyle=':', alpha=0.5)
        
        ax2.set_title(f'Dollar Prices - {seq_data["symbol"]} (Seq {seq_idx})')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Calculate dollar MSE
        dollar_mse = np.mean((prediction_dollars - target_dollars) ** 2)
        ax2.text(0.02, 0.98, f'MSE: ${dollar_mse:.2f}', transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Print statistics
        print(f"   Normalized MSE: {mse:.6f}")
        print(f"   Dollar MSE: ${dollar_mse:.2f}")
        print(f"   Mean prediction: {np.mean(prediction_np):.4f} (normalized)")
        print(f"   Std prediction: {np.std(prediction_np):.4f} (normalized)")
    
    plt.tight_layout()
    plt.savefig('timesfm_stock_test.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Test plot saved to timesfm_stock_test.png")
    
    # Additional diagnostic plot
    print("\n📊 Creating diagnostic plot...")
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    
    # Get one more sequence for detailed analysis
    seq_data = get_sample_sequence(data_dict, 0)
    context_tensor = torch.tensor(seq_data['normalized_context'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    freq_tensor = torch.zeros(1, dtype=torch.int32).to(DEVICE)
    
    with torch.no_grad():
        prediction = model(context_tensor, freq_tensor)
    prediction_np = prediction.cpu().numpy().squeeze()
    
    # Plot 1: Input distribution
    ax = axes2[0, 0]
    ax.hist(seq_data['normalized_context'], bins=50, alpha=0.7, label='Context')
    ax.hist(seq_data['normalized_target'], bins=50, alpha=0.7, label='Target')
    ax.hist(prediction_np, bins=50, alpha=0.7, label='Prediction')
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Value Distributions')
    ax.legend()
    
    # Plot 2: Prediction vs Target scatter
    ax = axes2[0, 1]
    ax.scatter(seq_data['normalized_target'], prediction_np, alpha=0.5)
    ax.plot([0.9, 1.1], [0.9, 1.1], 'r--', label='Perfect prediction')
    ax.set_xlabel('Target')
    ax.set_ylabel('Prediction')
    ax.set_title('Prediction vs Target')
    ax.legend()
    
    # Plot 3: Error over time
    ax = axes2[1, 0]
    errors = prediction_np - seq_data['normalized_target']
    ax.plot(errors)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Error Over Prediction Horizon')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Statistics
    ax = axes2[1, 1]
    ax.axis('off')
    stats_text = f"""Model Statistics:
    
Context length: {CONTEXT_LENGTH}
Horizon length: {HORIZON_LENGTH}
Patch size: {INPUT_PATCH_LEN}
Number of patches: {CONTEXT_LENGTH // INPUT_PATCH_LEN}

Data Statistics:
Mean context: {np.mean(seq_data['normalized_context']):.6f}
Std context: {np.std(seq_data['normalized_context']):.6f}
Mean target: {np.mean(seq_data['normalized_target']):.6f}
Std target: {np.std(seq_data['normalized_target']):.6f}
Mean prediction: {np.mean(prediction_np):.6f}
Std prediction: {np.std(prediction_np):.6f}

Performance:
MSE: {mse:.6f}
RMSE: {np.sqrt(mse):.6f}
MAE: {np.mean(np.abs(errors)):.6f}
"""
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('timesfm_diagnostic.png', dpi=150, bbox_inches='tight')
    print("✅ Diagnostic plot saved to timesfm_diagnostic.png")

if __name__ == "__main__":
    test_timesfm_prediction()
