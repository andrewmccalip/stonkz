#!/usr/bin/env python3
"""
Fine-tune TimesFM using our custom PyTorch implementation with proper weight loading.
"""
import os
import pickle
from pathlib import Path
import hashlib
import warnings
import shutil
import argparse
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import both custom model and official TimesFM
from pytorch_timesfm_model import TimesFMModel
import timesfm
from scale_aware_loss import RangePenaltyLoss, ScaleAwareLoss

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv"
CACHE_DIR = SCRIPT_DIR / "dataset_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Data & Training Configuration
CONTEXT_LENGTH = 448    # ~7.5 hours of minute-level historical data (matches good_timefm_stock_working.py)
HORIZON_LENGTH = 64     # Predict 64 minutes ahead (~1 hour) (matches good_timefm_stock_working.py)
INPUT_PATCH_LEN = 32    # Size of each input patch (448 / 32 = 14 patches)
BATCH_SIZE = 64          # Batch size for training (optimized for GPU utilization)
NUM_EPOCHS = 5000        # Max number of training epochs
LEARNING_RATE = 1e-5    # Learning rate for the optimizer
SEQUENCE_STRIDE = 64    # Stride for creating sequences
MAX_SEQUENCES = None   # Maximum number of sequences to use (None for all)

# Model Configuration
MODEL_REPO = "google/timesfm-1.0-200m-pytorch"
SAVE_PATH = "pytorch_timesfm_finetuned.pth"

# Official TimesFM Configuration (for baseline comparison)
TIMESFM_BACKEND = "gpu" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using {TIMESFM_BACKEND} for official TimesFM model")

# Plotting Configuration
PLOT_DIR = SCRIPT_DIR / "finetune_plots"
PLOT_DIR.mkdir(exist_ok=True)
PLOT_EVERY_ITERATIONS = 3000  # Plot every N iterations (reduced for better performance)
PLOT_TIMES_PER_EPOCH = None  # If set, overrides PLOT_EVERY_ITERATIONS. E.g., 4 = plot 4 times per epoch

# Validation Configuration
VAL_TIMES_PER_EPOCH = 1  # How many times to run validation per epoch (1 = only at end of epoch)
CREATE_VAL_PLOTS = True  # Create validation plots during training
VAL_PLOTS_PER_EPOCH = 20  # Number of random validation plots to create per epoch
VAL_PLOT_FREQUENCY = 5   # Create validation plots every N epochs (set to 1 for every epoch)

# Checkpoint Configuration
CHECKPOINT_DIR = SCRIPT_DIR / "finetune_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
USE_LATEST_CHECKPOINT = True  # Set to True to automatically resume from latest checkpoint
 
# Preprocessing Visualization
VISUALIZE_PREPROCESSING = True  # Set to True to create data visualization plots
MONTE_CARLO_SAMPLES = 10  # Number of random sequences to visualize

# Caching Configuration
USE_CACHED_DATA = True  # Set to False to force data reprocessing
CACHE_VERSION = "v3"  # Increment this to invalidate old caches

# Loss Configuration
USE_DIRECTIONAL_LOSS = True  # Enable directional accuracy in loss function
DIRECTIONAL_WEIGHT = 0.5  # Weight for directional component (0.5 = equal weight with MSE)
# ==============================================================================

class DirectionalLoss(nn.Module):
    """Custom loss function that combines MSE with directional accuracy."""
    
    def __init__(self, mse_weight=0.5, directional_weight=0.5, epsilon=1e-6):
        """
        Args:
            mse_weight: Weight for MSE component
            directional_weight: Weight for directional component
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.directional_weight = directional_weight
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Calculate combined loss focusing on directional accuracy.
        
        Args:
            predictions: Model predictions [batch_size, horizon_len]
            targets: Ground truth values [batch_size, horizon_len]
        
        Returns:
            Combined loss value
        """
        # Standard MSE loss
        mse_loss = self.mse_loss(predictions, targets)
        
        # Directional loss component
        # Calculate price changes (direction of movement)
        pred_changes = predictions[:, 1:] - predictions[:, :-1]  # [batch, horizon-1]
        target_changes = targets[:, 1:] - targets[:, :-1]  # [batch, horizon-1]
        
        # Sign agreement loss (1 - accuracy)
        # When signs match, loss is 0; when they don't match, loss is 1
        pred_signs = torch.sign(pred_changes)
        target_signs = torch.sign(target_changes)
        
        # Handle zero changes by treating them as correct predictions
        sign_matches = (pred_signs == target_signs) | (target_signs == 0)
        directional_accuracy = sign_matches.float().mean()
        directional_loss = 1.0 - directional_accuracy
        
        # Correlation-based loss (optional, captures trend alignment)
        # Higher correlation = lower loss
        batch_size = predictions.shape[0]
        correlation_loss = 0.0
        
        for i in range(batch_size):
            pred_i = predictions[i]
            target_i = targets[i]
            
            # Standardize to compute correlation
            pred_std = (pred_i - pred_i.mean()) / (pred_i.std() + self.epsilon)
            target_std = (target_i - target_i.mean()) / (target_i.std() + self.epsilon)
            
            # Pearson correlation
            correlation = (pred_std * target_std).mean()
            # Convert correlation to loss (1 - correlation) / 2 to get [0, 1] range
            correlation_loss += (1.0 - correlation) / 2.0
        
        correlation_loss /= batch_size
        
        # Combine directional and correlation losses
        dir_loss_combined = (directional_loss + correlation_loss) / 2.0
        
        # Final combined loss
        total_loss = self.mse_weight * mse_loss + self.directional_weight * dir_loss_combined
        
        # Store components for logging
        self.last_mse = mse_loss.item()
        self.last_directional = directional_loss.item()
        self.last_correlation = correlation_loss.item()
        self.last_accuracy = directional_accuracy.item()
        
        return total_loss

class StockDataset(Dataset):
    """Dataset for TimesFM fine-tuning with proper patching."""
    
    def __init__(self, data, context_len, horizon_len, patch_len, stride):
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.patch_len = patch_len
        self.num_patches = context_len // patch_len
        self.total_len = context_len + horizon_len
        
        # Group data by instrument_id to prevent mixing contracts
        print(f"📊 Creating sequences grouped by instrument_id...")
        self.sequences = []
        self.all_prices = []
        self.sequence_to_price_idx = []  # Maps sequence index to position in all_prices
        
        # Process each instrument separately
        for instrument_id, instrument_data in data.groupby('instrument_id'):
            if len(instrument_data) < self.total_len:
                continue  # Skip instruments with too little data
                
            instrument_prices = instrument_data['close'].values.astype(np.float32)
            symbol = instrument_data['symbol'].iloc[0]
            
            # Create sequences only within this instrument
            for i in range(0, len(instrument_prices) - self.total_len + 1, stride):
                # Check if we've reached the maximum number of sequences
                if MAX_SEQUENCES is not None and len(self.sequences) >= MAX_SEQUENCES:
                    break
                    
                start_idx = len(self.all_prices)
                self.all_prices.extend(instrument_prices[i:i + self.total_len])
                self.sequence_to_price_idx.append({
                    'start': start_idx,
                    'end': start_idx + self.total_len,
                    'instrument_id': instrument_id,
                    'symbol': symbol,
                    'base_price': instrument_prices[i]  # Use first price of sequence as base
                })
                self.sequences.append(len(self.sequences))
            
            # Break outer loop if we've reached the limit
            if MAX_SEQUENCES is not None and len(self.sequences) >= MAX_SEQUENCES:
                break
        
        self.all_prices = np.array(self.all_prices, dtype=np.float32)
        
        print(f"📊 Created {len(self.sequences)} sequences from {len(data)} data points.")
        if MAX_SEQUENCES is not None and len(self.sequences) >= MAX_SEQUENCES:
            print(f"   ⚠️  Limited to {MAX_SEQUENCES} sequences as configured")
        print(f"📊 Sequences span {len(self.sequence_to_price_idx)} different time windows")
        
        # Show sample of instruments used
        unique_symbols = list(set(seq['symbol'] for seq in self.sequence_to_price_idx))
        print(f"📊 Unique instruments in dataset: {len(unique_symbols)}")
        print(f"   Sample symbols: {unique_symbols[:5]}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_info = self.sequence_to_price_idx[idx]
        start_idx = seq_info['start']
        end_idx = seq_info['end']
        base_price = seq_info['base_price']
        
        # Get raw prices for this sequence
        raw_prices = self.all_prices[start_idx:end_idx]
        
        # Normalize prices relative to the base price of this sequence
        normalized_prices = raw_prices / base_price
        
        # Input sequence (context)
        input_seq = normalized_prices[:self.context_len]
        
        # Reshape into patches [num_patches, patch_len]
        input_patches = input_seq.reshape(self.num_patches, self.patch_len)
        
        # Create patch features (using differences and local statistics)
        patch_features = []
        for i in range(self.num_patches):
            patch = input_patches[i]
            
            # Compute various features for each patch
            features = np.zeros(64)  # Match the expected input dimension
            features[0:self.patch_len] = patch
            features[self.patch_len] = np.mean(patch)
            features[self.patch_len+1] = np.std(patch)
            features[self.patch_len+2] = patch[-1] - patch[0]  # Change
            features[self.patch_len+3] = np.max(patch)
            features[self.patch_len+4] = np.min(patch)
            # Pad remaining features with zeros
            
            patch_features.append(features)
        
        patch_features = np.array(patch_features)
        
        # Target sequence (what we want to predict)
        target_seq = normalized_prices[self.context_len:]
        
        # Debug check for zeros in target
        if idx == 0 and hasattr(self, '_debug_first') == False:
            self._debug_first = True
            print(f"\n  🔍 DEBUG - First dataset sample:")
            print(f"     Context length: {self.context_len}")
            print(f"     Target length: {len(target_seq)}")
            print(f"     Target values at indices 10-15: {target_seq[10:15] if len(target_seq) > 14 else target_seq[10:]}")
            print(f"     Target min: {target_seq.min():.6f}, max: {target_seq.max():.6f}")
            near_zeros = np.sum(np.abs(target_seq) < 1e-6)
            if near_zeros > 0:
                print(f"     ⚠️ WARNING: {near_zeros} near-zero values in target!")
        
        return {
            'input': torch.tensor(patch_features, dtype=torch.float32),
            'target': torch.tensor(target_seq, dtype=torch.float32),
            'freq': torch.tensor(0, dtype=torch.long),  # High frequency indicator
            'instrument_id': seq_info['instrument_id'],
            'symbol': seq_info['symbol'],
            'base_price': base_price
        }
    
    def denormalize(self, normalized_values, base_price):
        """Convert normalized values back to original scale."""
        return normalized_values * base_price

def visualize_monte_carlo_sequences(datasets, num_samples=1000):
    """Create Monte Carlo visualization of randomly sampled sequences."""
    print(f"\n📊 Creating Monte Carlo visualization with {num_samples} sequences...")
    print(f"   Using normalized prices (matching good_timefm_stock_working.py)")
    
    # Create plots directory
    plots_dir = SCRIPT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Combine all datasets for comprehensive sampling
    all_datasets = []
    dataset_names = []
    if 'train_dataset' in datasets:
        all_datasets.append(datasets['train_dataset'])
        dataset_names.append('Train')
    if 'val_dataset' in datasets:
        all_datasets.append(datasets['val_dataset'])
        dataset_names.append('Val')
    if 'test_dataset' in datasets:
        all_datasets.append(datasets['test_dataset'])
        dataset_names.append('Test')
    
    # Create figure with subplots for main visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.ravel()
    
    # Colors for different datasets
    colors = {'Train': 'blue', 'Val': 'green', 'Test': 'red'}
    
    # Plot 1: Monte Carlo overlay - all sequences normalized to start at 1.0
    ax1 = axes[0]
    ax1.set_title(f'Monte Carlo Overlay - {num_samples} Sequences (normalized to 1.0)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Normalized Price')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.8, linewidth=2, label='Starting point (1.0)')
    ax1.axvline(x=datasets['train_dataset'].context_len, color='red', linestyle='--', alpha=0.5, label='Context|Target Split')
    
    # Plot 2: Context sequences only
    ax2 = axes[1]
    ax2.set_title(f'Context Sequences Only', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Normalized Price')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Target sequences only
    ax3 = axes[2]
    ax3.set_title('Target Sequences Only (Predictions)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time Steps (from prediction start)')
    ax3.set_ylabel('Normalized Price')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics
    ax4 = axes[3]
    ax4.set_title('Sequence Statistics', fontsize=14, fontweight='bold')
    
    # Sample sequences from each dataset
    samples_per_dataset = num_samples // len(all_datasets)
    all_contexts = []
    all_targets = []
    all_full_sequences = []
    sequence_count = 0
    
    for dataset, name in zip(all_datasets, dataset_names):
        # Random sampling
        total_sequences = len(dataset)
        sample_indices = np.random.choice(total_sequences, 
                                        min(samples_per_dataset, total_sequences), 
                                        replace=False)
        
        for idx in sample_indices:
            sample = dataset[idx]
            
            # Get context and target
            input_patches = sample['input'].numpy()  # [num_patches, patch_features]
            target = sample['target'].numpy()  # [horizon_len]
            
            # Reconstruct context from patches (taking just the price values)
            context = []
            for i in range(dataset.num_patches):
                patch_prices = input_patches[i, :dataset.patch_len]
                context.extend(patch_prices)
            context = np.array(context[:dataset.context_len])
            
            # Normalize context to start at 1.0
            if len(context) > 0 and context[0] != 0:
                normalization_factor = 1.0 / context[0]
                normalized_context = context * normalization_factor
                normalized_target = target * normalization_factor
                
                # Plot full sequence in main Monte Carlo overlay (ax1)
                full_seq = np.concatenate([normalized_context, normalized_target])
                full_x = np.arange(len(full_seq))
                ax1.plot(full_x, full_seq, alpha=0.3, color=colors[name], linewidth=0.8)
                
                # Plot context only (ax2)
                context_x = np.arange(len(normalized_context))
                ax2.plot(context_x, normalized_context, alpha=0.3, color=colors[name], linewidth=0.8)
                
                # Plot target only (ax3)
                target_x = np.arange(len(normalized_target))
                ax3.plot(target_x, normalized_target, alpha=0.3, color=colors[name], linewidth=0.8)
                
                # Store for statistics
                all_contexts.append(normalized_context)
                all_targets.append(normalized_target)
                all_full_sequences.append(full_seq)
                
                # Create individual plot for this sequence
                sequence_count += 1
                fig_seq, ax_seq = plt.subplots(1, 1, figsize=(12, 6))
                
                # Get instrument info for this sequence
                instrument_id = sample.get('instrument_id', 'Unknown')
                symbol = sample.get('symbol', 'Unknown')
                base_price = sample.get('base_price', 1.0)
                
                # Plot the full sequence
                ax_seq.plot(full_x, full_seq, color=colors[name], linewidth=2, label=f'{name} sequence')
                ax_seq.axvline(x=dataset.context_len, color='red', linestyle='--', alpha=0.5, label='Context|Target')
                ax_seq.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, label='Starting point (1.0)')
                
                # Add grid and labels
                ax_seq.grid(True, alpha=0.3)
                ax_seq.set_xlabel('Time Steps')
                ax_seq.set_ylabel('Normalized Price')
                ax_seq.set_title(f'Sequence {sequence_count:04d} - {symbol} (ID: {instrument_id}) - {name} Dataset')
                ax_seq.legend()
                
                # Add some statistics text
                context_end_val = normalized_context[-1]
                target_end_val = normalized_target[-1]
                context_change = (context_end_val - 1.0) * 100
                target_change = (target_end_val - context_end_val) * 100
                total_change = (target_end_val - 1.0) * 100
                
                stats_str = (f'Symbol: {symbol} | Base Price: ${base_price:.2f}\n'
                           f'Context change: {context_change:.2f}% | Target change: {target_change:.2f}% | Total: {total_change:.2f}%')
                ax_seq.text(0.02, 0.98, stats_str, transform=ax_seq.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Save individual plot
                seq_plot_path = plots_dir / f'sequence_{sequence_count:04d}_{name.lower()}_{idx}.png'
                plt.savefig(seq_plot_path, dpi=100, bbox_inches='tight')
                plt.close(fig_seq)
    
    # Calculate and plot statistics
    all_contexts = np.array(all_contexts)
    all_targets = np.array(all_targets)
    
    # Calculate mean and std for both context and targets
    context_mean = np.mean(all_contexts, axis=0)
    context_std = np.std(all_contexts, axis=0)
    context_x = np.arange(len(context_mean))
    
    target_mean = np.mean(all_targets, axis=0)
    target_std = np.std(all_targets, axis=0)
    target_x = np.arange(len(target_mean))
    
    # Add mean line to the Monte Carlo overlay plot
    full_mean = np.concatenate([context_mean, target_mean])
    ax1.plot(full_mean, 'black', linewidth=3, label='Mean trajectory', zorder=1000)
    
    # Add legends for ax1
    for name, color in colors.items():
        ax1.plot([], [], color=color, alpha=0.5, linewidth=2, label=f'{name} sequences')
    ax1.legend(loc='best')
    
    # Add mean and std bands to context plot (ax2)
    ax2.plot(context_mean, 'black', linewidth=2, label='Mean', zorder=100)
    ax2.fill_between(context_x,
                     context_mean - context_std,
                     context_mean + context_std,
                     alpha=0.3, color='gray', label='±1 STD')
    ax2.legend()
    
    # Add mean and std bands to target plot (ax3)
    ax3.plot(target_mean, 'black', linewidth=2, label='Mean', zorder=100)
    ax3.fill_between(target_x,
                     target_mean - target_std,
                     target_mean + target_std,
                     alpha=0.3, color='gray', label='±1 STD')
    ax3.legend()
    
    # Statistics text
    stats_text = f"""Dataset Statistics:
    
Total Sequences Sampled: {len(all_contexts)}
- Train: {sum(1 for d in dataset_names if d == 'Train') * samples_per_dataset}
- Val: {sum(1 for d in dataset_names if d == 'Val') * samples_per_dataset}
- Test: {sum(1 for d in dataset_names if d == 'Test') * samples_per_dataset}

Context Length: {datasets['train_dataset'].context_len}
Horizon Length: {datasets['train_dataset'].horizon_len}
Patch Length: {datasets['train_dataset'].patch_len}

Context Statistics (normalized):
- Start: 1.00 (by design)
- End Mean: {context_mean[-1]:.4f} ± {context_std[-1]:.4f}
- Min Mean: {np.min(context_mean):.4f}
- Max Mean: {np.max(context_mean):.4f}

Target Statistics (normalized):
- Start Mean: {target_mean[0]:.4f} ± {target_std[0]:.4f}
- End Mean: {target_mean[-1]:.4f} ± {target_std[-1]:.4f}
- Min Mean: {np.min(target_mean):.4f}
- Max Mean: {np.max(target_mean):.4f}

Price Movement Statistics:
- Avg Context Change: {(context_mean[-1] - 1.0) * 100:.2f}%
- Avg Target Change: {(target_mean[-1] - target_mean[0]) * 100:.2f}%
- Max Observed Change: {(np.max(all_full_sequences) - 1.0) * 100:.2f}%
- Min Observed Change: {(np.min(all_full_sequences) - 1.0) * 100:.2f}%
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    

    
    plt.tight_layout()
    
    # Save plot to plots directory
    viz_path = plots_dir / 'preprocessing_monte_carlo.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Monte Carlo visualization saved to: {viz_path}")
    print(f"   - Sampled {len(all_contexts)} sequences")
    print(f"   - Context sequences normalized to start at 1.0")
    print(f"   - Shows mean ± 1 STD bands")
    print(f"✅ Individual sequence plots saved to: {plots_dir}")
    print(f"   - Created {sequence_count} individual plots")
    print(f"   - Files: sequence_0001_*.png through sequence_{sequence_count:04d}_*.png")

def load_and_prepare_data(force_reprocess=False):
    """Load and prepare stock data for training.
    
    Args:
        force_reprocess: If True, ignore cache and reprocess data
    """
    # Create cache key including version and configuration
    cache_key_str = f"{DATASET_PATH}_{CONTEXT_LENGTH}_{HORIZON_LENGTH}_{INPUT_PATCH_LEN}_{SEQUENCE_STRIDE}_{MAX_SEQUENCES}_{CACHE_VERSION}"
    cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
    cache_file = CACHE_DIR / f"pytorch_data_{cache_key}.pkl"
    
    # Check if we should use cached data
    use_cache = USE_CACHED_DATA and not force_reprocess
    
    if cache_file.exists() and use_cache:
        print("📦 Loading cached prepared data...")
        print(f"   Cache file: {cache_file.name}")
        print(f"   Cache size: {cache_file.stat().st_size / (1024**2):.2f} MB")
        
        try:
            with open(cache_file, 'rb') as f:
                data_dict = pickle.load(f)
            
            # Verify cache contents
            print(f"   ✅ Loaded cached data successfully!")
            print(f"   Train sequences: {len(data_dict['train_dataset'])}")
            print(f"   Val sequences: {len(data_dict['val_dataset'])}")
            print(f"   Test sequences: {len(data_dict['test_dataset'])}")
            return data_dict
        except Exception as e:
            print(f"   ⚠️ Failed to load cache: {e}")
            print(f"   Proceeding with data reprocessing...")
    elif force_reprocess:
        print("🔄 Force reprocessing flag set - ignoring cache")
    elif not USE_CACHED_DATA:
        print("🔄 Cache disabled (USE_CACHED_DATA=False) - processing data")
    
    print(f"📊 Loading data from {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df['timestamp'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Filter for ES futures symbols (like in finetuning.py)
    print("🔍 Filtering for ES futures symbols...")
    original_rows = len(df)
    
    # Filter criteria:
    # 1. Exactly 4 characters
    # 2. Starts with 'ES' 
    # 3. No dashes (excludes spreads like ESH4-ESM4)
    # 4. Third character is a month code (H, M, U, Z)
    # 5. Fourth character is a digit (year)
    month_codes = ['H', 'M', 'U', 'Z']  # Mar, Jun, Sep, Dec
    
    mask = (
        (df['symbol'].str.len() == 4) &
        (df['symbol'].str.startswith('ES')) &
        (~df['symbol'].str.contains('-')) &
        (df['symbol'].str[2].isin(month_codes)) &
        (df['symbol'].str[3].str.isdigit())
    )
    
    df = df[mask].copy()
    print(f"   Filtered from {original_rows:,} to {len(df):,} rows (ES futures only)")
    
    # Group by instrument_id to ensure continuity within sequences
    print("🔍 Grouping by instrument_id to prevent mixing contracts...")
    df = df.sort_values(['instrument_id', 'timestamp']).reset_index(drop=True)
    
    # Show unique instruments
    unique_instruments = df[['instrument_id', 'symbol']].drop_duplicates()
    print(f"   Found {len(unique_instruments)} unique instruments")
    print(f"   Sample instruments: {unique_instruments.head(10).to_string()}")
    
    df = df.dropna(subset=['close'])
    
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    train_data = df[:train_end]
    val_data = df[train_end:val_end]
    test_data = df[val_end:]
    
    print(f"📈 Data splits:")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val:   {len(val_data)} samples")
    print(f"   Test:  {len(test_data)} samples")
    
    train_dataset = StockDataset(train_data, CONTEXT_LENGTH, HORIZON_LENGTH, INPUT_PATCH_LEN, SEQUENCE_STRIDE)
    val_dataset = StockDataset(val_data, CONTEXT_LENGTH, HORIZON_LENGTH, INPUT_PATCH_LEN, SEQUENCE_STRIDE)
    test_dataset = StockDataset(test_data, CONTEXT_LENGTH, HORIZON_LENGTH, INPUT_PATCH_LEN, SEQUENCE_STRIDE)
    
    # Note: DataLoaders will be recreated in main() with the correct batch size
    # For now, we just store the datasets in the cache
    train_loader = None
    val_loader = None
    test_loader = None
    
    data_dict = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }
    
    # Save to cache
    print(f"\n💾 Saving processed data to cache...")
    print(f"   Cache file: {cache_file.name}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data_dict, f)
        cache_size = cache_file.stat().st_size / (1024**2)
        print(f"   ✅ Cache saved successfully! Size: {cache_size:.2f} MB")
        print(f"   ℹ️  To force reprocessing, set USE_CACHED_DATA=False or use --force-reprocess flag")
    except Exception as e:
        print(f"   ⚠️ Failed to save cache: {e}")
    
    return data_dict

def create_validation_plots(model, val_loader, epoch, num_plots=20, official_model=None):
    """Create random validation plots showing history/ground truth/prediction with official TimesFM comparison."""
    print(f"\n📊 Creating {num_plots} validation plots for epoch {epoch}...")
    
    # Create validation plots directory
    val_plots_dir = PLOT_DIR / "validation"
    val_plots_dir.mkdir(exist_ok=True)
    
    # Clear previous epoch's validation plots
    for old_plot in val_plots_dir.glob(f"epoch_{epoch:03d}_*.png"):
        old_plot.unlink()
    
    # Initialize official TimesFM model if not provided
    if official_model is None:
        print("   🤖 Initializing official TimesFM model for comparison...")
        try:
            import timesfm
            official_model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=TIMESFM_BACKEND,
                    per_core_batch_size=32,
                    horizon_len=128,  # Model's default horizon
                    num_layers=20,
                    use_positional_embedding=True,
                    context_len=512,  # Model's default context length
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=MODEL_REPO
                ),
            )
            print("   ✅ Official TimesFM model loaded for validation plots")
        except Exception as e:
            print(f"   ⚠️ Failed to load official TimesFM model: {e}")
            official_model = None
    
    model.eval()
    with torch.no_grad():
        # Collect all validation samples
        all_samples = []
        for batch_idx, batch in enumerate(val_loader):
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            freq = batch['freq'].to(DEVICE)
            
            # Get predictions
            predictions = model(inputs, freq)
            
            # Store samples with metadata
            for i in range(inputs.shape[0]):
                sample_data = {
                    'input': inputs[i].cpu(),
                    'target': targets[i].cpu(),
                    'prediction': predictions[i].cpu(),
                    'freq': freq[i].cpu(),
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'instrument_id': batch.get('instrument_id', ['Unknown'])[i] if 'instrument_id' in batch else 'Unknown',
                    'symbol': batch.get('symbol', ['Unknown'])[i] if 'symbol' in batch else 'Unknown',
                    'base_price': batch.get('base_price', [1.0])[i] if 'base_price' in batch else 1.0
                }
                all_samples.append(sample_data)
        
        # Randomly select samples to plot
        if len(all_samples) < num_plots:
            selected_samples = all_samples
            print(f"   ⚠️ Only {len(all_samples)} validation samples available, plotting all")
        else:
            import random
            selected_samples = random.sample(all_samples, num_plots)
        
        print(f"   📈 Creating {len(selected_samples)} individual validation plots...")
        
        # Create individual plots
        for plot_idx, sample in enumerate(selected_samples):
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))
            
            # Extract data
            input_patches = sample['input'].numpy()  # [num_patches, patch_features]
            target = sample['target'].numpy()  # [horizon_len]
            prediction = sample['prediction'].numpy()  # [horizon_len]
            
            # Reconstruct context from patches (taking just the price values)
            dataset = val_loader.dataset
            context = []
            for i in range(dataset.num_patches):
                patch_prices = input_patches[i, :dataset.patch_len]
                context.extend(patch_prices)
            context = np.array(context[:dataset.context_len])
            
            # Create x-axis
            context_x = np.arange(len(context))
            prediction_x = np.arange(len(context), len(context) + len(target))
            
            # Get official TimesFM prediction if available
            official_prediction = None
            if official_model is not None:
                try:
                    # Official TimesFM expects list of lists
                    inputs_list = [context.tolist()]
                    freq_list = [0]  # High frequency indicator
                    official_forecast, _ = official_model.forecast(inputs_list, freq_list)
                    official_prediction = np.array(official_forecast[0][:HORIZON_LENGTH])  # Limit to our horizon
                except Exception as e:
                    print(f"   ⚠️ Official TimesFM prediction failed for sample {plot_idx+1}: {e}")
                    # Create a simple baseline as fallback
                    if len(context) > 0:
                        last_val = context[-1]
                        trend = (context[-1] - context[-10]) / 10 if len(context) >= 10 else 0
                        official_prediction = np.array([last_val + trend * i for i in range(1, HORIZON_LENGTH + 1)])
                    else:
                        official_prediction = None
            
            # Plot context (history) in blue
            ax.plot(context_x, context, 'b-', linewidth=2, label='History (Context)', alpha=0.8)
            
            # Plot ground truth in black
            ax.plot(prediction_x, target, 'k-', linewidth=3, label='Ground Truth', alpha=0.9)
            
            # Plot fine-tuned prediction in red
            ax.plot(prediction_x, prediction, 'r-', linewidth=2, label='Fine-tuned TimesFM', alpha=0.8)
            
            # Plot official TimesFM prediction in orange if available
            if official_prediction is not None:
                ax.plot(prediction_x, official_prediction, 'orange', linewidth=2, label='Official TimesFM', alpha=0.7, linestyle='--')
            
            # Connect context to predictions with thin lines
            if len(context) > 0 and len(target) > 0:
                ax.plot([context_x[-1], prediction_x[0]], [context[-1], target[0]], 'k-', linewidth=1, alpha=0.3)
                ax.plot([context_x[-1], prediction_x[0]], [context[-1], prediction[0]], 'r-', linewidth=1, alpha=0.3)
                if official_prediction is not None:
                    ax.plot([context_x[-1], prediction_x[0]], [context[-1], official_prediction[0]], 'orange', linewidth=1, alpha=0.3)
            
            # Add vertical line to separate context from predictions
            ax.axvline(x=len(context), color='gray', linestyle='--', alpha=0.5, label='Prediction Start')
            
            # Calculate metrics for fine-tuned model
            finetuned_mse = np.mean((prediction - target) ** 2)
            finetuned_mae = np.mean(np.abs(prediction - target))
            
            # Directional accuracy for fine-tuned model
            if len(target) > 1:
                target_direction = np.sign(np.diff(target))
                finetuned_pred_direction = np.sign(np.diff(prediction))
                finetuned_dir_accuracy = np.mean(target_direction == finetuned_pred_direction) * 100
            else:
                finetuned_dir_accuracy = 50.0
            
            # Correlation for fine-tuned model
            if np.std(prediction) > 1e-6 and np.std(target) > 1e-6:
                finetuned_correlation = np.corrcoef(prediction, target)[0, 1]
            else:
                finetuned_correlation = 0.0
            
            # Calculate metrics for official model if available
            official_mse = official_mae = official_dir_accuracy = official_correlation = None
            if official_prediction is not None:
                official_mse = np.mean((official_prediction - target) ** 2)
                official_mae = np.mean(np.abs(official_prediction - target))
                
                if len(target) > 1:
                    official_pred_direction = np.sign(np.diff(official_prediction))
                    official_dir_accuracy = np.mean(target_direction == official_pred_direction) * 100
                else:
                    official_dir_accuracy = 50.0
                
                if np.std(official_prediction) > 1e-6 and np.std(target) > 1e-6:
                    official_correlation = np.corrcoef(official_prediction, target)[0, 1]
                else:
                    official_correlation = 0.0
            
            # Price changes
            context_change = (context[-1] - context[0]) / context[0] * 100 if len(context) > 0 and context[0] != 0 else 0
            target_change = (target[-1] - target[0]) / target[0] * 100 if len(target) > 0 and target[0] != 0 else 0
            finetuned_pred_change = (prediction[-1] - prediction[0]) / prediction[0] * 100 if len(prediction) > 0 and prediction[0] != 0 else 0
            official_pred_change = (official_prediction[-1] - official_prediction[0]) / official_prediction[0] * 100 if official_prediction is not None and len(official_prediction) > 0 and official_prediction[0] != 0 else 0
            
            # Formatting and labels
            ax.set_xlabel('Time Steps (Minutes)', fontsize=12)
            ax.set_ylabel('Normalized Price', fontsize=12)
            ax.set_title(f'Validation Sample {plot_idx+1:02d} - Epoch {epoch} - {sample["symbol"]} (ID: {sample["instrument_id"]})', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add metrics text box with model comparison
            if official_prediction is not None:
                # Show comparison metrics
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
                    f'Base Price: ${sample["base_price"]:.2f}'
                )
            else:
                # Fallback to single model metrics
                metrics_text = (
                    f'Fine-tuned TimesFM Metrics:\n'
                    f'MSE: {finetuned_mse:.6f}\n'
                    f'MAE: {finetuned_mae:.6f}\n'
                    f'Dir. Accuracy: {finetuned_dir_accuracy:.1f}%\n'
                    f'Correlation: {finetuned_correlation:.3f}\n\n'
                    f'Price Changes:\n'
                    f'Context: {context_change:.2f}%\n'
                    f'Target: {target_change:.2f}%\n'
                    f'Predicted: {finetuned_pred_change:.2f}%\n\n'
                    f'Base Price: ${sample["base_price"]:.2f}\n\n'
                    f'(Official TimesFM unavailable)'
                )
            
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10, fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
            
            # Set reasonable y-axis limits for normalized prices
            all_values = np.concatenate([context, target, prediction])
            y_min, y_max = np.min(all_values), np.max(all_values)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            # Save plot
            plot_filename = f"epoch_{epoch:03d}_sample_{plot_idx+1:02d}_{sample['symbol']}.png"
            plot_path = val_plots_dir / plot_filename
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
        
        print(f"   ✅ Validation plots saved to: {val_plots_dir}")
        print(f"   📁 Files: epoch_{epoch:03d}_sample_01_*.png through epoch_{epoch:03d}_sample_{len(selected_samples):02d}_*.png")
    
    model.train()  # Set back to training mode

def create_training_plots(history, predictions_sample=None, targets_sample=None, iteration=0, 
                         model=None, official_model=None, sample_context=None, sample_freq=None, dataset=None):
    """Create and save training progress plots."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0, 0]
    if len(history['train_loss']) > 0:
        ax1.plot(history['train_loss'], label='Train Loss', alpha=0.8)
        ax1.plot(history['val_loss'], label='Val Loss', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
    
    # Plot 2: Validation Statistics
    ax2 = axes[0, 1]
    if USE_DIRECTIONAL_LOSS and 'val_dir_accuracy' in history and len(history['val_dir_accuracy']) > 0:
        # Create twin axis for correlation
        ax2_twin = ax2.twinx()
        
        # Plot directional accuracy on left axis
        epochs = np.arange(1, len(history['val_dir_accuracy']) + 1)
        line1 = ax2.plot(epochs, np.array(history['val_dir_accuracy']) * 100, 'b-', 
                         label='Val Dir. Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Directional Accuracy (%)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_ylim(0, 100)
        
        # Plot correlation on right axis
        if 'val_correlation' in history and len(history['val_correlation']) > 0:
            line2 = ax2_twin.plot(epochs, history['val_correlation'], 'r-', 
                                 label='Val Correlation', linewidth=2)
            ax2_twin.set_ylabel('Correlation', color='r')
            ax2_twin.tick_params(axis='y', labelcolor='r')
            ax2_twin.set_ylim(-1, 1)
        
        # Add baseline at 50% for directional accuracy
        ax2.axhline(y=50, color='b', linestyle='--', alpha=0.3)
        
        # Combined legend
        lines = line1
        if 'val_correlation' in history:
            lines += line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        
        ax2.set_title('Validation Metrics')
        ax2.grid(True, alpha=0.3)
    else:
        # Fallback to learning rate if not using directional loss
        if 'learning_rates' in history and len(history['learning_rates']) > 0:
            ax2.plot(history['learning_rates'])
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
    
    # Plot 3: Recent Loss (last 100 iterations) or Directional Accuracy
    ax3 = axes[1, 0]
    if USE_DIRECTIONAL_LOSS and 'directional_accuracy' in history and len(history['directional_accuracy']) > 0:
        # Show directional accuracy when using directional loss
        recent_accuracy = np.array(history['directional_accuracy'][-1000:]) * 100  # Last 1000 iterations, convert to percentage
        ax3.plot(recent_accuracy, alpha=0.7, color='green', label='Directional Accuracy')
        
        # Add range violations if available
        if 'range_violations' in history and len(history['range_violations']) > 0:
            recent_violations = np.array(history['range_violations'][-1000:]) * 100
            ax3_twin = ax3.twinx()
            ax3_twin.plot(recent_violations, alpha=0.7, color='red', label='Range Violations %')
            ax3_twin.set_ylabel('Range Violations (%)', color='red')
            ax3_twin.tick_params(axis='y', labelcolor='red')
            ax3_twin.set_ylim(0, 100)
        
        ax3.set_xlabel('Iteration (recent)')
        ax3.set_ylabel('Directional Accuracy (%)', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        ax3.set_title(f'Recent Metrics (last {len(recent_accuracy)} iterations)')
        ax3.set_ylim(0, 100)
        ax3.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Random (50%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
    elif 'iteration_losses' in history and len(history['iteration_losses']) > 0:
        recent_losses = history['iteration_losses'][-1000:]  # Last 1000 iterations
        ax3.plot(recent_losses, alpha=0.7)
        ax3.set_xlabel('Iteration (recent)')
        ax3.set_ylabel('Loss')
        ax3.set_title(f'Recent Loss (last {len(recent_losses)} iterations)')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sample Predictions vs Targets
    ax4 = axes[1, 1]
    if predictions_sample is not None and targets_sample is not None:
        # Take first sample from batch
        pred = predictions_sample[0].cpu().numpy()
        target = targets_sample[0].cpu().numpy()
        
        x = np.arange(len(pred))
        ax4.plot(x, target, label='Target', alpha=0.8, linewidth=2)
        ax4.plot(x, pred, label='Prediction', alpha=0.8, linewidth=2)
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Normalized Price')
        ax4.set_title('Sample Prediction vs Target')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        # Set fixed y-axis scale for normalized prices
        ax4.set_ylim(0.98, 1.02)
    
    # Plot 5: Official TimesFM vs Fine-tuned Model Comparison
    ax5 = axes[0, 2]
    ax5.set_title('Model Comparison: Official TimesFM vs Fine-tuned', fontweight='bold')
    ax5.set_xlabel('Time Steps')
    ax5.set_ylabel('Normalized Price')
    ax5.grid(True, alpha=0.3)
    
    if model is not None and official_model is not None and sample_context is not None and targets_sample is not None:
        try:
            # Get predictions from both models
            with torch.no_grad():
                model.eval()
                
                # Fine-tuned model prediction (using sample_input which contains patches)
                if hasattr(model, '__call__'):
                    finetuned_pred = model(sample_context, sample_freq)
                
                model.train()  # Set back to training mode
            
            # Get context data for official TimesFM
            if dataset is not None:
                # Reconstruct the full context from the dataset sample
                sample_data = dataset[0]  # Get first sample
                context_prices = []
                for i in range(dataset.num_patches):
                    patch_prices = sample_context[0, i, :dataset.patch_len].cpu().numpy()
                    context_prices.extend(patch_prices)
                context = np.array(context_prices[:dataset.context_len])
            else:
                # Fallback: use flattened context
                context = sample_context[0].cpu().numpy().flatten()[:CONTEXT_LENGTH]
            
            # Official TimesFM prediction
            try:
                inputs = [context.tolist()]  # Batch of 1
                freq = [0]  # High frequency indicator
                official_forecast, _ = official_model.forecast(inputs, freq)
                official_pred = official_forecast[0][:HORIZON_LENGTH]  # Extract first series, limit to horizon
            except Exception as e:
                print(f"⚠️ Official TimesFM error in visualization: {e}")
                # Use a simple baseline as fallback
                last_val = context[-1]
                trend = (context[-1] - context[-10]) / 10 if len(context) >= 10 else 0
                official_pred = np.array([last_val + trend * i for i in range(1, HORIZON_LENGTH + 1)])
            
            # Get target and finetuned predictions
            target = targets_sample[0].cpu().numpy()  # Shape: [HORIZON_LENGTH]
            finetuned = finetuned_pred[0].cpu().numpy()  # Shape: [HORIZON_LENGTH]
            
            # Debug: Detailed analysis of predictions
            print(f"\n  🔍 DEBUG - Model Comparison Analysis:")
            print(f"     Context shape: {context.shape}, last value: {context[-1]:.6f}")
            print(f"     Official TimesFM prediction:")
            print(f"       First 10: {official_pred[:10]}")
            print(f"       Around index 12: {official_pred[10:15] if len(official_pred) > 14 else official_pred[10:]}")
            print(f"       Last 10: {official_pred[-10:]}")
            print(f"     Fine-tuned model prediction:")
            print(f"       First 10: {finetuned[:10]}")
            print(f"       Around index 12: {finetuned[10:15] if len(finetuned) > 14 else finetuned[10:]}")
            print(f"       Last 10: {finetuned[-10:]}")
            
            # Check for anomalies
            finetuned_zeros = np.sum(np.abs(finetuned) < 1e-6)
            official_zeros = np.sum(np.abs(official_pred) < 1e-6)
            if finetuned_zeros > 0:
                print(f"     ⚠️ Fine-tuned has {finetuned_zeros} near-zero values!")
            if official_zeros > 0:
                print(f"     ⚠️ Official has {official_zeros} near-zero values!")
            
            # Create x-axis for the full sequence
            context_x = np.arange(len(context))
            prediction_x = np.arange(len(context), len(context) + len(target))
            
            # Plot context (history) in gray
            ax5.plot(context_x, context, 'gray', label='History (Context)', linewidth=2, alpha=0.6)
            
            # Connect context to predictions with a thin line
            if len(context) > 0 and len(target) > 0:
                # Plot ground truth
                ax5.plot([context_x[-1], prediction_x[0]], [context[-1], target[0]], 'black', linewidth=1, alpha=0.5)
                ax5.plot(prediction_x, target, 'black', label='Ground Truth', linewidth=2.5, alpha=0.9)
                
                # Plot official TimesFM prediction
                ax5.plot([context_x[-1], prediction_x[0]], [context[-1], official_pred[0]], 'red', linewidth=1, alpha=0.3)
                ax5.plot(prediction_x, official_pred, 'red', label='Official TimesFM', linewidth=2, alpha=0.7, linestyle='--')
                
                # Plot fine-tuned prediction
                ax5.plot([context_x[-1], prediction_x[0]], [context[-1], finetuned[0]], 'green', linewidth=1, alpha=0.3)
                ax5.plot(prediction_x, finetuned, 'green', label='Fine-tuned TimesFM', linewidth=2, alpha=0.8)
            
            # Add vertical line to separate context from predictions
            ax5.axvline(x=len(context), color='blue', linestyle=':', alpha=0.5, label='Prediction Start')
            
            # Calculate MSE and directional accuracy metrics
            official_mse = np.mean((official_pred - target) ** 2)
            finetuned_mse = np.mean((finetuned - target) ** 2)
            improvement = ((official_mse - finetuned_mse) / official_mse) * 100 if official_mse > 0 else 0
            
            # Calculate directional accuracy (% of correct direction predictions)
            # Direction is based on change from one timestep to the next
            target_direction = np.sign(np.diff(target))
            official_direction = np.sign(np.diff(official_pred))
            finetuned_direction = np.sign(np.diff(finetuned))
            
            official_dir_accuracy = np.mean(official_direction == target_direction) * 100
            finetuned_dir_accuracy = np.mean(finetuned_direction == target_direction) * 100
            
            # Calculate trend correlation
            official_corr = np.corrcoef(official_pred, target)[0, 1]
            finetuned_corr = np.corrcoef(finetuned, target)[0, 1]
            
            stats_text = (f'Prediction MSE:\n'
                         f'Official: {official_mse:.6f} | Fine-tuned: {finetuned_mse:.6f}\n'
                         f'MSE Improvement: {improvement:.1f}%\n\n'
                         f'Directional Accuracy:\n'
                         f'Official: {official_dir_accuracy:.1f}% | Fine-tuned: {finetuned_dir_accuracy:.1f}%\n'
                         f'Correlation:\n'
                         f'Official: {official_corr:.3f} | Fine-tuned: {finetuned_corr:.3f}')
            
            ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax5.legend(loc='lower right')
            ax5.set_xlim(-10, len(context) + len(target) + 10)
            
            # Set fixed y-axis scale for normalized prices (0.98 to 1.02)
            ax5.set_ylim(0.98, 1.02)
        except Exception as e:
            ax5.text(0.5, 0.5, f'Error generating comparison:\n{str(e)}', 
                    transform=ax5.transAxes, ha='center', va='center')
    else:
        ax5.text(0.5, 0.5, 'Model comparison not available', 
                transform=ax5.transAxes, ha='center', va='center', fontsize=12)
    
    # Plot 6: Prediction Error Distribution
    ax6 = axes[1, 2]
    if predictions_sample is not None and targets_sample is not None:
        errors = predictions_sample.cpu().numpy() - targets_sample.cpu().numpy()
        ax6.hist(errors.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax6.set_xlabel('Prediction Error')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Prediction Error Distribution')
        ax6.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax6.text(0.02, 0.98, f'Mean: {mean_error:.6f}\nStd: {std_error:.6f}', 
                transform=ax6.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax6.text(0.5, 0.5, 'No error data available', 
                transform=ax6.transAxes, ha='center', va='center')
    
    # Plot 7: Validation Accuracy vs Epoch
    ax7 = axes[2, 0]
    if 'val_dir_accuracy' in history and len(history['val_dir_accuracy']) > 0:
        epochs = np.arange(1, len(history['val_dir_accuracy']) + 1)
        val_acc = np.array(history['val_dir_accuracy']) * 100
        ax7.plot(epochs, val_acc, 'b-', linewidth=2, marker='o', markersize=4)
        ax7.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Validation Directional Accuracy (%)')
        ax7.set_title('Validation Accuracy Progress')
        ax7.set_ylim(0, 100)
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        
        # Add text with latest value
        if len(val_acc) > 0:
            latest_acc = val_acc[-1]
            ax7.text(0.95, 0.05, f'Latest: {latest_acc:.1f}%', 
                    transform=ax7.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax7.text(0.5, 0.5, 'No validation accuracy data yet', 
                transform=ax7.transAxes, ha='center', va='center')
    
    # Plot 8: Range Violations vs Iteration
    ax8 = axes[2, 1]
    if 'range_violations' in history and len(history['range_violations']) > 0:
        violations = np.array(history['range_violations']) * 100
        iterations = np.arange(len(violations))
        ax8.plot(iterations, violations, 'r-', alpha=0.7, linewidth=1)
        ax8.set_xlabel('Iteration')
        ax8.set_ylabel('Range Violations (%)')
        ax8.set_title(f'Predictions Outside [{0.98:.2f}, {1.02:.2f}] Range')
        ax8.set_ylim(0, 100)
        ax8.grid(True, alpha=0.3)
        
        # Add smoothed trend
        if len(violations) > 100:
            window = min(100, len(violations) // 10)
            smoothed = np.convolve(violations, np.ones(window)/window, mode='valid')
            smooth_x = np.arange(len(smoothed))
            ax8.plot(smooth_x + window//2, smoothed, 'darkred', linewidth=2, label='Smoothed trend')
            ax8.legend()
        
        # Add text with latest value
        if len(violations) > 0:
            latest_viol = violations[-1]
            ax8.text(0.95, 0.95, f'Latest: {latest_viol:.1f}%', 
                    transform=ax8.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax8.text(0.5, 0.5, 'No range violation data\n(RangePenaltyLoss not active)', 
                transform=ax8.transAxes, ha='center', va='center')
    
    # Plot 9: Combined Metrics Summary
    ax9 = axes[2, 2]
    ax9.axis('off')
    
    # Create summary text
    summary_parts = ['📊 Training Summary\n' + '='*30]
    
    if iteration > 0:
        summary_parts.append(f'\nIteration: {iteration:,}')
    
    if len(history['train_loss']) > 0:
        summary_parts.append(f'Epochs: {len(history["train_loss"])}')
        summary_parts.append(f'Best Val Loss: {min(history["val_loss"]):.6f}')
    
    if 'directional_accuracy' in history and len(history['directional_accuracy']) > 0:
        recent_dir_acc = np.mean(history['directional_accuracy'][-100:]) * 100
        summary_parts.append(f'\nRecent Dir Acc: {recent_dir_acc:.1f}%')
    
    if 'val_dir_accuracy' in history and len(history['val_dir_accuracy']) > 0:
        best_val_acc = max(history['val_dir_accuracy']) * 100
        summary_parts.append(f'Best Val Dir Acc: {best_val_acc:.1f}%')
    
    if 'range_violations' in history and len(history['range_violations']) > 0:
        recent_violations = np.mean(history['range_violations'][-100:]) * 100
        summary_parts.append(f'\nRecent Violations: {recent_violations:.1f}%')
        
        # Trend analysis
        if len(history['range_violations']) > 200:
            early_viol = np.mean(history['range_violations'][:100]) * 100
            improvement = early_viol - recent_violations
            if improvement > 0:
                summary_parts.append(f'Improvement: ↓{improvement:.1f}%')
            else:
                summary_parts.append(f'Degradation: ↑{-improvement:.1f}%')
    
    ax9.text(0.1, 0.9, '\n'.join(summary_parts), 
            transform=ax9.transAxes, fontsize=12, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))
    
    # Add comprehensive title with validation statistics
    title_parts = [f'Training Progress - Iteration {iteration}']
    
    # Add scale constraint info if using RangePenaltyLoss
    if USE_DIRECTIONAL_LOSS:
        title_parts[0] += ' | Using RangePenaltyLoss (target range: [0.98, 1.02])'
    
    # Add latest validation metrics if available
    if 'val_dir_accuracy' in history and history['val_dir_accuracy']:
        latest_val_dir_acc = history['val_dir_accuracy'][-1]
        latest_val_corr = history['val_correlation'][-1] if 'val_correlation' in history and history['val_correlation'] else 0
        latest_val_mse = history['val_mse'][-1] if 'val_mse' in history and history['val_mse'] else 0
        title_parts.append(
            f'Latest Validation - Dir. Accuracy: {latest_val_dir_acc:.1%}, Correlation: {latest_val_corr:.3f}, MSE: {latest_val_mse:.4f}'
        )
    
    # Add range violation info if available
    if 'range_violations' in history and history['range_violations']:
        latest_violations = history['range_violations'][-1]
        title_parts.append(f'Range Violations: {latest_violations:.1%} of predictions outside [0.98, 1.02]')
    
    plt.suptitle('\n'.join(title_parts), fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust for suptitle
    
    # Save plot
    plot_path = PLOT_DIR / f'training_iter_{iteration}.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Also save as latest.png for easy viewing
    latest_path = PLOT_DIR / 'latest.png'
    import shutil
    shutil.copy(plot_path, latest_path)
    
    print(f"\n  📊 Plot saved: {plot_path}")
    print(f"  📊 Latest plot: {latest_path}")

def train_model(model, train_loader, val_loader, start_epoch=0, checkpoint_data=None):
    """Fine-tune the TimesFM model."""
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    
    # Choose loss function based on configuration
    if USE_DIRECTIONAL_LOSS:
        # Use RangePenaltyLoss for better scale handling
        criterion = RangePenaltyLoss(
            mse_weight=0.5,
            range_weight=0.3,
            directional_weight=0.2,
            expected_center=1.0,  # For normalized stock prices
            expected_range=0.02   # ±2% range for tighter constraints
        )
        print(f"📊 Using RangePenaltyLoss with scale constraints:")
        print(f"   - MSE weight: 0.50")
        print(f"   - Range weight: 0.30 (keeping predictions in [{0.98:.2f}, {1.02:.2f}])")
        print(f"   - Directional weight: 0.20")
    else:
        criterion = nn.MSELoss()
        print(f"📊 Using standard MSE loss")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * 0.01)
    
    # Initialize official TimesFM model for comparison
    print("📊 Initializing official TimesFM model for baseline comparison...")
    official_model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=TIMESFM_BACKEND,
            per_core_batch_size=32,
            horizon_len=128,  # Model's default horizon (will output 128 but we'll use first 64)
            num_layers=20,
            use_positional_embedding=True,
            context_len=512,  # Model's default (can handle up to 512, we'll use 448)
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id=MODEL_REPO
        ),
    )
    print(f"✅ Official TimesFM model loaded on {TIMESFM_BACKEND}")
    
    # Load checkpoint data if provided
    if checkpoint_data is not None:
        print(f"📥 Resuming from checkpoint (epoch {checkpoint_data['epoch'] + 1})")
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        best_val_loss = checkpoint_data.get('val_loss', float('inf'))
        history = checkpoint_data.get('history', {
            'train_loss': [], 
            'val_loss': [], 
            'learning_rates': [], 
            'iteration_losses': [],
            'directional_accuracy': [],
            'range_violations': [],
        })
        # Calculate total iterations from checkpoint
        total_iterations = len(history.get('iteration_losses', []))
        print(f"  ✅ Loaded optimizer, scheduler, and training history")
        print(f"  📊 Previous best val loss: {best_val_loss:.6f}")
        print(f"  🔢 Continuing from iteration: {total_iterations}")
    else:
        best_val_loss = float('inf')
        history = {
            'train_loss': [], 
            'val_loss': [], 
            'learning_rates': [], 
            'iteration_losses': [],
            'directional_accuracy': [],  # Track directional accuracy over time
            'range_violations': [],  # Track predictions outside expected range
        }
        total_iterations = 0
    
    actual_epochs = NUM_EPOCHS - start_epoch
    print(f"🏋️ Training for {actual_epochs} epochs (epochs {start_epoch+1} to {NUM_EPOCHS})")
    print(f"🔥 Model on device: {next(model.parameters()).device}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        # Training phase
        model.train()
        train_losses = []
        
        # Calculate when to plot based on configuration
        total_batches = len(train_loader)
        if PLOT_TIMES_PER_EPOCH is not None and PLOT_TIMES_PER_EPOCH > 0:
            # Plot N times per epoch, evenly spaced
            plot_interval = max(1, total_batches // PLOT_TIMES_PER_EPOCH)
            plot_at_iterations = set()
            for i in range(PLOT_TIMES_PER_EPOCH):
                iter_in_epoch = min(i * plot_interval, total_batches - 1)
                plot_at_iterations.add(epoch * total_batches + iter_in_epoch)
            print(f"  📊 Will create plots at {PLOT_TIMES_PER_EPOCH} points during epoch {epoch+1}")
        else:
            # Use fixed iteration interval
            plot_at_iterations = None
            plot_interval = PLOT_EVERY_ITERATIONS
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for batch_idx, batch in enumerate(train_pbar):
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            freq = batch['freq'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            with torch.cuda.amp.autocast():
                predictions = model(inputs, freq)
                loss = criterion(predictions, targets)
                
                # Debug: Check for prediction issues on first batch of each epoch
                if batch_idx == 0:
                    pred_check = predictions[0].detach().cpu().numpy()
                    print(f"\n  🔍 First batch debug (Epoch {epoch+1}):")
                    print(f"     Input shape: {inputs.shape}")
                    print(f"     Predictions shape: {predictions.shape}")
                    print(f"     Prediction at indices 10-15: {pred_check[10:15]}")
                    
                    # Check gradient flow
                    if predictions.requires_grad:
                        print(f"     Gradients enabled: ✓")
                    else:
                        print(f"     ⚠️ Gradients disabled!")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
            history['iteration_losses'].append(loss.item())
            
            # Track directional metrics if using directional loss
            if USE_DIRECTIONAL_LOSS and hasattr(criterion, 'last_accuracy'):
                history['directional_accuracy'].append(criterion.last_accuracy)
                # For RangePenaltyLoss, track range violations
                if hasattr(criterion, 'last_range_violations'):
                    if 'range_violations' not in history:
                        history['range_violations'] = []
                    history['range_violations'].append(criterion.last_range_violations)
            
            total_iterations += 1
            
            # Update progress bar
            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = np.mean(train_losses[-10:]) if len(train_losses) >= 10 else np.mean(train_losses)
                
                postfix_dict = {
                    'loss': f"{loss.item():.6f}",
                    'avg_loss': f"{avg_loss:.6f}",
                    'lr': f"{current_lr:.2e}"
                }
                
                # Add directional metrics if using directional loss
                if USE_DIRECTIONAL_LOSS and hasattr(criterion, 'last_accuracy'):
                    postfix_dict['dir_acc'] = f"{criterion.last_accuracy:.2%}"
                
                train_pbar.set_postfix(postfix_dict)
            
            # Detailed progress every 100 batches
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_loss_100 = np.mean(train_losses[-100:]) if len(train_losses) >= 100 else np.mean(train_losses)
                print(f"\n  📊 Batch {batch_idx}/{len(train_loader)} ({batch_idx/len(train_loader)*100:.1f}%)")
                print(f"      Current Loss: {loss.item():.6f}")
                print(f"      Avg Loss (last 100): {avg_loss_100:.6f}")
                
                if USE_DIRECTIONAL_LOSS and hasattr(criterion, 'last_accuracy'):
                    print(f"      Directional Accuracy: {criterion.last_accuracy:.2%}")
                    if hasattr(criterion, 'last_range_violations'):
                        print(f"      Range Violations: {criterion.last_range_violations:.2%}")
                        print(f"      (Predictions outside [0.98, 1.02] range)")
                
                print(f"      Learning Rate: {current_lr:.2e}")
                if torch.cuda.is_available():
                    print(f"      GPU Memory: {torch.cuda.memory_allocated() / (1024**3):.1f}GB")
            
            # Create plots based on configuration
            should_plot = False
            if plot_at_iterations is not None:
                # Plot based on times per epoch
                should_plot = total_iterations in plot_at_iterations
            else:
                # Plot based on fixed iteration interval
                should_plot = (total_iterations > 0 and total_iterations % plot_interval == 0)
            
            if should_plot:
                with torch.no_grad():
                    # Get a sample prediction for visualization
                    sample_predictions = predictions.detach()
                    sample_targets = targets.detach()
                    sample_input = inputs.detach()
                    sample_freq = freq.detach()
                    
                    # Debug: Print prediction values to check for zeros
                    print(f"\n  🔍 DEBUG - Prediction Analysis (Iteration {total_iterations}):")
                    pred_np = sample_predictions[0].cpu().numpy()  # First sample
                    target_np = sample_targets[0].cpu().numpy()
                    
                    print(f"     Prediction shape: {pred_np.shape}")
                    print(f"     First 20 values: {pred_np[:20]}")
                    print(f"     Last 20 values: {pred_np[-20:]}")
                    print(f"     Min value: {pred_np.min():.6f}, Max value: {pred_np.max():.6f}")
                    print(f"     Mean: {pred_np.mean():.6f}, Std: {pred_np.std():.6f}")
                    
                    # Check for zeros or near-zeros
                    near_zero_mask = np.abs(pred_np) < 1e-6
                    num_near_zeros = np.sum(near_zero_mask)
                    if num_near_zeros > 0:
                        print(f"     ⚠️ WARNING: {num_near_zeros} values are near zero!")
                        first_zero_idx = np.where(near_zero_mask)[0][0] if num_near_zeros > 0 else -1
                        print(f"     First near-zero at index: {first_zero_idx}")
                    
                    # Compare with target
                    print(f"     Target min: {target_np.min():.6f}, max: {target_np.max():.6f}")
                    print(f"     Target mean: {target_np.mean():.6f}, std: {target_np.std():.6f}")
                
                create_training_plots(
                    history,
                    predictions_sample=sample_predictions,
                    targets_sample=sample_targets,
                    iteration=total_iterations,
                    model=model,
                    official_model=official_model,
                    sample_context=sample_input,
                    sample_freq=sample_freq,
                    dataset=train_loader.dataset
                )
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_directional_accuracies = []
        val_correlations = []
        val_mse_components = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
            for batch in val_pbar:
                inputs = batch['input'].to(DEVICE)
                targets = batch['target'].to(DEVICE)
                freq = batch['freq'].to(DEVICE)
                
                with torch.cuda.amp.autocast():
                    predictions = model(inputs, freq)
                    loss = criterion(predictions, targets)
                
                val_losses.append(loss.item())
                
                # Calculate validation metrics
                if USE_DIRECTIONAL_LOSS and hasattr(criterion, 'forward'):
                    # Calculate directional accuracy for validation
                    pred_direction = (predictions[:, 1:] > predictions[:, :-1]).float()
                    true_direction = (targets[:, 1:] > targets[:, :-1]).float()
                    dir_accuracy = (pred_direction == true_direction).float().mean().item()
                    val_directional_accuracies.append(dir_accuracy)
                    
                    # Calculate correlation
                    batch_correlations = []
                    for i in range(predictions.shape[0]):
                        if predictions[i].std() > 1e-6 and targets[i].std() > 1e-6:
                            corr = np.corrcoef(predictions[i].cpu().numpy(), targets[i].cpu().numpy())[0, 1]
                            if not np.isnan(corr):
                                batch_correlations.append(corr)
                    if batch_correlations:
                        val_correlations.append(np.mean(batch_correlations))
                    
                    # MSE component
                    mse = F.mse_loss(predictions, targets).item()
                    val_mse_components.append(mse)
                
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.6f}",
                    'dir_acc': f"{dir_accuracy:.2%}" if val_directional_accuracies else "N/A"
                })
        
        avg_val_loss = np.mean(val_losses)
        avg_val_dir_accuracy = np.mean(val_directional_accuracies) if val_directional_accuracies else 0.5
        avg_val_correlation = np.mean(val_correlations) if val_correlations else 0.0
        avg_val_mse = np.mean(val_mse_components) if val_mse_components else avg_val_loss
        
        # Create validation plots if enabled
        if CREATE_VAL_PLOTS and (epoch + 1) % VAL_PLOT_FREQUENCY == 0:
            print(f"\n📊 Creating validation plots for epoch {epoch + 1}...")
            try:
                create_validation_plots(model, val_loader, epoch + 1, num_plots=VAL_PLOTS_PER_EPOCH, official_model=official_model)
            except Exception as e:
                print(f"   ⚠️ Failed to create validation plots: {e}")
                import traceback
                traceback.print_exc()
        
        # Update learning rate
        scheduler.step()
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Add validation metrics to history
        if 'val_dir_accuracy' not in history:
            history['val_dir_accuracy'] = []
            history['val_correlation'] = []
            history['val_mse'] = []
        
        history['val_dir_accuracy'].append(avg_val_dir_accuracy)
        history['val_correlation'].append(avg_val_correlation)
        history['val_mse'].append(avg_val_mse)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        
        # If using directional loss, show validation metrics
        if USE_DIRECTIONAL_LOSS:
            print(f"  Val Dir. Accuracy: {avg_val_dir_accuracy:.2%}")
            print(f"  Val Correlation:   {avg_val_correlation:.3f}")
            print(f"  Val MSE:          {avg_val_mse:.6f}")
        
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = CHECKPOINT_DIR / "best_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'history': history
            }, best_model_path)
            print(f"  ✅ New best model saved (val_loss: {avg_val_loss:.6f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'history': history
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
        
        # Early stopping
        if epoch > 50 and avg_val_loss > best_val_loss * 1.1:
            early_stop_epochs = 50
            if all(history['val_loss'][-(i+1)] > best_val_loss for i in range(min(early_stop_epochs, len(history['val_loss'])))):
                print(f"  🛑 Early stopping: No improvement for {early_stop_epochs} epochs")
                break
        
        print()
    
    print("✅ Fine-tuning complete!")
    return model, history


def main():
    """Main fine-tuning pipeline."""
    # Declare globals at the start
    global NUM_EPOCHS, BATCH_SIZE, USE_CACHED_DATA, VISUALIZE_PREPROCESSING, PLOT_TIMES_PER_EPOCH, PLOT_EVERY_ITERATIONS, CREATE_VAL_PLOTS, VAL_PLOTS_PER_EPOCH, VAL_PLOT_FREQUENCY
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fine-tune TimesFM on stock data')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of data, ignoring cache')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable cache usage for this run')
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip preprocessing visualization')
    parser.add_argument('--keep-plots', action='store_true',
                        help='Keep existing plots instead of clearing them')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--plots-per-epoch', type=int, default=None,
                        help='Number of plots to create per epoch (overrides --plot-every-iter)')
    parser.add_argument('--plot-every-iter', type=int, default=PLOT_EVERY_ITERATIONS,
                        help=f'Create plots every N iterations (default: {PLOT_EVERY_ITERATIONS})')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., /workspace/stonkz/finetune_checkpoints/best_model.pth)')
    parser.add_argument('--val-plots', type=int, default=VAL_PLOTS_PER_EPOCH,
                        help=f'Number of validation plots to create per epoch (default: {VAL_PLOTS_PER_EPOCH})')
    parser.add_argument('--val-plot-freq', type=int, default=VAL_PLOT_FREQUENCY,
                        help=f'Create validation plots every N epochs (default: {VAL_PLOT_FREQUENCY})')
    parser.add_argument('--no-val-plots', action='store_true',
                        help='Disable validation plot creation')
    args = parser.parse_args()
    
    # Update global config based on args
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    PLOT_TIMES_PER_EPOCH = args.plots_per_epoch
    PLOT_EVERY_ITERATIONS = args.plot_every_iter
    VAL_PLOTS_PER_EPOCH = args.val_plots
    VAL_PLOT_FREQUENCY = args.val_plot_freq
    if args.no_cache:
        USE_CACHED_DATA = False
    if args.skip_visualization:
        VISUALIZE_PREPROCESSING = False
    if args.no_val_plots:
        CREATE_VAL_PLOTS = False
    
    print("🚀 PyTorch TimesFM Fine-tuning Pipeline")
    print("=" * 50)
    print(f"🔥 Using device: {DEVICE}")
    print(f"📊 Comparison baseline: Official Google TimesFM model")
    print("=" * 50)
    
    # Clear all plot directories at startup (unless --keep-plots is specified)
    if not args.keep_plots:
        print("\n🧹 Clearing plot directories...")
        plot_dirs = [
            SCRIPT_DIR / "finetune_plots",
            SCRIPT_DIR / "plots", 
            SCRIPT_DIR / "stock_plots"
        ]
        
        for plot_dir in plot_dirs:
            if plot_dir.exists():
                # Count files before deletion (including test files)
                plot_files = list(plot_dir.glob("*.png")) + list(plot_dir.glob("*.jpg"))
                test_files = [f for f in plot_files if f.name.startswith("test_")]
                
                if plot_files:
                    if test_files:
                        print(f"   Removing {len(plot_files)} files from {plot_dir.name}/ (including {len(test_files)} test files)")
                    else:
                        print(f"   Removing {len(plot_files)} files from {plot_dir.name}/")
                    
                    for file in plot_files:
                        try:
                            file.unlink()
                        except Exception as e:
                            print(f"   ⚠️ Failed to delete {file.name}: {e}")
                else:
                    print(f"   ✓ {plot_dir.name}/ is already empty")
            else:
                print(f"   ✓ {plot_dir.name}/ doesn't exist")
        
        print("   ✅ Plot directories cleaned")
    else:
        print("\n🗂️  Keeping existing plots (--keep-plots flag set)")
    
    # Display configuration to confirm it matches good_timefm_stock_working.py
    print("\n📈 Prediction Configuration (matching good_timefm_stock_working.py):")
    print(f"   Context window: {CONTEXT_LENGTH} minutes (~{CONTEXT_LENGTH/60:.1f} hours)")
    print(f"   Prediction horizon: {HORIZON_LENGTH} minutes (~{HORIZON_LENGTH/60:.1f} hours)")
    print(f"   Input patch length: {INPUT_PATCH_LEN} minutes")
    print(f"   Number of patches: {CONTEXT_LENGTH // INPUT_PATCH_LEN}")
    print(f"   Using normalized prices: True (like working example)")
    print("=" * 50)
    
    # Display cache settings
    print("\n💾 Cache Configuration:")
    print(f"   Cache enabled: {USE_CACHED_DATA and not args.no_cache}")
    print(f"   Force reprocess: {args.force_reprocess}")
    print(f"   Cache version: {CACHE_VERSION}")
    print(f"   Cache directory: {CACHE_DIR}")
    print("=" * 50)
    
    # Display plotting configuration
    print("\n📊 Plotting Configuration:")
    if PLOT_TIMES_PER_EPOCH is not None:
        print(f"   Training plots per epoch: {PLOT_TIMES_PER_EPOCH}")
        print(f"   Mode: Evenly spaced throughout each epoch")
    else:
        print(f"   Training plot every: {PLOT_EVERY_ITERATIONS} iterations")
        print(f"   Mode: Fixed iteration interval")
    
    print(f"\n📊 Validation Plotting:")
    if CREATE_VAL_PLOTS:
        print(f"   Validation plots enabled: {VAL_PLOTS_PER_EPOCH} plots every {VAL_PLOT_FREQUENCY} epochs")
        print(f"   Plot directory: {PLOT_DIR / 'validation'}")
    else:
        print(f"   Validation plots disabled")
    print("=" * 50)

    # Load data first
    data_dict = load_and_prepare_data(force_reprocess=args.force_reprocess)
    
    # Create DataLoaders with the correct batch size (from command line args)
    print(f"\n📊 Creating DataLoaders with batch size: {BATCH_SIZE}")
    data_dict['train_loader'] = DataLoader(
        data_dict['train_dataset'], 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    data_dict['val_loader'] = DataLoader(
        data_dict['val_dataset'], 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    data_dict['test_loader'] = DataLoader(
        data_dict['test_dataset'], 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    # Visualize preprocessing if enabled
    if VISUALIZE_PREPROCESSING:
        visualize_monte_carlo_sequences(data_dict, num_samples=MONTE_CARLO_SAMPLES)
        print("\n✅ Preprocessing visualization complete!")
        
    # Initialize model after visualization
    print("\n🤖 Initializing TimesFM model...")
    model = TimesFMModel(
        context_len=CONTEXT_LENGTH,
        horizon_len=HORIZON_LENGTH,
        input_patch_len=INPUT_PATCH_LEN
    )
    
    # Load pre-trained weights
    print("\n📥 Loading pre-trained TimesFM weights...")
    model_path = snapshot_download(repo_id=MODEL_REPO)
    checkpoint_path = os.path.join(model_path, "torch_model.ckpt")
    
    success = model.load_pretrained_weights(checkpoint_path)
    if not success:
        print("⚠️ Warning: Could not load all pre-trained weights. Proceeding with partial initialization.")
    
    # Move model to device
    model = model.to(DEVICE)
    print(f"\n✅ Model ready with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"🔥 Model on device: {next(model.parameters()).device}")
    
    # Load checkpoint if specified or if USE_LATEST_CHECKPOINT is True
    start_epoch = 0
    checkpoint_data = None
    checkpoint_path = None
    
    # First check if we should use the latest checkpoint
    if USE_LATEST_CHECKPOINT and not args.resume_from:
        # Find the latest checkpoint
        checkpoints = list(CHECKPOINT_DIR.glob("*.pth"))
        if checkpoints:
            # Sort by modification time to get the latest
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
            checkpoint_path = latest_checkpoint
            print(f"\n🔍 Found latest checkpoint: {checkpoint_path.name}")
        else:
            print(f"\n⚠️ USE_LATEST_CHECKPOINT is True but no checkpoints found in {CHECKPOINT_DIR}")
    
    # Override with command line argument if provided
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        print(f"\n📝 Using checkpoint from command line: {checkpoint_path}")
    
    # Load the checkpoint if we have one
    if checkpoint_path and checkpoint_path.exists():
        print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
        checkpoint_data = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        start_epoch = checkpoint_data['epoch'] + 1  # Start from next epoch
        
        print(f"  ✅ Loaded model state from epoch {checkpoint_data['epoch']}")
        print(f"  📊 Checkpoint val loss: {checkpoint_data['val_loss']:.6f}")
        print(f"  📊 Checkpoint train loss: {checkpoint_data['train_loss']:.6f}")
        
        # Update NUM_EPOCHS if needed to ensure we don't train less than requested
        if start_epoch >= NUM_EPOCHS:
            print(f"  ⚠️ Checkpoint epoch ({start_epoch}) >= requested epochs ({NUM_EPOCHS})")
            print(f"  📝 Updating NUM_EPOCHS to {start_epoch + args.epochs}")
            NUM_EPOCHS = start_epoch + args.epochs
    elif checkpoint_path:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Starting training from scratch...")
    
    # Fine-tune model
    print(f"\n🏋️ Fine-tuning TimesFM...")
    trained_model, history = train_model(
        model, 
        data_dict['train_loader'], 
        data_dict['val_loader'],
        start_epoch=start_epoch,
        checkpoint_data=checkpoint_data
    )
    
    print(f"\n✅ PyTorch TimesFM fine-tuning completed!")
    print(f"📁 Best model saved to: {CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"📁 Checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"📁 Training plots saved to: {PLOT_DIR}")
    
    # Final GPU status
    if torch.cuda.is_available():
        print(f"\n🔥 Final GPU Status:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / (1024**3):.1f}GB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / (1024**3):.1f}GB")


if __name__ == "__main__":
    main()
