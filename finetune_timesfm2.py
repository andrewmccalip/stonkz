#!/usr/bin/env python3
"""
TimesFM 2.0 Finetuning Script - ES Futures Dataset
Uses the organized sequence dataset with proper time-based splitting to avoid contamination.

=== TENSORBOARD MONITORING ===
This script includes comprehensive TensorBoard logging for real-time training monitoring.

USAGE:
1. Start training: python finetune_timesfm2.py
2. In another terminal, start TensorBoard: tensorboard --logdir=tensorboard_logs
3. Open browser to: http://localhost:6006
4. View real-time training progress, metrics, and plots

TENSORBOARD FEATURES:
- Real-time loss curves (train/validation)
- Performance metrics (accuracy, correlation, MSE, MAE)
- Learning rate scheduling
- Gradient norms and training efficiency
- Comprehensive training plots (automatically uploaded)
- Model architecture graph
- Hyperparameter tracking

TABS IN TENSORBOARD:
- SCALARS: All numeric metrics over time
- IMAGES: Training progress plots and validation samples
- GRAPHS: Model architecture visualization
- HISTOGRAMS: Weight and gradient distributions
- HPARAMS: Hyperparameter comparison across runs

TIPS:
- Refresh browser to see latest updates
- Use the smoothing slider to reduce noise in plots
- Compare multiple runs by training with different parameters
- Download plots directly from TensorBoard interface
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict, Any
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter

# Add TimesFM to path
SCRIPT_DIR = Path(__file__).parent
TIMESFM_DIR = SCRIPT_DIR / "FlaMinGo-timesfm-clean" / "src"
sys.path.append(str(TIMESFM_DIR))

# TimesFM imports
from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner

# ==============================================================================
# Configuration
# ==============================================================================

# Dataset paths
SEQUENCES_DIR = SCRIPT_DIR / "datasets" / "sequences"
CACHE_DIR = SCRIPT_DIR / "timesfm_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Plotting directory
FINETUNE_PLOTS_DIR = SCRIPT_DIR / "finetune_plots"
FINETUNE_PLOTS_DIR.mkdir(exist_ok=True)

# TensorBoard Configuration
TENSORBOARD_LOG_DIR = SCRIPT_DIR / "tensorboard_logs"
TENSORBOARD_LOG_DIR.mkdir(exist_ok=True)

# Model configuration - Using original TimesFM 2.0 500m (NOT Flamingo)
MODEL_CONFIG = {
    "repo_id": "google/timesfm-2.0-500m-pytorch",  # Original Google TimesFM 2.0 500m model
    "context_len": 512,      # Match our sequence length (will be padded/truncated as needed)
    "horizon_len": 128,      # Model's default horizon (will match our 96 + padding)
    "num_layers": 50,        # TimesFM 2.0 500m
    "per_core_batch_size": 16,
    "use_positional_embedding": False,
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "num_epochs": 5,         # Reduced for testing
    "learning_rate": 1e-5,   # Lower LR for finetuning
    "weight_decay": 0.01,
    "freq_type": 0,          # High frequency (minute data)
    "use_quantile_loss": True,
    "log_every_n_steps": 10,
    "val_check_interval": 0.5,  # Validate twice per epoch
    "use_wandb": False,      # Disabled for testing
    "wandb_project": "timesfm2-es-futures",
}

# Sample configuration (for testing)
SAMPLE_CONFIG = {
    "use_sample": True,      # Use random sample instead of full dataset
    "total_sequences": 1000, # Total sequences to use
    "train_ratio": 0.7,      # 70% for training
    "val_ratio": 0.2,        # 20% for validation  
    "test_ratio": 0.1,       # 10% for testing
    "random_seed": 42,       # For reproducible sampling
}

# Data splitting (time-based to avoid contamination)
SPLIT_CONFIG = {
    "train_end_date": "2020-12-31",    # Train: 2010-2020
    "val_end_date": "2022-12-31",      # Val: 2021-2022  
    "test_start_date": "2023-01-01",   # Test: 2023+
    "context_minutes": 416,            # Historical context
    "prediction_minutes": 96,          # Future prediction
}

# ==============================================================================
# Dataset Class
# ==============================================================================

class ESFuturesSequenceDataset(Dataset):
    """Dataset for ES futures sequences compatible with TimesFM 2.0"""
    
    def __init__(self, 
                 sequence_files: List[Path],
                 context_length: int = 416,
                 horizon_length: int = 96,
                 freq_type: int = 0,
                 normalize: bool = True,
                 verbose: bool = False):
        """
        Initialize dataset from sequence files.
        
        Args:
            sequence_files: List of sequence CSV file paths
            context_length: Number of past timesteps for context
            horizon_length: Number of future timesteps to predict
            freq_type: TimesFM frequency type (0=high, 1=medium, 2=low)
            normalize: Whether to normalize sequences to start at 1.0
            verbose: Print detailed information
        """
        
        self.sequence_files = sequence_files
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.freq_type = freq_type
        self.normalize = normalize
        self.verbose = verbose
        
        # Load and prepare all sequences
        self.sequences = self._load_sequences()
        
        if self.verbose:
            print(f"📊 Dataset initialized:")
            print(f"   Sequence files: {len(self.sequence_files)}")
            print(f"   Total samples: {len(self.sequences)}")
            print(f"   Context length: {self.context_length}")
            print(f"   Horizon length: {self.horizon_length}")
            print(f"   Frequency type: {self.freq_type}")
    
    def _load_sequences(self) -> List[Dict[str, Any]]:
        """Load all sequence files and create training samples"""
        
        sequences = []
        
        for seq_file in tqdm(self.sequence_files, desc="Loading sequences", disable=not self.verbose):
            try:
                # Load sequence data
                df = pd.read_csv(seq_file)
                
                if len(df) < self.context_length + self.horizon_length:
                    continue  # Skip sequences that are too short
                
                # Extract close prices (normalized)
                close_prices = df['close'].values
                
                # Create sliding window samples from this sequence
                max_start = len(close_prices) - (self.context_length + self.horizon_length)
                
                for start_idx in range(0, max_start + 1, 60):  # Stride by 60 minutes
                    context_end = start_idx + self.context_length
                    horizon_end = context_end + self.horizon_length
                    
                    context_data = close_prices[start_idx:context_end]
                    horizon_data = close_prices[context_end:horizon_end]
                    
                    # Additional normalization if requested
                    if self.normalize and len(context_data) > 0:
                        base_price = context_data[0]
                        if base_price > 0:
                            context_data = context_data / base_price
                            horizon_data = horizon_data / base_price
                    
                    sequences.append({
                        'context': context_data,
                        'horizon': horizon_data,
                        'file': seq_file.name,
                        'start_idx': start_idx
                    })
                    
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Error loading {seq_file.name}: {e}")
                continue
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Tuple of (context, padding, freq, horizon) tensors
        """
        
        sequence = self.sequences[index]
        
        # Convert to tensors
        context = torch.tensor(sequence['context'], dtype=torch.float32)
        horizon = torch.tensor(sequence['horizon'], dtype=torch.float32)
        
        # Pad horizon to match model's expected horizon length (128)
        model_horizon_len = 128
        if len(horizon) < model_horizon_len:
            # Pad with the last value
            padding_needed = model_horizon_len - len(horizon)
            last_value = horizon[-1] if len(horizon) > 0 else 1.0
            horizon_padding = torch.full((padding_needed,), last_value, dtype=torch.float32)
            horizon = torch.cat([horizon, horizon_padding])
        elif len(horizon) > model_horizon_len:
            # Truncate to model horizon length
            horizon = horizon[:model_horizon_len]
        
        # Create context padding (zeros for TimesFM)
        padding = torch.zeros_like(context)
        
        # Frequency type
        freq = torch.tensor([self.freq_type], dtype=torch.long)
        
        return context, padding, freq, horizon

# ==============================================================================
# Data Loading and Splitting
# ==============================================================================

def get_sequence_files() -> List[Path]:
    """Get all sequence files sorted by date"""
    
    if not SEQUENCES_DIR.exists():
        raise FileNotFoundError(f"Sequences directory not found: {SEQUENCES_DIR}")
    
    sequence_files = list(SEQUENCES_DIR.glob("sequence_*.csv"))
    
    if len(sequence_files) == 0:
        raise FileNotFoundError(f"No sequence files found in {SEQUENCES_DIR}")
    
    # Sort by filename (which contains date)
    sequence_files.sort()
    
    print(f"📁 Found {len(sequence_files)} sequence files")
    print(f"   Date range: {sequence_files[0].name} to {sequence_files[-1].name}")
    
    return sequence_files

def parse_sequence_date(filename: str) -> datetime:
    """Parse date from sequence filename: sequence_YYYY-MM-DD_HHMMPT.csv"""
    
    try:
        # Extract date part: sequence_2010-06-07_0000PT.csv -> 2010-06-07
        date_part = filename.split('_')[1]  # Get YYYY-MM-DD part
        return datetime.strptime(date_part, '%Y-%m-%d')
    except (IndexError, ValueError) as e:
        raise ValueError(f"Cannot parse date from filename {filename}: {e}")

def split_sequences_by_time(sequence_files: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split sequence files by time to avoid contamination.
    If using sample mode, randomly sample from all files instead.
    
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    
    if SAMPLE_CONFIG["use_sample"]:
        return split_sequences_random_sample(sequence_files)
    
    # Parse split dates
    train_end = datetime.strptime(SPLIT_CONFIG["train_end_date"], '%Y-%m-%d')
    val_end = datetime.strptime(SPLIT_CONFIG["val_end_date"], '%Y-%m-%d')
    test_start = datetime.strptime(SPLIT_CONFIG["test_start_date"], '%Y-%m-%d')
    
    train_files = []
    val_files = []
    test_files = []
    
    for seq_file in sequence_files:
        try:
            file_date = parse_sequence_date(seq_file.name)
            
            if file_date <= train_end:
                train_files.append(seq_file)
            elif file_date <= val_end:
                val_files.append(seq_file)
            elif file_date >= test_start:
                test_files.append(seq_file)
            # Files between val_end and test_start are excluded (buffer period)
                
        except ValueError as e:
            print(f"⚠️ Skipping file with invalid date: {seq_file.name}")
            continue
    
    print(f"📊 Time-based split:")
    print(f"   Train: {len(train_files)} files (up to {SPLIT_CONFIG['train_end_date']})")
    print(f"   Val:   {len(val_files)} files ({train_end.strftime('%Y-%m-%d')} to {SPLIT_CONFIG['val_end_date']})")
    print(f"   Test:  {len(test_files)} files (from {SPLIT_CONFIG['test_start_date']})")
    
    return train_files, val_files, test_files

def split_sequences_random_sample(sequence_files: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Randomly sample sequences for testing purposes.
    
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    
    import random
    
    # Set random seed for reproducibility
    random.seed(SAMPLE_CONFIG["random_seed"])
    np.random.seed(SAMPLE_CONFIG["random_seed"])
    
    # Randomly sample files
    total_sequences = min(SAMPLE_CONFIG["total_sequences"], len(sequence_files))
    sampled_files = random.sample(sequence_files, total_sequences)
    
    # Split sampled files
    n_train = int(total_sequences * SAMPLE_CONFIG["train_ratio"])
    n_val = int(total_sequences * SAMPLE_CONFIG["val_ratio"])
    n_test = total_sequences - n_train - n_val
    
    train_files = sampled_files[:n_train]
    val_files = sampled_files[n_train:n_train + n_val]
    test_files = sampled_files[n_train + n_val:]
    
    print(f"📊 Random sample split:")
    print(f"   Total available: {len(sequence_files)} files")
    print(f"   Sampled: {total_sequences} files")
    print(f"   Train: {len(train_files)} files ({SAMPLE_CONFIG['train_ratio']*100:.0f}%)")
    print(f"   Val:   {len(val_files)} files ({SAMPLE_CONFIG['val_ratio']*100:.0f}%)")
    print(f"   Test:  {len(test_files)} files ({SAMPLE_CONFIG['test_ratio']*100:.0f}%)")
    
    return train_files, val_files, test_files

def create_datasets(train_files: List[Path], 
                   val_files: List[Path], 
                   test_files: List[Path]) -> Tuple[Dataset, Dataset, Dataset]:
    """Create train, validation, and test datasets"""
    
    print("🔄 Creating datasets...")
    
    train_dataset = ESFuturesSequenceDataset(
        sequence_files=train_files,
        context_length=SPLIT_CONFIG["context_minutes"],
        horizon_length=SPLIT_CONFIG["prediction_minutes"],
        freq_type=TRAINING_CONFIG["freq_type"],
        normalize=True,
        verbose=True
    )
    
    val_dataset = ESFuturesSequenceDataset(
        sequence_files=val_files,
        context_length=SPLIT_CONFIG["context_minutes"],
        horizon_length=SPLIT_CONFIG["prediction_minutes"],
        freq_type=TRAINING_CONFIG["freq_type"],
        normalize=True,
        verbose=True
    )
    
    test_dataset = ESFuturesSequenceDataset(
        sequence_files=test_files,
        context_length=SPLIT_CONFIG["context_minutes"],
        horizon_length=SPLIT_CONFIG["prediction_minutes"],
        freq_type=TRAINING_CONFIG["freq_type"],
        normalize=True,
        verbose=True
    )
    
    return train_dataset, val_dataset, test_dataset

# ==============================================================================
# Model Setup
# ==============================================================================

def create_timesfm_model(load_weights: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
    """Create TimesFM 2.0 model"""
    
    print("🤖 Creating TimesFM 2.0 model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    # Create model configuration
    hparams = TimesFmHparams(
        backend=device,
        per_core_batch_size=MODEL_CONFIG["per_core_batch_size"],
        horizon_len=MODEL_CONFIG["horizon_len"],
        num_layers=MODEL_CONFIG["num_layers"],
        use_positional_embedding=MODEL_CONFIG["use_positional_embedding"],
        context_len=MODEL_CONFIG["context_len"],
    )
    
    # Initialize TimesFM
    tfm = TimesFm(
        hparams=hparams,
        checkpoint=TimesFmCheckpoint(huggingface_repo_id=MODEL_CONFIG["repo_id"])
    )
    
    # Create the model
    model = PatchedTimeSeriesDecoder(tfm._model_config)
    
    if load_weights:
        print("   Loading pretrained weights...")
        from huggingface_hub import snapshot_download
        checkpoint_path = Path(snapshot_download(MODEL_CONFIG["repo_id"])) / "torch_model.ckpt"
        
        if checkpoint_path.exists():
            loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(loaded_checkpoint)
            print("   ✅ Pretrained weights loaded successfully")
        else:
            print("   ⚠️ Checkpoint not found, using random initialization")
    
    model_info = {
        'hparams': hparams,
        'config': tfm._model_config,
        'device': device
    }
    
    return model, model_info

# ==============================================================================
# Custom Verbose Finetuner
# ==============================================================================

class VerboseTimesFMFinetuner(TimesFMFinetuner):
    """Enhanced TimesFM Finetuner with verbose logging and plotting"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': [],
            'batch_times': []
        }
        self.start_time = None
        
        # Additional tracking for comprehensive analysis
        self.learning_rates = []
        self.gradient_norms = []
        self.loss_components = {}
        self.performance_metrics = {
            'directional_accuracy': [],
            'correlation': [],
            'mse': [],
            'mae': []
        }
        
        # TensorBoard logging
        self.tensorboard_writer = None
        self.global_step = 0
        
    def _train_epoch_verbose(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        """Train for one epoch with verbose logging"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        epoch_start_time = time.time()
        
        print(f"\n🎯 Epoch {epoch + 1}/{self.config.num_epochs}")
        print(f"   Batches: {num_batches}, Batch size: {self.config.batch_size}")
        
        batch_losses = []
        self._epoch_grad_norms = []
        
        with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False) as pbar:
            for batch_idx, batch in enumerate(pbar):
                batch_start_time = time.time()
                
                try:
                    loss, _ = self._process_batch(batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Track gradient norms
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    self._epoch_grad_norms.append(total_norm)
                    
                    optimizer.step()
                    
                    batch_loss = loss.item()
                    total_loss += batch_loss
                    batch_losses.append(batch_loss)
                    
                    batch_time = time.time() - batch_start_time
                    self.training_history['batch_times'].append(batch_time)
                    
                    # TensorBoard logging for batch-level metrics
                    if self.tensorboard_writer is not None:
                        self.tensorboard_writer.add_scalar('Batch/Loss', batch_loss, self.global_step)
                        self.tensorboard_writer.add_scalar('Batch/Time', batch_time, self.global_step)
                        if hasattr(self, '_epoch_grad_norms') and self._epoch_grad_norms:
                            self.tensorboard_writer.add_scalar('Batch/GradientNorm', self._epoch_grad_norms[-1], self.global_step)
                    
                    self.global_step += 1
                    
                    # Update progress bar
                    avg_loss_so_far = total_loss / (batch_idx + 1)
                    pbar.set_postfix({
                        'Loss': f'{batch_loss:.6f}',
                        'Avg': f'{avg_loss_so_far:.6f}',
                        'Time': f'{batch_time:.2f}s'
                    })
                    
                    # Log every N steps
                    if (batch_idx + 1) % self.config.log_every_n_steps == 0:
                        print(f"      Batch {batch_idx + 1}/{num_batches}: Loss={batch_loss:.6f}, Time={batch_time:.2f}s")
                        
                except Exception as e:
                    print(f"      ❌ Error in batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        self.training_history['epoch_times'].append(epoch_time)
        
        # Track average gradient norm for this epoch
        avg_grad_norm = None
        if hasattr(self, '_epoch_grad_norms'):
            avg_grad_norm = np.mean(self._epoch_grad_norms)
            self.gradient_norms.append(avg_grad_norm)
            delattr(self, '_epoch_grad_norms')
        
        # TensorBoard logging for epoch-level metrics
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar('Epoch/TrainLoss', avg_loss, epoch)
            self.tensorboard_writer.add_scalar('Epoch/TrainTime', epoch_time, epoch)
            self.tensorboard_writer.add_scalar('Epoch/TrainLossStd', np.std(batch_losses), epoch)
            if avg_grad_norm is not None:
                self.tensorboard_writer.add_scalar('Epoch/GradientNorm', avg_grad_norm, epoch)
        
        print(f"   ✅ Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"   📊 Train Loss: {avg_loss:.6f} (std: {np.std(batch_losses):.6f})")
        
        return avg_loss
    
    def _validate_verbose(self, val_loader: DataLoader, epoch: int) -> float:
        """Perform validation with verbose logging"""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        val_start_time = time.time()
        
        print(f"   🔍 Validating...")
        
        val_losses = []
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", leave=False) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    try:
                        loss, predictions = self._process_batch(batch)
                        batch_loss = loss.item()
                        total_loss += batch_loss
                        val_losses.append(batch_loss)
                        
                        # Collect predictions and targets for metrics calculation
                        if predictions is not None and 'target' in batch:
                            val_predictions.append(predictions.cpu())
                            val_targets.append(batch['target'].cpu())
                        
                        pbar.set_postfix({'Val Loss': f'{batch_loss:.6f}'})
                        
                    except Exception as e:
                        print(f"      ❌ Error in validation batch {batch_idx}: {e}")
                        continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        val_time = time.time() - val_start_time
        
        # Calculate performance metrics if we have predictions and targets
        if val_predictions and val_targets:
            try:
                # Concatenate all predictions and targets
                all_predictions = torch.cat(val_predictions, dim=0)
                all_targets = torch.cat(val_targets, dim=0)
                
                # Calculate metrics
                mse = torch.mean((all_predictions - all_targets) ** 2).item()
                mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
                
                # Directional accuracy
                pred_direction = (all_predictions[:, 1:] > all_predictions[:, :-1]).float()
                true_direction = (all_targets[:, 1:] > all_targets[:, :-1]).float()
                dir_accuracy = (pred_direction == true_direction).float().mean().item()
                
                # Correlation
                correlations = []
                for i in range(all_predictions.shape[0]):
                    pred_i = all_predictions[i].numpy()
                    target_i = all_targets[i].numpy()
                    if np.std(pred_i) > 1e-6 and np.std(target_i) > 1e-6:
                        corr = np.corrcoef(pred_i, target_i)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                
                avg_correlation = np.mean(correlations) if correlations else 0.0
                
                # Store metrics
                self.performance_metrics['mse'].append(mse)
                self.performance_metrics['mae'].append(mae)
                self.performance_metrics['directional_accuracy'].append(dir_accuracy)
                self.performance_metrics['correlation'].append(avg_correlation)
                
                print(f"   📊 Val Metrics: MSE={mse:.6f}, MAE={mae:.6f}, Dir.Acc={dir_accuracy:.1%}, Corr={avg_correlation:.3f}")
                
                # TensorBoard logging for validation metrics
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar('Validation/MSE', mse, epoch)
                    self.tensorboard_writer.add_scalar('Validation/MAE', mae, epoch)
                    self.tensorboard_writer.add_scalar('Validation/DirectionalAccuracy', dir_accuracy, epoch)
                    self.tensorboard_writer.add_scalar('Validation/Correlation', avg_correlation, epoch)
                
            except Exception as e:
                print(f"   ⚠️ Error calculating performance metrics: {e}")
        
        # TensorBoard logging for validation loss
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar('Epoch/ValLoss', avg_loss, epoch)
            self.tensorboard_writer.add_scalar('Epoch/ValTime', val_time, epoch)
            self.tensorboard_writer.add_scalar('Epoch/ValLossStd', np.std(val_losses), epoch)
        
        print(f"   📊 Val Loss: {avg_loss:.6f} (std: {np.std(val_losses):.6f}) in {val_time:.2f}s")
        
        return avg_loss
    
    def plot_training_progress(self, save_path: Optional[Path] = None):
        """Create comprehensive training progress plots with detailed analysis for stock predictors"""
        
        if len(self.training_history['train_loss']) == 0:
            print("   ⚠️ No training history available for plotting")
            return
        
        # Create 3x3 subplot layout for comprehensive analysis
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Plot 1: Training and Validation Loss
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss', linewidth=2, alpha=0.8)
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Add best loss annotations
        if self.training_history['val_loss']:
            best_val_loss = min(self.training_history['val_loss'])
            best_epoch = self.training_history['val_loss'].index(best_val_loss) + 1
            ax1.annotate(f'Best: {best_val_loss:.6f}\nEpoch {best_epoch}', 
                        xy=(best_epoch, best_val_loss), xytext=(0.7, 0.8),
                        textcoords='axes fraction', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
        
        # Plot 2: Loss Difference (Overfitting Monitor)
        ax2 = axes[0, 1]
        if len(self.training_history['train_loss']) > 1:
            loss_diff = [v - t for t, v in zip(self.training_history['train_loss'], self.training_history['val_loss'])]
            ax2.plot(epochs, loss_diff, 'g-', linewidth=2)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No Overfitting')
            ax2.fill_between(epochs, loss_diff, 0, where=[d > 0 for d in loss_diff], 
                            color='red', alpha=0.3, label='Overfitting')
            ax2.fill_between(epochs, loss_diff, 0, where=[d <= 0 for d in loss_diff], 
                            color='green', alpha=0.3, label='Underfitting')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Val Loss - Train Loss')
            ax2.set_title('Overfitting Monitor')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate Schedule (if available)
        ax3 = axes[0, 2]
        if hasattr(self, 'learning_rates') and self.learning_rates:
            ax3.plot(epochs, self.learning_rates, 'purple', linewidth=2, marker='o', markersize=3)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No learning rate data\navailable', ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Epoch Training Time
        ax4 = axes[1, 0]
        if self.training_history['epoch_times']:
            ax4.plot(epochs, self.training_history['epoch_times'], 'purple', linewidth=2, marker='o', markersize=4)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Training Time (seconds)')
            ax4.set_title('Epoch Training Time')
            ax4.grid(True, alpha=0.3)
            
            # Add trend line
            if len(self.training_history['epoch_times']) > 1:
                z = np.polyfit(epochs, self.training_history['epoch_times'], 1)
                p = np.poly1d(z)
                ax4.plot(epochs, p(epochs), "r--", alpha=0.8, linewidth=1, label=f'Trend: {z[0]:.2f}s/epoch')
                ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax4.transAxes)
        
        # Plot 5: Batch Time Distribution
        ax5 = axes[1, 1]
        if self.training_history['batch_times']:
            n, bins, patches = ax5.hist(self.training_history['batch_times'], bins=50, alpha=0.7, 
                                       color='orange', edgecolor='black', density=True)
            ax5.set_xlabel('Batch Processing Time (seconds)')
            ax5.set_ylabel('Density')
            ax5.set_title('Batch Time Distribution')
            ax5.grid(True, alpha=0.3)
            
            # Add statistics
            mean_time = np.mean(self.training_history['batch_times'])
            std_time = np.std(self.training_history['batch_times'])
            median_time = np.median(self.training_history['batch_times'])
            ax5.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.3f}s')
            ax5.axvline(median_time, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_time:.3f}s')
            ax5.legend()
            
            # Add text box with detailed stats
            stats_text = f'Mean: {mean_time:.3f}s\nStd: {std_time:.3f}s\nMedian: {median_time:.3f}s\nMin: {min(self.training_history["batch_times"]):.3f}s\nMax: {max(self.training_history["batch_times"]):.3f}s'
            ax5.text(0.98, 0.98, stats_text, transform=ax5.transAxes, 
                    verticalalignment='top', horizontalalignment='right', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
        else:
            ax5.text(0.5, 0.5, 'No batch timing data available', ha='center', va='center', transform=ax5.transAxes)
        
        # Plot 6: Loss Components (if available from custom loss functions)
        ax6 = axes[1, 2]
        if hasattr(self, 'loss_components') and self.loss_components:
            # Plot different loss components
            for component_name, values in self.loss_components.items():
                if values:  # Only plot if we have data
                    component_epochs = range(1, len(values) + 1)
                    ax6.plot(component_epochs, values, label=component_name, linewidth=2, alpha=0.8)
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Loss Component Value')
            ax6.set_title('Loss Components Breakdown')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_yscale('log')
        else:
            ax6.text(0.5, 0.5, 'No loss component data\n(Standard MSE loss)', ha='center', va='center', transform=ax6.transAxes)
        
        # Plot 7: Gradient Norms (if tracked)
        ax7 = axes[2, 0]
        if hasattr(self, 'gradient_norms') and self.gradient_norms:
            grad_epochs = range(1, len(self.gradient_norms) + 1)
            ax7.plot(grad_epochs, self.gradient_norms, 'brown', linewidth=2, alpha=0.8)
            ax7.set_xlabel('Epoch')
            ax7.set_ylabel('Gradient Norm')
            ax7.set_title('Gradient Norm Evolution')
            ax7.grid(True, alpha=0.3)
            ax7.set_yscale('log')
            
            # Add gradient clipping threshold if applicable
            ax7.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Clipping Threshold')
            ax7.legend()
        else:
            ax7.text(0.5, 0.5, 'No gradient norm data\ntracked', ha='center', va='center', transform=ax7.transAxes)
        
        # Plot 8: Model Performance Metrics
        ax8 = axes[2, 1]
        if hasattr(self, 'performance_metrics') and self.performance_metrics:
            # Plot metrics like accuracy, correlation, etc.
            for metric_name, values in self.performance_metrics.items():
                if values:
                    metric_epochs = range(1, len(values) + 1)
                    if 'accuracy' in metric_name.lower():
                        # Convert to percentage for accuracy metrics
                        values_pct = [v * 100 for v in values]
                        ax8.plot(metric_epochs, values_pct, label=f'{metric_name} (%)', linewidth=2, alpha=0.8)
                    else:
                        ax8.plot(metric_epochs, values, label=metric_name, linewidth=2, alpha=0.8)
            
            ax8.set_xlabel('Epoch')
            ax8.set_ylabel('Metric Value')
            ax8.set_title('Model Performance Metrics')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
            
            # Add baseline for accuracy metrics
            if any('accuracy' in name.lower() for name in self.performance_metrics.keys()):
                ax8.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        else:
            ax8.text(0.5, 0.5, 'No performance metrics\ntracked', ha='center', va='center', transform=ax8.transAxes)
        
        # Plot 9: Training Summary and Statistics
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        # Create comprehensive summary text
        summary_parts = ['📊 Training Summary', '=' * 25]
        
        if epochs:
            summary_parts.extend([
                f'Total Epochs: {len(epochs)}',
                f'Best Val Loss: {min(self.training_history["val_loss"]):.6f}',
                f'Final Train Loss: {self.training_history["train_loss"][-1]:.6f}',
                f'Final Val Loss: {self.training_history["val_loss"][-1]:.6f}',
                ''
            ])
        
        # Training efficiency metrics
        if self.training_history['epoch_times']:
            total_time = sum(self.training_history['epoch_times'])
            avg_time = np.mean(self.training_history['epoch_times'])
            summary_parts.extend([
                '⏱️ Training Efficiency:',
                f'Total Time: {total_time/3600:.2f} hours',
                f'Avg Epoch Time: {avg_time:.1f}s',
                f'Est. Time/1000 epochs: {avg_time*1000/3600:.1f}h',
                ''
            ])
        
        # Batch processing stats
        if self.training_history['batch_times']:
            avg_batch_time = np.mean(self.training_history['batch_times'])
            total_batches = len(self.training_history['batch_times'])
            summary_parts.extend([
                '🔄 Batch Processing:',
                f'Total Batches: {total_batches:,}',
                f'Avg Batch Time: {avg_batch_time:.3f}s',
                f'Batches/sec: {1/avg_batch_time:.1f}',
                ''
            ])
        
        # Loss improvement analysis
        if len(self.training_history['val_loss']) > 1:
            initial_val_loss = self.training_history['val_loss'][0]
            final_val_loss = self.training_history['val_loss'][-1]
            improvement = (initial_val_loss - final_val_loss) / initial_val_loss * 100
            summary_parts.extend([
                '📈 Loss Improvement:',
                f'Initial Val Loss: {initial_val_loss:.6f}',
                f'Final Val Loss: {final_val_loss:.6f}',
                f'Improvement: {improvement:.1f}%',
                ''
            ])
        
        # Overfitting analysis
        if len(self.training_history['train_loss']) > 1:
            loss_diff = [v - t for t, v in zip(self.training_history['train_loss'], self.training_history['val_loss'])]
            avg_overfitting = np.mean([d for d in loss_diff if d > 0])
            overfitting_epochs = sum(1 for d in loss_diff if d > 0)
            summary_parts.extend([
                '🎯 Overfitting Analysis:',
                f'Overfitting Epochs: {overfitting_epochs}/{len(loss_diff)}',
                f'Avg Overfitting: {avg_overfitting:.6f}' if overfitting_epochs > 0 else 'No Overfitting Detected',
                ''
            ])
        
        # Performance metrics summary
        if hasattr(self, 'performance_metrics') and self.performance_metrics:
            summary_parts.append('🎯 Final Performance:')
            for metric_name, values in self.performance_metrics.items():
                if values:
                    final_value = values[-1]
                    if 'accuracy' in metric_name.lower():
                        summary_parts.append(f'{metric_name}: {final_value:.1%}')
                    else:
                        summary_parts.append(f'{metric_name}: {final_value:.4f}')
            summary_parts.append('')
        
        # Model configuration
        summary_parts.extend([
            '⚙️ Configuration:',
            f'Context Length: {SPLIT_CONFIG["context_minutes"]}',
            f'Horizon Length: {SPLIT_CONFIG["prediction_minutes"]}',
            f'Batch Size: {TRAINING_CONFIG["batch_size"]}',
            f'Learning Rate: {TRAINING_CONFIG["learning_rate"]}',
        ])
        
        ax9.text(0.05, 0.95, '\n'.join(summary_parts), 
                transform=ax9.transAxes, fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))
        
        # Add overall title
        plt.suptitle(f'TimesFM 2.0 Fine-tuning Analysis - {len(epochs)} Epochs', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Save plot with timestamp
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = FINETUNE_PLOTS_DIR / f"training_progress_{timestamp}.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Log to TensorBoard
        if self.tensorboard_writer is not None:
            # Convert plot to image for TensorBoard
            fig_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            fig_array = fig_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Add image to TensorBoard (HWC format)
            current_epoch = len(self.training_history['train_loss']) - 1
            self.tensorboard_writer.add_image('Training/ProgressPlot', fig_array, current_epoch, dataformats='HWC')
        
        plt.close()
        
        # Also save as latest.png for easy viewing
        latest_path = FINETUNE_PLOTS_DIR / 'training_progress_latest.png'
        import shutil
        shutil.copy(save_path, latest_path)
        
        print(f"   📊 Training progress plot saved: {save_path}")
        print(f"   📊 Latest plot: {latest_path}")
        return save_path
    
    def finetune_verbose(self, train_dataset: Dataset, val_dataset: Dataset) -> Dict[str, Any]:
        """Enhanced finetune method with verbose logging and plotting"""
        
        self.start_time = time.time()
        self.model = self.model.to(self.device)
        
        # Initialize TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_run_dir = TENSORBOARD_LOG_DIR / f"timesfm_run_{timestamp}"
        self.tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_run_dir))
        
        train_loader = self._create_dataloader(train_dataset, is_train=True)
        val_loader = self._create_dataloader(val_dataset, is_train=False)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        print(f"\n🚀 Starting Enhanced TimesFM 2.0 Finetuning")
        print(f"   Device: {self.device}")
        print(f"   Training samples: {len(train_dataset):,}")
        print(f"   Validation samples: {len(val_dataset):,}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Epochs: {self.config.num_epochs}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   TensorBoard logs: {tensorboard_run_dir}")
        print("=" * 60)
        
        # Log hyperparameters to TensorBoard
        hparams = {
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'num_epochs': self.config.num_epochs,
            'context_length': SPLIT_CONFIG["context_minutes"],
            'horizon_length': SPLIT_CONFIG["prediction_minutes"],
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device)
        }
        
        # Log model architecture (try to get a sample input for the graph)
        try:
            sample_batch = next(iter(train_loader))
            sample_input = (
                sample_batch['context'].to(self.device)[:1],  # Take first sample
                sample_batch['context_padding'].to(self.device)[:1],
                sample_batch['freq'].to(self.device)[:1]
            )
            self.tensorboard_writer.add_graph(self.model, sample_input)
            print("   📊 Model graph logged to TensorBoard")
        except Exception as e:
            print(f"   ⚠️ Could not log model graph: {e}")
        
        self.tensorboard_writer.add_hparams(hparams, {})
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        try:
            for epoch in range(self.config.num_epochs):
                # Training
                train_loss = self._train_epoch_verbose(train_loader, optimizer, epoch)
                self.training_history['train_loss'].append(train_loss)
                
                # Validation
                val_loss = self._validate_verbose(val_loader, epoch)
                self.training_history['val_loss'].append(val_loss)
                
                # Track learning rate
                current_lr = optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                
                # TensorBoard logging for learning rate
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar('Epoch/LearningRate', current_lr, epoch)
                
                # Track best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    print(f"   🌟 New best validation loss: {val_loss:.6f}")
                
                # Plot progress
                if (epoch + 1) % 2 == 0 or epoch == self.config.num_epochs - 1:
                    self.plot_training_progress()
                
                # Summary
                elapsed_time = time.time() - self.start_time
                print(f"   ⏱️  Total elapsed: {elapsed_time:.2f}s, ETA: {elapsed_time/(epoch+1)*(self.config.num_epochs-epoch-1):.2f}s")
                print("-" * 60)
                
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        
        total_time = time.time() - self.start_time
        
        print(f"\n✅ Finetuning completed!")
        print(f"   Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"   Best validation loss: {best_val_loss:.6f} (epoch {best_epoch + 1})")
        
        # Handle case where training was interrupted
        if self.training_history['train_loss']:
            print(f"   Final train loss: {self.training_history['train_loss'][-1]:.6f}")
        else:
            print(f"   Final train loss: N/A (training interrupted)")
            
        if self.training_history['val_loss']:
            print(f"   Final val loss: {self.training_history['val_loss'][-1]:.6f}")
        else:
            print(f"   Final val loss: N/A (validation not completed)")
        
        # Final plot
        final_plot_path = self.plot_training_progress()
        
        # Log final metrics to TensorBoard
        if self.tensorboard_writer is not None:
            final_metrics = {
                'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else float('inf'),
                'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else float('inf'),
                'best_val_loss': best_val_loss,
                'total_time_minutes': total_time / 60,
                'best_epoch': best_epoch,
                'epochs_completed': len(self.training_history['train_loss'])
            }
            
            # Update hparams with final results
            self.tensorboard_writer.add_hparams(hparams, final_metrics)
            
            # Close TensorBoard writer
            self.tensorboard_writer.close()
            print(f"   📊 TensorBoard logs saved to: {tensorboard_run_dir}")
            print(f"   🌐 Start TensorBoard with: tensorboard --logdir={TENSORBOARD_LOG_DIR}")
            print(f"   🌐 Then open: http://localhost:6006")
        
        return {
            'history': self.training_history,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'total_time': total_time,
            'final_plot': final_plot_path,
            'tensorboard_dir': str(tensorboard_run_dir) if self.tensorboard_writer else None
        }

# ==============================================================================
# Training Functions
# ==============================================================================

def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('timesfm_finetuning.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def run_finetuning(train_dataset: Dataset, 
                  val_dataset: Dataset,
                  model: nn.Module) -> Dict[str, Any]:
    """Run the finetuning process with verbose logging and plotting"""
    
    logger = setup_logging()
    
    # Create finetuning configuration
    config = FinetuningConfig(
        batch_size=TRAINING_CONFIG["batch_size"],
        num_epochs=TRAINING_CONFIG["num_epochs"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        freq_type=TRAINING_CONFIG["freq_type"],
        use_quantile_loss=TRAINING_CONFIG["use_quantile_loss"],
        use_wandb=TRAINING_CONFIG["use_wandb"],
        wandb_project=TRAINING_CONFIG["wandb_project"],
        log_every_n_steps=TRAINING_CONFIG["log_every_n_steps"],
        val_check_interval=TRAINING_CONFIG["val_check_interval"],
    )
    
    # Create verbose finetuner
    finetuner = VerboseTimesFMFinetuner(model, config, logger=logger)
    
    print(f"\n🚀 Starting Enhanced TimesFM 2.0 finetuning...")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Plots will be saved to: {FINETUNE_PLOTS_DIR}")
    
    # Run verbose finetuning
    results = finetuner.finetune_verbose(
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    return results

def save_model(model: nn.Module, model_info: Dict[str, Any], results: Dict[str, Any]):
    """Save the finetuned model"""
    
    save_dir = CACHE_DIR / f"finetuned_timesfm2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(exist_ok=True)
    
    # Save model state dict
    model_path = save_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    
    # Save configuration and results
    config_path = save_dir / "config.json"
    import json
    
    save_config = {
        'model_config': MODEL_CONFIG,
        'training_config': TRAINING_CONFIG,
        'split_config': SPLIT_CONFIG,
        'training_results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(config_path, 'w') as f:
        json.dump(save_config, f, indent=2, default=str)
    
    print(f"💾 Model saved to: {save_dir}")
    print(f"   Model weights: {model_path}")
    print(f"   Configuration: {config_path}")
    
    return save_dir

# ==============================================================================
# Evaluation Functions
# ==============================================================================

def evaluate_model(model: nn.Module, test_dataset: Dataset, device: str) -> Dict[str, float]:
    """Evaluate the finetuned model on test set"""
    
    print("📊 Evaluating model on test set...")
    
    model.eval()
    model = model.to(device)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    total_loss = 0.0
    total_samples = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            x_context, x_padding, freq, x_future = [t.to(device) for t in batch]
            
            # Forward pass
            pred = model(x_context, x_padding.float(), freq)
            pred_mean = pred[..., 0]  # Get mean predictions
            last_patch_pred = pred_mean[:, -1, :]  # [B, horizon_len]
            
            # Calculate loss
            loss = torch.mean((last_patch_pred - x_future.squeeze(-1)) ** 2)
            total_loss += loss.item() * x_context.size(0)
            total_samples += x_context.size(0)
            
            # Store predictions for analysis
            predictions.extend(last_patch_pred.cpu().numpy())
            actuals.extend(x_future.squeeze(-1).cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / total_samples
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Additional metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    
    # Directional accuracy
    pred_direction = np.sign(np.diff(predictions, axis=1))
    actual_direction = np.sign(np.diff(actuals, axis=1))
    directional_accuracy = np.mean(pred_direction == actual_direction)
    
    metrics = {
        'test_loss': avg_loss,
        'mse': mse,
        'mae': mae,
        'directional_accuracy': directional_accuracy,
        'num_samples': total_samples
    }
    
    print(f"📈 Test Results:")
    print(f"   Test Loss: {avg_loss:.6f}")
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    print(f"   Directional Accuracy: {directional_accuracy:.3f}")
    print(f"   Samples: {total_samples:,}")
    
    return metrics

# ==============================================================================
# Main Function
# ==============================================================================

def main():
    """Main finetuning pipeline"""
    
    print("🚀 TimesFM 2.0 ES Futures Finetuning Pipeline")
    print("=" * 60)
    print()
    print("📊 TENSORBOARD MONITORING ENABLED")
    print("   1. Training will start shortly...")
    print("   2. Open a new terminal and run: tensorboard --logdir=tensorboard_logs")
    print("   3. Open browser to: http://localhost:6006")
    print("   4. View real-time training progress, metrics, and plots!")
    print("=" * 60)
    
    try:
        # 1. Load and split sequence files
        print("\n📁 Step 1: Loading sequence files...")
        sequence_files = get_sequence_files()
        train_files, val_files, test_files = split_sequences_by_time(sequence_files)
        
        # 2. Create datasets
        print("\n📊 Step 2: Creating datasets...")
        train_dataset, val_dataset, test_dataset = create_datasets(train_files, val_files, test_files)
        
        # 3. Create model
        print("\n🤖 Step 3: Creating model...")
        model, model_info = create_timesfm_model(load_weights=True)
        
        # 4. Run finetuning
        print("\n🎯 Step 4: Finetuning...")
        results = run_finetuning(train_dataset, val_dataset, model)
        
        # 5. Evaluate model
        print("\n📊 Step 5: Evaluation...")
        test_metrics = evaluate_model(model, test_dataset, model_info['device'])
        results['test_metrics'] = test_metrics
        
        # 6. Save model
        print("\n💾 Step 6: Saving model...")
        save_dir = save_model(model, model_info, results)
        
        print(f"\n✅ Finetuning completed successfully!")
        print(f"   Model saved to: {save_dir}")
        print(f"   Final test loss: {test_metrics['test_loss']:.6f}")
        print(f"   Directional accuracy: {test_metrics['directional_accuracy']:.3f}")
        
    except Exception as e:
        print(f"\n❌ Error during finetuning: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
