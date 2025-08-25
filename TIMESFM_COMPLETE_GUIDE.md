# 🚀 TimesFM Stock Prediction System - Complete Guide

## Overview

This project implements a comprehensive stock prediction system using Google's TimesFM model with two main components:

1. **`good_timefm_stock_working.py`** - A working implementation that uses the official pre-trained TimesFM model for ES futures prediction
2. **`pytorch_timesfm_finetune.py`** - A fine-tuning pipeline that trains a custom implementation on your specific data while comparing against the official model

## 📊 Core Configuration

Both scripts use identical time window parameters for consistency:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Context Length** | 448 minutes | ~7.5 hours of historical minute-level data |
| **Horizon Length** | 64 minutes | ~1 hour prediction window |
| **Patch Length** | 32 minutes | Divides context into 14 patches (448÷32) |
| **Data Normalization** | Yes | Normalized prices improve model performance |
| **Backend** | Auto-detect | GPU if available, otherwise CPU |

## 🎯 Working Example: `good_timefm_stock_working.py`

### Purpose
Demonstrates the official TimesFM model's capabilities on ES futures data with real-time predictions throughout a trading day.

### Key Features
- Uses Google's pre-trained `timesfm-1.0-200m-pytorch` model
- Makes predictions every 30 minutes during trading hours
- Visualizes predictions vs ground truth
- Calculates MSE/MAE metrics
- Supports GPU acceleration

### Usage
```bash
python good_timefm_stock_working.py
```

### Output
- Main plot showing all predictions on the trading day
- Individual plots for each prediction point
- Direction analysis (UP/DOWN/FLAT)
- Saved to `stock_plots/` directory

## 🔧 Fine-tuning Pipeline: `pytorch_timesfm_finetune.py`

### Purpose
Fine-tunes a custom PyTorch implementation of TimesFM on your specific ES futures data while comparing performance against Google's official model.

### Key Features
- Custom PyTorch implementation that can be trained
- Side-by-side comparison with official TimesFM
- Real-time training progress visualization
- Checkpoint saving and best model tracking
- Monte Carlo visualization of data distribution

### Official Model Integration
The fine-tuning script now uses the official Google TimesFM as the baseline comparison:

```python
# Initialize official TimesFM model for comparison
official_model = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend=TIMESFM_BACKEND,
        per_core_batch_size=32,
        horizon_len=128,
        num_layers=20,
        use_positional_embedding=True,
        context_len=512,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    ),
)
```

### Training Process
1. **Data Loading**: Loads ES futures data with proper instrument grouping
2. **Preprocessing**: Creates normalized sequences with context/horizon splits
3. **Training**: Fine-tunes custom model with AdamW optimizer
4. **Comparison**: Real-time plots showing:
   - Official TimesFM predictions (red dashed)
   - Fine-tuned model predictions (green solid)
   - Ground truth (black)
   - MSE improvement percentage

### Usage
```bash
# Basic usage (uses cache if available)
python pytorch_timesfm_finetune.py

# Force data reprocessing (ignore cache)
python pytorch_timesfm_finetune.py --force-reprocess

# Disable cache for this run only
python pytorch_timesfm_finetune.py --no-cache

# Skip visualization to save time
python pytorch_timesfm_finetune.py --skip-visualization

# Keep existing plots (don't clear at startup)
python pytorch_timesfm_finetune.py --keep-plots

# Custom training parameters
python pytorch_timesfm_finetune.py --epochs 100 --batch-size 16

# Clear cache manually
python clear_cache.py
```

### Data Caching

The fine-tuning script includes intelligent data caching to save time on subsequent runs:

#### How it Works
1. **First Run**: Processes data and saves to cache (~2-5 minutes)
2. **Subsequent Runs**: Loads from cache (~5 seconds)
3. **Cache Key**: Based on dataset path, configuration, and version

#### Cache Configuration
```python
USE_CACHED_DATA = True   # Enable/disable caching globally
CACHE_VERSION = "v2"     # Increment to invalidate old caches
```

#### When Cache is Invalidated
- Different context/horizon lengths
- Different sequence stride or max sequences
- Changed cache version
- Using `--force-reprocess` flag

### Output
- Training progress plots in `finetune_plots/`
- Monte Carlo visualization in `plots/preprocessing_monte_carlo.png`
- Individual sequence plots in `plots/`
- Best model checkpoint in `finetune_checkpoints/`

## 📈 Data Processing

Both scripts use the same data processing approach:

### Data Source
- File: `databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv`
- Format: 1-minute OHLCV bars for ES futures
- Filtering: Only single contracts (ESH4, ESM4, etc.), no spreads

### Normalization
- Prices are normalized relative to sequence start
- Helps model learn patterns rather than absolute levels
- Both scripts use `close_norm` column

### Sequence Creation
- Context: 448 minutes of historical data
- Target: 64 minutes of future data
- Stride: 64 minutes between sequences
- Grouping: By instrument_id to prevent mixing contracts

## 🔥 Hardware Requirements

### GPU Support
- **Automatic Detection**: Both scripts auto-detect GPU availability
- **Supported GPUs**: NVIDIA with compute capability 6.0+
- **Memory Requirements**: ~4GB for inference, ~8GB for training

### CPU Fallback
- Both scripts work on CPU (slower but functional)
- Batch sizes automatically adjusted
- No code changes needed

## 📊 Visualization Features

### Working Example Plots
1. **Main Overview**: All predictions on single chart with direction indicators
2. **Individual Predictions**: Detailed view of each prediction point
3. **Error Metrics**: MSE/MAE for each prediction

### Fine-tuning Plots
1. **Training Progress**: Loss curves, learning rate schedule
2. **Model Comparison**: Official vs Fine-tuned predictions
3. **Error Distribution**: Histogram of prediction errors
4. **Monte Carlo Analysis**: Distribution of training sequences

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Install requirements
pip install torch transformers timesfm pandas numpy matplotlib

# Verify GPU (optional)
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

### 2. Prepare Data
Place your data file at:
```
databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv
```

### 3. Run Working Example
```bash
# Test official TimesFM on your data
python good_timefm_stock_working.py
```

### 4. Fine-tune Model
```bash
# Train custom model with comparisons
python pytorch_timesfm_finetune.py
```

### 5. Test Integration
```bash
# Verify TimesFM integration
python test_timesfm_integration.py
```

## 📈 Performance Metrics

### Core Metrics

#### Directional Accuracy (Primary Focus)
The fine-tuning pipeline now emphasizes **directional accuracy** - correctly predicting whether prices will go up or down:

- **Directional Accuracy %**: Percentage of timesteps where the model correctly predicts the direction of price movement
- **Correlation**: Pearson correlation between predictions and ground truth (captures overall trend alignment)
- **Sign Agreement**: Whether predicted and actual price changes have the same sign

#### Traditional Metrics
- **MSE**: Mean Squared Error between predictions and ground truth
- **MAE**: Mean Absolute Error for interpretability
- **Improvement %**: How much fine-tuning improves over official model

### Custom Directional Loss Function

The pipeline includes a sophisticated loss function that balances accuracy and directionality:

```python
# Configuration in pytorch_timesfm_finetune.py
USE_DIRECTIONAL_LOSS = True  # Enable directional training
DIRECTIONAL_WEIGHT = 0.5     # 50% weight on direction, 50% on MSE
```

The loss combines:
1. **MSE Component**: Traditional squared error for magnitude accuracy
2. **Directional Component**: Penalizes incorrect direction predictions
3. **Correlation Loss**: Encourages matching overall trend patterns

### Visualization Updates

All plots now show:
- **Fixed Y-axis scale (0.98-1.02)**: Better visibility of small movements
- **Directional accuracy tracking**: Real-time accuracy during training
- **Multi-metric comparison**: MSE, directional accuracy, and correlation

### Expected Results
- **Random baseline**: 50% directional accuracy (coin flip)
- **Good model**: 60-70% directional accuracy
- **Excellent model**: 70%+ directional accuracy with high correlation
- **Focus**: Better to have 65% directional accuracy than perfect MSE with wrong directions

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size in pytorch_timesfm_finetune.py
   BATCH_SIZE = 4  # Instead of 8
   ```

2. **Import Errors**
   ```bash
   # Ensure timesfm is installed
   pip install timesfm[torch]
   ```

3. **Data Not Found**
   - Check file path matches exactly
   - Ensure databento/ES/ directory exists
   - Verify CSV has required columns

### Debugging Tips
- Check GPU status: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Test data loading separately before training

## 🎯 Next Steps

1. **Experiment with Hyperparameters**
   - Learning rate: Try 1e-4 to 1e-6
   - Batch size: Larger = more stable, smaller = faster updates
   - Sequence stride: Smaller = more training data

2. **Extend Prediction Horizon**
   - Modify HORIZON_LENGTH (must be ≤128)
   - Adjust visualization accordingly

3. **Add More Features**
   - Include volume data
   - Add technical indicators
   - Multi-instrument training

4. **Production Deployment**
   - Save best model checkpoint
   - Create inference-only script
   - Add real-time data feed

## 📚 References

- [TimesFM Paper](https://arxiv.org/abs/2310.10688)
- [Official TimesFM Repository](https://github.com/google-research/timesfm)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

*Happy Trading! 🚀📈*
