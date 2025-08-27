#!/usr/bin/env python3
"""
Chronos-T5-Large Prediction Interface
====================================

This module provides a clean interface for making predictions using Amazon's Chronos-T5-Large
time series forecasting model. Chronos is a family of pretrained time series forecasting 
models based on language model architectures (T5).

Key Features:
- Uses Amazon's Chronos-T5-Large (710M parameters) for high-quality predictions
- Accepts normalized close price time series data
- Compatible with existing prediction system architecture
- Integrated error handling with human-readable error codes
- Model caching for efficient repeated predictions
- GPU acceleration when available

Model Details:
- Architecture: T5-based transformer with 710M parameters
- Input: Normalized time series data (close prices)
- Context Length: 416 minutes (configurable)
- Prediction Horizon: 96 minutes (configurable)
- Output: Probabilistic forecasts with multiple samples

Author: AI Assistant
Date: 2025-08-26
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import plotting module
from plotting import plot_prediction_results

# ==============================================================================
# Configuration
# ==============================================================================

# Model Configuration - Chronos-T5-Large
MODEL_NAME = "amazon/chronos-t5-large"
CONTEXT_LENGTH = 416    # Historical context in minutes (~6.9 hours)
HORIZON_LENGTH = 64     # Prediction horizon in minutes (~1.6 hours)
NUM_SAMPLES = 100     # Number of forecast samples for probabilistic prediction (increased for better quality)

# Inference Configuration
TEMPERATURE = 1.0     # Sampling temperature (1.0 = default, higher = more diverse)
TOP_K = 50           # Top-k sampling
TOP_P = 1.0          # Nucleus sampling

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Data Configuration
DATASETS_DIR = Path("datasets/ES")
PLOTS_DIR = Path("prediction_plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ==============================================================================
# Error Codes
# ==============================================================================

ERROR_CODES = {
    'CHRONOS_001': 'Model loading failed - Check internet connection and model availability',
    'CHRONOS_002': 'Invalid input data - Data must be 1D array or list of normalized close prices',
    'CHRONOS_003': 'Insufficient data - Need at least 50 data points for reliable prediction',
    'CHRONOS_004': 'Data validation failed - Contains NaN, inf, or non-numeric values',
    'CHRONOS_005': 'Context length mismatch - Input data length does not match expected context length',
    'CHRONOS_006': 'Prediction failed - Error during model inference',
    'CHRONOS_007': 'Device error - GPU requested but not available',
    'CHRONOS_008': 'Memory error - Insufficient memory for model or prediction',
    'CHRONOS_009': 'Data normalization error - Unable to normalize input data',
    'CHRONOS_010': 'Output processing error - Failed to process model predictions'
}

# ==============================================================================
# Global Model Cache
# ==============================================================================

_chronos_model = None

def _load_model():
    """Load and cache the Chronos-T5-Large model."""
    global _chronos_model
    
    if _chronos_model is not None:
        return _chronos_model
    
    try:
        print(f"🤖 Loading Chronos-T5-Large model...")
        print(f"   Model: {MODEL_NAME}")
        print(f"   Device: {DEVICE}")
        print(f"   Dtype: {TORCH_DTYPE}")
        
        # Import chronos here to provide better error message if not installed
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "Chronos package not found. Please install it with:\n"
                "pip install git+https://github.com/amazon-science/chronos-forecasting.git"
            )
        
        # Load the model with appropriate device and dtype
        _chronos_model = ChronosPipeline.from_pretrained(
            MODEL_NAME,
            device_map=DEVICE,
            torch_dtype=TORCH_DTYPE,
        )
        
        print(f"✅ Chronos-T5-Large model loaded successfully")
        return _chronos_model
        
    except ImportError as e:
        error_msg = f"{ERROR_CODES['CHRONOS_001']}: {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"{ERROR_CODES['CHRONOS_001']}: {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

def _validate_input_data(data, context_length=CONTEXT_LENGTH):
    """Validate input data for Chronos prediction."""
    try:
        # Convert to numpy array if needed
        if isinstance(data, (list, pd.Series)):
            data = np.array(data)
        elif isinstance(data, torch.Tensor):
            data = data.numpy()
        elif not isinstance(data, np.ndarray):
            raise ValueError("Data must be array-like (list, numpy array, pandas Series, or torch Tensor)")
        
        # Check for 1D data
        if data.ndim != 1:
            raise ValueError(f"Data must be 1D, got shape {data.shape}")
        
        # Check minimum length
        if len(data) < 50:
            raise ValueError(f"{ERROR_CODES['CHRONOS_003']}: Got {len(data)} points")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError(f"{ERROR_CODES['CHRONOS_004']}")
        
        # Check context length match
        if len(data) != context_length:
            raise ValueError(f"{ERROR_CODES['CHRONOS_005']}: Expected {context_length}, got {len(data)}")
        
        return data.astype(np.float32)
        
    except Exception as e:
        if "CHRONOS_" in str(e):
            raise RuntimeError(str(e))
        else:
            raise RuntimeError(f"{ERROR_CODES['CHRONOS_002']}: {str(e)}")

def _prepare_model_input(data, verbose=False):
    """Prepare data for Chronos model input - data is already normalized."""
    try:
        # Validate input data
        validated_data = _validate_input_data(data)
        
        # Data is already normalized to start at 1.0, so no additional scaling needed
        # Convert directly to torch tensor
        context_tensor = torch.tensor(validated_data, dtype=torch.float32)
        
        if verbose:
            print(f"🔍 Validating input data...")
            print(f"✅ Data validated: {len(validated_data)} points, range [{validated_data.min():.6f}, {validated_data.max():.6f}]")
            print(f"📊 Preparing data for Chronos...")
            print(f"   Using pre-normalized data (no additional scaling)")
            print(f"   Context tensor shape: {context_tensor.shape}")
            print(f"   Context tensor dtype: {context_tensor.dtype}")
        
        return context_tensor  # No scaling factor needed
        
    except Exception as e:
        if "CHRONOS_" in str(e):
            raise RuntimeError(str(e))
        else:
            raise RuntimeError(f"{ERROR_CODES['CHRONOS_009']}: {str(e)}")

# ==============================================================================
# Main Prediction Function
# ==============================================================================

def predict_chronos(close_data, context_length=CONTEXT_LENGTH, horizon_length=HORIZON_LENGTH, 
                   num_samples=NUM_SAMPLES, verbose=False):
    """
    Generate predictions using Amazon's Chronos-T5-Large model.
    
    Args:
        close_data (array-like): Normalized close price time series data
        context_length (int): Length of historical context (default: 416)
        horizon_length (int): Length of prediction horizon (default: 96)
        num_samples (int): Number of forecast samples (default: 100)
        verbose (bool): Whether to print detailed progress information
    
    Returns:
        dict: Dictionary containing:
            - 'predictions': numpy array of median predictions (horizon_length,)
            - 'prediction_samples': numpy array of all samples (num_samples, horizon_length)
            - 'confidence_intervals': dict with 'low' and 'high' bounds
            - 'metadata': dict with model and prediction information
    
    Raises:
        RuntimeError: If prediction fails with specific error code
    """
    
    if verbose:
        print(f"🔮 Starting Chronos-T5-Large prediction...")
        print(f"   Context length: {context_length} minutes")
        print(f"   Horizon length: {horizon_length} minutes")
        print(f"   Number of samples: {num_samples}")
    
    try:
        # Prepare input data (already normalized)
        context_tensor = _prepare_model_input(close_data, verbose=verbose)
        
        # Load model
        if verbose:
            print(f"🤖 Loading model...")
        model = _load_model()
        
        # Generate prediction
        if verbose:
            print(f"🔮 Generating prediction with Chronos-T5-Large...")
            print(f"   Context range: [{context_tensor.min():.6f}, {context_tensor.max():.6f}]")
            print(f"   Context end value: {context_tensor[-1]:.6f}")
        
        # Perform inference with improved parameters
        with torch.no_grad():
            forecast = model.predict(
                context=context_tensor, 
                prediction_length=horizon_length,
                num_samples=num_samples,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P
            )
        
        # Process output
        # forecast shape: [num_series, num_samples, prediction_length]
        # Since we have 1 series: [1, num_samples, prediction_length]
        forecast_samples = forecast[0].numpy()  # Shape: [num_samples, prediction_length]
        
        # No scaling needed - data is already in correct normalized format
        # Calculate statistics directly on predictions
        median_forecast = np.median(forecast_samples, axis=0)  # Shape: [prediction_length]
        low_quantile = np.quantile(forecast_samples, 0.1, axis=0)   # 10th percentile
        high_quantile = np.quantile(forecast_samples, 0.9, axis=0)  # 90th percentile
        
        if verbose:
            print(f"✅ Prediction generated: {len(median_forecast)} points")
            print(f"   Range: [{median_forecast.min():.6f}, {median_forecast.max():.6f}]")
            print(f"   First 5: {median_forecast[:5]}")
            print(f"   Last 5: {median_forecast[-5:]}")
            print(f"   Prediction start: {median_forecast[0]:.6f}")
            # Calculate continuity gap (no scaling needed)
            context_end = close_data[-1] if hasattr(close_data, '__getitem__') else context_tensor[-1].item()
            print(f"   Continuity gap: {median_forecast[0] - context_end:.6f}")
        
        # Prepare return data
        result = {
            'predictions': median_forecast,
            'prediction_samples': forecast_samples,
            'confidence_intervals': {
                'low': low_quantile,
                'high': high_quantile
            },
            'metadata': {
                'model_name': 'Chronos-T5-Large',
                'model_repo': MODEL_NAME,
                'context_length': context_length,
                'horizon_length': horizon_length,
                'num_samples': num_samples,
                'device': DEVICE,
                'parameters': '710M',
                'prediction_start': median_forecast[0],
                'prediction_end': median_forecast[-1],
                'context_end': close_data[-1] if hasattr(close_data, '__getitem__') else context_tensor[-1].item(),
                'continuity_gap': median_forecast[0] - (close_data[-1] if hasattr(close_data, '__getitem__') else context_tensor[-1].item())
            }
        }
        
        return result
        
    except RuntimeError as e:
        # Re-raise RuntimeError with error codes
        raise e
    except Exception as e:
        error_msg = f"{ERROR_CODES['CHRONOS_006']}: {str(e)}"
        if verbose:
            print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

# ==============================================================================
# Testing and Demonstration
# ==============================================================================

def _test_with_random_csv():
    """Test Chronos prediction with a random CSV file from the datasets."""
    print("🧪 Testing Chronos-T5-Large prediction with random dataset...")
    
    try:
        # Get all CSV files
        csv_files = list(DATASETS_DIR.glob("*.csv"))
        if not csv_files:
            print(f"❌ No CSV files found in {DATASETS_DIR}")
            return
        
        # Select random file
        random_file = random.choice(csv_files)
        print(f"📁 Selected random file: {random_file.name}")
        
        # Load the dataset
        df = pd.read_csv(random_file)
        print(f"📊 Loaded {len(df)} rows")
        
        # Parse timestamp and sort
        df['timestamp'] = pd.to_datetime(df['timestamp_pt'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Find 7 AM Pacific time
        df['hour'] = df['timestamp'].dt.hour
        start_candidates = df[df['hour'] == 7].index.tolist()
        
        if not start_candidates:
            print("⚠️  No 7 AM data found, using random start point")
            start_candidates = list(range(CONTEXT_LENGTH, len(df) - HORIZON_LENGTH))
        
        if not start_candidates:
            print(f"❌ Insufficient data in file. Need at least {CONTEXT_LENGTH + HORIZON_LENGTH} points")
            return
        
        # Select random start point around 7 AM
        start_idx = random.choice(start_candidates)
        start_time = df.loc[start_idx, 'timestamp']
        print(f"🕰️ Starting prediction at index {start_idx} ({start_time})")
        
        # Extract context and future data
        context_end_idx = start_idx + CONTEXT_LENGTH
        future_end_idx = context_end_idx + HORIZON_LENGTH
        
        if future_end_idx > len(df):
            print(f"❌ Not enough future data. Adjusting indices.")
            future_end_idx = len(df)
            context_end_idx = future_end_idx - HORIZON_LENGTH
            start_idx = context_end_idx - CONTEXT_LENGTH
        
        # Get historical data (context)
        historical_data = df.iloc[start_idx:context_end_idx].copy()
        actual_future = df.iloc[context_end_idx:future_end_idx].copy()
        
        print(f"📈 Historical data: {len(historical_data)} points")
        print(f"   Range: [{historical_data['close'].min():.6f}, {historical_data['close'].max():.6f}]")
        print(f"   First: {historical_data['close'].iloc[0]:.6f}, Last: {historical_data['close'].iloc[-1]:.6f}")
        
        # Extract close prices for prediction
        close_prices = historical_data['close'].values
        
        # Make prediction
        result = predict_chronos(
            close_data=close_prices,
            context_length=CONTEXT_LENGTH,
            horizon_length=HORIZON_LENGTH,
            num_samples=NUM_SAMPLES,
            verbose=True
        )
        
        predictions = result['predictions']
        confidence_intervals = result['confidence_intervals']
        metadata = result['metadata']
        
        print(f"\n✅ Prediction successful!")
        print(f"📊 Predicted next {HORIZON_LENGTH} minutes:")
        print(f"   Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"   First 10: {predictions[:10]}")
        print(f"   Last 10: {predictions[-10:]}")
        
        # Compare with actual future data if available
        if len(actual_future) > 0:
            actual_close = actual_future['close'].values
            print(f"\n📊 Actual future data:")
            print(f"   Range: [{actual_close.min():.6f}, {actual_close.max():.6f}]")
            print(f"   First 10: {actual_close[:10]}")
            print(f"   Last 10: {actual_close[-10:]}")
            
            # Calculate basic metrics
            min_len = min(len(predictions), len(actual_close))
            if min_len > 0:
                mse = np.mean((predictions[:min_len] - actual_close[:min_len]) ** 2)
                mae = np.mean(np.abs(predictions[:min_len] - actual_close[:min_len]))
                
                print(f"\n📈 Quick evaluation:")
                print(f"   MSE: {mse:.6f}")
                print(f"   MAE: {mae:.6f}")
        
        # Continuity check
        context_end = close_prices[-1]
        prediction_start = predictions[0]
        actual_start = actual_future['close'].iloc[0] if len(actual_future) > 0 else None
        
        print(f"\n🔗 Continuity check:")
        print(f"   Context end: {context_end:.6f}")
        print(f"   Prediction start: {prediction_start:.6f} (diff: {prediction_start - context_end:.6f})")
        if actual_start is not None:
            print(f"   Actual start: {actual_start:.6f} (diff: {actual_start - context_end:.6f})")
        
        # Create visualization plot
        print(f"\n🎨 Creating visualization plot...")
        try:
            # Prepare data for plotting
            # Create save path for the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = PLOTS_DIR / f"chronos-t5-large_prediction_{timestamp}.png"
            
            # Prepare ground truth data (extract close prices if available)
            ground_truth_close = None
            if len(actual_future) > 0:
                ground_truth_close = actual_future['close'].values
            
            plot_result = plot_prediction_results(
                context_data=historical_data['close'].values,  # Extract close prices for context
                prediction_data=predictions,
                ground_truth_data=ground_truth_close,
                title=f"Chronos-T5-Large Prediction - {random_file.stem}",
                model_name="Chronos-T5-Large",
                save_path=str(save_path),
                show_plot=False
            )
            
            if plot_result and 'plot_path' in plot_result:
                print(f"📊 Plot saved to: {plot_result['plot_path']}")
                if 'metrics' in plot_result:
                    metrics = plot_result['metrics']
                    print(f"📈 Detailed metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"   {key}: {value:.3f}")
                        else:
                            print(f"   {key}: {value}")
            
        except Exception as plot_error:
            print(f"⚠️  Plotting failed: {plot_error}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("🚀 Chronos-T5-Large Prediction Interface")
    print("=" * 50)
    print(f"Model Type: Chronos-T5-Large")
    print(f"Parameters: 710M")
    print(f"Context Length: {CONTEXT_LENGTH} minutes")
    print(f"Horizon Length: {HORIZON_LENGTH} minutes")
    print(f"Number of Samples: {NUM_SAMPLES}")
    
    _test_with_random_csv()
