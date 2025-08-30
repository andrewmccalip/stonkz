#!/usr/bin/env python3
"""
TimesFM v2.0 Prediction Interface - Clean inference API using official TimesFM 2.0 model.
Uses the official Google TimesFM 2.0 API with proper normalization and configuration.
"""

import os
import sys
import random
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sys
sys.path.append('timesfm/src')
import timesfm.timesfm_torch as timesfm

# Add project paths
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

# Import plotting module
from plotting import plot_prediction_results

# ==============================================================================
# Configuration
# ==============================================================================

# Model Configuration - TimesFM 2.0 500M parameters
MODEL_REPO = "google/timesfm-2.0-500m-pytorch"
CONTEXT_LENGTH = 416    # Historical context in minutes (~6.9 hours)
HORIZON_LENGTH = 96     # Prediction horizon in minutes (~1.6 hours)
DEVICE = "cpu"  # Use CPU for compatibility

# TimesFM 2.0 Fixed Parameters (required for 500M model)
INPUT_PATCH_LEN = 32
OUTPUT_PATCH_LEN = 128
NUM_LAYERS = 50
MODEL_DIMS = 1280
USE_POSITIONAL_EMBEDDING = False

# Dataset Configuration
DATASETS_DIR = SCRIPT_DIR / "datasets" / "ES"

# ==============================================================================
# Error Codes (same as before)
# ==============================================================================

class PredictionError:
    """Error codes for TimesFM prediction system"""
    
    # Success
    SUCCESS = 0                    # "Prediction completed successfully"
    
    # Input Data Errors (100-199)
    INSUFFICIENT_DATA = 101        # "Not enough historical data points (need at least 416)"
    INVALID_DATA_FORMAT = 102      # "Input data format is invalid (expected 1D array or list)"
    CONTAINS_NAN = 103            # "Input data contains NaN or infinite values"
    CONTAINS_ZEROS = 104          # "Input data contains zero values (cannot normalize)"
    DATA_NOT_NORMALIZED = 105     # "Input data appears to not be normalized (expected to start near 1.0)"
    
    # Model Errors (200-299)
    MODEL_LOAD_FAILED = 201       # "Failed to load TimesFM model"
    MODEL_WEIGHTS_FAILED = 202    # "Failed to load pre-trained weights"
    MODEL_INFERENCE_FAILED = 203  # "Model inference failed during prediction"
    DEVICE_ERROR = 204           # "Failed to move model/data to specified device"
    
    # System Errors (300-399)
    MEMORY_ERROR = 301           # "Insufficient memory for prediction"
    CUDA_ERROR = 302             # "CUDA error during processing"
    UNKNOWN_ERROR = 399          # "Unknown error occurred"

# Error code to message mapping
ERROR_MESSAGES = {
    PredictionError.SUCCESS: "Prediction completed successfully",
    PredictionError.INSUFFICIENT_DATA: "Not enough historical data points (need at least 416)",
    PredictionError.INVALID_DATA_FORMAT: "Input data format is invalid (expected 1D array or list)",
    PredictionError.CONTAINS_NAN: "Input data contains NaN or infinite values",
    PredictionError.CONTAINS_ZEROS: "Input data contains zero values (cannot normalize)",
    PredictionError.DATA_NOT_NORMALIZED: "Input data appears to not be normalized (expected to start near 1.0)",
    PredictionError.MODEL_LOAD_FAILED: "Failed to load TimesFM model",
    PredictionError.MODEL_WEIGHTS_FAILED: "Failed to load pre-trained weights",
    PredictionError.MODEL_INFERENCE_FAILED: "Model inference failed during prediction",
    PredictionError.DEVICE_ERROR: "Failed to move model/data to specified device",
    PredictionError.MEMORY_ERROR: "Insufficient memory for prediction",
    PredictionError.CUDA_ERROR: "CUDA error during processing",
    PredictionError.UNKNOWN_ERROR: "Unknown error occurred"
}

def get_error_message(error_code):
    """Get human-readable error message for error code"""
    return ERROR_MESSAGES.get(error_code, f"Unknown error code: {error_code}")

# ==============================================================================
# Model Cache
# ==============================================================================

_cached_model = None
_model_loaded = False

def _load_model():
    """Load and cache the official TimesFM 2.0 model"""
    global _cached_model, _model_loaded
    
    if _model_loaded and _cached_model is not None:
        return _cached_model, PredictionError.SUCCESS
    
    try:
        print(f"🤖 Loading TimesFM 2.0 500M model...")
        
        # Initialize TimesFM 2.0 with correct parameters
        model = timesfm.TimesFmTorch(
            hparams=timesfm.timesfm_base.TimesFmHparams(
                backend=DEVICE,
                per_core_batch_size=32,
                horizon_len=HORIZON_LENGTH,  # Set to our desired horizon
                input_patch_len=INPUT_PATCH_LEN,      # Fixed for 500M model
                output_patch_len=OUTPUT_PATCH_LEN,    # Fixed for 500M model
                num_layers=NUM_LAYERS,                # Fixed for 500M model
                model_dims=MODEL_DIMS,                # Fixed for 500M model
                use_positional_embedding=USE_POSITIONAL_EMBEDDING,  # Fixed for 500M model
                context_len=2048,  # Model's max context (we'll use up to 416)
            ),
            checkpoint=timesfm.timesfm_base.TimesFmCheckpoint(
                huggingface_repo_id=MODEL_REPO
            ),
        )
        
        # Cache the model
        _cached_model = model
        _model_loaded = True
        
        print(f"✅ TimesFM 2.0 500M model loaded successfully")
        return model, PredictionError.SUCCESS
        
    except Exception as e:
        print(f"❌ Failed to load TimesFM 2.0 model: {e}")
        import traceback
        traceback.print_exc()
        return None, PredictionError.MODEL_LOAD_FAILED

# ==============================================================================
# Data Validation
# ==============================================================================

def _validate_input_data(data):
    """
    Validate input data for TimesFM prediction.
    
    Args:
        data: Input time series data (1D array or list)
    
    Returns:
        tuple: (validated_array, error_code)
    """
    
    # Convert to numpy array
    try:
        if isinstance(data, list):
            data_array = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            data_array = data.astype(np.float32)
        elif isinstance(data, pd.Series):
            data_array = data.values.astype(np.float32)
        else:
            return None, PredictionError.INVALID_DATA_FORMAT
    except:
        return None, PredictionError.INVALID_DATA_FORMAT
    
    # Check if 1D
    if data_array.ndim != 1:
        return None, PredictionError.INVALID_DATA_FORMAT
    
    # Check sufficient length
    if len(data_array) < CONTEXT_LENGTH:
        return None, PredictionError.INSUFFICIENT_DATA
    
    # Check for NaN or infinite values
    if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
        return None, PredictionError.CONTAINS_NAN
    
    # Check for zeros (would break normalization)
    if np.any(data_array == 0):
        return None, PredictionError.CONTAINS_ZEROS
    
    # Check if data appears normalized (should start near 1.0)
    first_value = data_array[0]
    if not (0.5 <= first_value <= 2.0):  # Reasonable range for normalized data
        return None, PredictionError.DATA_NOT_NORMALIZED
    
    return data_array, PredictionError.SUCCESS

# ==============================================================================
# Main Prediction Function
# ==============================================================================

def predict_timesfm_v2(data, verbose=False):
    """
    Generate TimesFM 2.0 predictions using the official Google TimesFM 2.0 API.
    
    Args:
        data: Input time series data (1D array, list, or pandas Series)
              Expected to be pre-normalized close prices
              Must have at least 416 data points
        verbose: If True, print detailed progress information
    
    Returns:
        tuple: (predictions, error_code)
               predictions: numpy array of predicted values (96 points) or None if error
               error_code: PredictionError code indicating success or failure
    
    Example:
        >>> # Assuming you have normalized close prices
        >>> predictions, error = predict_timesfm_v2(normalized_close_prices)
        >>> if error == PredictionError.SUCCESS:
        >>>     print(f"Predicted next 96 minutes: {predictions}")
        >>> else:
        >>>     print(f"Error: {get_error_message(error)}")
    """
    
    if verbose:
        print("🔮 Starting TimesFM 2.0 prediction...")
    
    try:
        # Validate input data
        if verbose:
            print("🔍 Validating input data...")
        data_array, error_code = _validate_input_data(data)
        if error_code != PredictionError.SUCCESS:
            if verbose:
                print(f"❌ Data validation failed: {get_error_message(error_code)}")
            return None, error_code
        
        if verbose:
            print(f"✅ Data validated: {len(data_array)} points, range [{data_array.min():.6f}, {data_array.max():.6f}]")
        
        # Load model
        if verbose:
            print("🤖 Loading model...")
        model, error_code = _load_model()
        if error_code != PredictionError.SUCCESS:
            if verbose:
                print(f"❌ Model loading failed: {get_error_message(error_code)}")
            return None, error_code
        
        # Prepare data for TimesFM 2.0
        if verbose:
            print("📊 Preparing data for TimesFM 2.0...")
        
        # Take the last CONTEXT_LENGTH points
        context_data = data_array[-CONTEXT_LENGTH:]
        
        # TimesFM 2.0 expects list of arrays (batch format)
        forecast_input = [context_data]  # Batch of 1 time series
        
        # Frequency indicator: 0 for high frequency (minute-level data)
        # 0: high frequency, long horizon (recommended for up to daily granularity)
        # 1: medium frequency (weekly, monthly)  
        # 2: low frequency, short horizon (quarterly, yearly)
        frequency_input = [0]  # High frequency for minute-level data
        
        # Generate prediction using TimesFM 2.0 API
        if verbose:
            print("🔮 Generating prediction with TimesFM 2.0...")
            print(f"   Context length: {len(context_data)} points")
            print(f"   Context range: [{context_data.min():.6f}, {context_data.max():.6f}]")
            print(f"   Context end value: {context_data[-1]:.6f}")
            print(f"   Frequency: {frequency_input[0]} (high frequency)")
        
        # Call the forecast method with normalize=True (crucial!)
        point_forecast, experimental_quantile_forecast = model.forecast(
            forecast_input,
            freq=frequency_input,
            #normalize=True,  # This is key for proper scaling!
        )
        
        # Extract predictions (first batch)
        predictions = np.array(point_forecast[0], dtype=np.float32)
        
        # Limit to our desired horizon length
        predictions = predictions[:HORIZON_LENGTH]
        
        if verbose:
            print(f"✅ Prediction generated: {len(predictions)} points")
            print(f"   Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
            print(f"   First 5: {predictions[:5]}")
            print(f"   Last 5: {predictions[-5:]}")
            print(f"   Prediction start: {predictions[0]:.6f}")
            print(f"   Continuity gap: {predictions[0] - context_data[-1]:.6f}")
        
        return predictions, PredictionError.SUCCESS
        
    except Exception as e:
        if verbose:
            print(f"❌ Unexpected error during inference: {e}")
            import traceback
            traceback.print_exc()
        return None, PredictionError.MODEL_INFERENCE_FAILED

# ==============================================================================
# Utility Functions
# ==============================================================================

def get_model_info():
    """Get information about the loaded model"""
    if _model_loaded and _cached_model is not None:
        return {
            'model_loaded': True,
            'model_type': 'TimesFM 2.0 500M',
            'context_length': CONTEXT_LENGTH,
            'horizon_length': HORIZON_LENGTH,
            'model_repo': MODEL_REPO,
            'backend': DEVICE,
            'parameters': '500M',
            'input_patch_len': INPUT_PATCH_LEN,
            'output_patch_len': OUTPUT_PATCH_LEN,
            'num_layers': NUM_LAYERS,
            'model_dims': MODEL_DIMS
        }
    else:
        return {
            'model_loaded': False,
            'model_type': 'TimesFM 2.0 500M',
            'context_length': CONTEXT_LENGTH,
            'horizon_length': HORIZON_LENGTH,
            'model_repo': MODEL_REPO,
            'parameters': '500M'
        }

def clear_model_cache():
    """Clear the cached model to free memory"""
    global _cached_model, _model_loaded
    if _cached_model is not None:
        del _cached_model
        _cached_model = None
        _model_loaded = False
        print("🧹 Model cache cleared")

# ==============================================================================
# Test Function
# ==============================================================================

def _test_with_random_csv():
    """Test the prediction function with a random CSV from datasets"""
    
    print("🧪 Testing TimesFM 2.0 prediction with random dataset...")
    
    # Find all CSV files
    csv_files = list(DATASETS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {DATASETS_DIR}")
        return
    
    # Pick random file
    random_file = random.choice(csv_files)
    print(f"📁 Selected random file: {random_file.name}")
    
    try:
        # Load data
        df = pd.read_csv(random_file)
        print(f"📊 Loaded {len(df)} rows")
        
        # Convert timestamps
        df['timestamp_pt'] = pd.to_datetime(df['timestamp_pt'])
        
        # Find 7 AM Pacific time
        df_7am = df[df['timestamp_pt'].dt.hour == 7]
        if len(df_7am) == 0:
            print("⚠️ No 7 AM data found, using first available data")
            start_idx = CONTEXT_LENGTH
        else:
            start_idx = df_7am.index[0]
        
        print(f"🕰️ Starting prediction at index {start_idx} ({df.iloc[start_idx]['timestamp_pt']})")
        
        # Get historical data (416 points before start_idx)
        if start_idx < CONTEXT_LENGTH:
            print(f"❌ Not enough historical data (need {CONTEXT_LENGTH}, have {start_idx})")
            return
        
        # Extract close prices (already normalized in the CSV)
        historical_data = df['close'].iloc[start_idx - CONTEXT_LENGTH:start_idx].values
        
        print(f"📈 Historical data: {len(historical_data)} points")
        print(f"   Range: [{historical_data.min():.6f}, {historical_data.max():.6f}]")
        print(f"   First: {historical_data[0]:.6f}, Last: {historical_data[-1]:.6f}")
        
        # Make prediction
        predictions, error_code = predict_timesfm_v2(historical_data, verbose=True)
        
        if error_code == PredictionError.SUCCESS:
            print(f"\n✅ Prediction successful!")
            print(f"📊 Predicted next {len(predictions)} minutes:")
            print(f"   Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
            print(f"   First 10: {predictions[:10]}")
            print(f"   Last 10: {predictions[-10:]}")
            
            # Compare with actual future data if available
            future_end_idx = start_idx + HORIZON_LENGTH
            actual_future = None
            if future_end_idx <= len(df):
                actual_future = df['close'].iloc[start_idx:future_end_idx].values
                print(f"\n📊 Actual future data:")
                print(f"   Range: [{actual_future.min():.6f}, {actual_future.max():.6f}]")
                print(f"   First 10: {actual_future[:10]}")
                print(f"   Last 10: {actual_future[-10:]}")
                
                # Calculate simple metrics
                mse = np.mean((predictions - actual_future) ** 2)
                mae = np.mean(np.abs(predictions - actual_future))
                print(f"\n📈 Quick evaluation:")
                print(f"   MSE: {mse:.6f}")
                print(f"   MAE: {mae:.6f}")
                
                # Check continuity
                context_end = historical_data[-1]
                pred_start = predictions[0]
                actual_start = actual_future[0]
                print(f"\n🔗 Continuity check:")
                print(f"   Context end: {context_end:.6f}")
                print(f"   Prediction start: {pred_start:.6f} (diff: {pred_start - context_end:.6f})")
                print(f"   Actual start: {actual_start:.6f} (diff: {actual_start - context_end:.6f})")
            else:
                print(f"\n⚠️ Not enough future data for comparison")
            
            # Create visualization plot
            print(f"\n🎨 Creating visualization plot...")
            try:
                plot_result = plot_prediction_results(
                    context_data=historical_data,
                    prediction_data=predictions,
                    ground_truth_data=actual_future,
                    title=f"TimesFM 2.0 Prediction - {random_file.stem}",
                    model_name="TimesFM 2.0",
                    show_plot=False,
                    verbose=False
                )
                print(f"📊 Plot saved to: {plot_result['plot_path']}")
                
                if plot_result['metrics']:
                    metrics = plot_result['metrics']
                    print(f"📈 Detailed metrics:")
                    print(f"   Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
                    print(f"   Correlation: {metrics['correlation']:.3f}")
                    print(f"   MAPE: {metrics['mape']:.2f}%")
                
            except Exception as e:
                print(f"⚠️ Failed to create plot: {e}")
        else:
            print(f"\n❌ Prediction failed: {get_error_message(error_code)}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# Main Guard
# ==============================================================================

if __name__ == "__main__":
    print("🚀 TimesFM 2.0 Prediction Interface")
    print("=" * 50)
    
    # Show model info
    info = get_model_info()
    print(f"Model Type: {info['model_type']}")
    print(f"Parameters: {info.get('parameters', 'Unknown')}")
    print(f"Context Length: {info['context_length']} minutes")
    print(f"Horizon Length: {info['horizon_length']} minutes")
    print(f"Model Repository: {info['model_repo']}")
    print("=" * 50)
    
    # Run test
    _test_with_random_csv()
