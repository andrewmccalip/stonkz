#!/usr/bin/env python3
"""
Kronos Prediction Interface - Clean inference API for Kronos model.
Handles OHLCV candlestick data with proper normalization and error handling.
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
import torch

# Add project paths
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR / "Kronos"))
sys.path.append(str(SCRIPT_DIR / "Kronos" / "examples"))

# Import Kronos components
from model import Kronos, KronosTokenizer, KronosPredictor

# Import plotting module
from plotting import plot_prediction_results

# ==============================================================================
# Configuration
# ==============================================================================

# Model Configuration
KRONOS_MODEL = "NeoQuasar/Kronos-base"
KRONOS_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
CONTEXT_LENGTH = 416    # Historical context in minutes (~6.9 hours)
HORIZON_LENGTH = 96     # Prediction horizon in minutes (~1.6 hours)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONTEXT = 512

# Dataset Configuration
DATASETS_DIR = SCRIPT_DIR / "datasets" / "ES"

# ==============================================================================
# Error Codes
# ==============================================================================

class PredictionError:
    """Error codes for Kronos prediction system"""
    
    # Success
    SUCCESS = 0                    # "Prediction completed successfully"
    
    # Input Data Errors (100-199)
    INSUFFICIENT_DATA = 101        # "Not enough historical data points (need at least 416)"
    INVALID_DATA_FORMAT = 102      # "Input data format is invalid (expected m×5 OHLCV array)"
    CONTAINS_NAN = 103            # "Input data contains NaN or infinite values"
    CONTAINS_ZEROS = 104          # "Input data contains zero values (cannot normalize)"
    DATA_NOT_NORMALIZED = 105     # "Input data appears to not be normalized (expected to start near 1.0)"
    MISSING_OHLCV_COLUMNS = 106   # "Missing required OHLCV columns (open, high, low, close, volume)"
    INVALID_OHLCV_VALUES = 107    # "Invalid OHLCV values (high < low, etc.)"
    
    # Model Errors (200-299)
    MODEL_LOAD_FAILED = 201       # "Failed to load Kronos model"
    TOKENIZER_LOAD_FAILED = 202   # "Failed to load Kronos tokenizer"
    PREDICTOR_INIT_FAILED = 203   # "Failed to initialize Kronos predictor"
    MODEL_INFERENCE_FAILED = 204  # "Model inference failed during prediction"
    DEVICE_ERROR = 205           # "Failed to move model/data to specified device"
    
    # System Errors (300-399)
    MEMORY_ERROR = 301           # "Insufficient memory for prediction"
    CUDA_ERROR = 302             # "CUDA error during processing"
    UNKNOWN_ERROR = 399          # "Unknown error occurred"

# Error code to message mapping
ERROR_MESSAGES = {
    PredictionError.SUCCESS: "Prediction completed successfully",
    PredictionError.INSUFFICIENT_DATA: "Not enough historical data points (need at least 416)",
    PredictionError.INVALID_DATA_FORMAT: "Input data format is invalid (expected m×5 OHLCV array)",
    PredictionError.CONTAINS_NAN: "Input data contains NaN or infinite values",
    PredictionError.CONTAINS_ZEROS: "Input data contains zero values (cannot normalize)",
    PredictionError.DATA_NOT_NORMALIZED: "Input data appears to not be normalized (expected to start near 1.0)",
    PredictionError.MISSING_OHLCV_COLUMNS: "Missing required OHLCV columns (open, high, low, close, volume)",
    PredictionError.INVALID_OHLCV_VALUES: "Invalid OHLCV values (high < low, etc.)",
    PredictionError.MODEL_LOAD_FAILED: "Failed to load Kronos model",
    PredictionError.TOKENIZER_LOAD_FAILED: "Failed to load Kronos tokenizer",
    PredictionError.PREDICTOR_INIT_FAILED: "Failed to initialize Kronos predictor",
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

_cached_predictor = None
_model_loaded = False

def _load_model():
    """Load and cache the Kronos model"""
    global _cached_predictor, _model_loaded
    
    if _model_loaded and _cached_predictor is not None:
        return _cached_predictor, PredictionError.SUCCESS
    
    try:
        print(f"🤖 Loading Kronos model on {DEVICE}...")
        
        # Load model
        try:
            model = Kronos.from_pretrained(KRONOS_MODEL)
        except Exception as e:
            print(f"❌ Failed to load Kronos model: {e}")
            return None, PredictionError.MODEL_LOAD_FAILED
        
        # Load tokenizer
        try:
            tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER)
        except Exception as e:
            print(f"❌ Failed to load Kronos tokenizer: {e}")
            return None, PredictionError.TOKENIZER_LOAD_FAILED
        
        # Initialize predictor
        try:
            predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)
        except Exception as e:
            print(f"❌ Failed to initialize Kronos predictor: {e}")
            return None, PredictionError.PREDICTOR_INIT_FAILED
        
        # Cache the predictor
        _cached_predictor = predictor
        _model_loaded = True
        
        print(f"✅ Kronos model loaded successfully on {DEVICE}")
        return predictor, PredictionError.SUCCESS
        
    except torch.cuda.OutOfMemoryError:
        return None, PredictionError.CUDA_ERROR
    except MemoryError:
        return None, PredictionError.MEMORY_ERROR
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, PredictionError.MODEL_LOAD_FAILED

# ==============================================================================
# Data Validation
# ==============================================================================

def _validate_input_data(data):
    """
    Validate input OHLCV data for Kronos prediction.
    
    Args:
        data: Input OHLCV data (2D array, DataFrame, or list of lists)
              Expected format: m×5 with columns [open, high, low, close, volume]
    
    Returns:
        tuple: (validated_dataframe, error_code)
    """
    
    # Convert to DataFrame
    try:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] != 5:
                return None, PredictionError.INVALID_DATA_FORMAT
            df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
        elif isinstance(data, list):
            # Assume list of lists or list of tuples
            try:
                arr = np.array(data)
                if arr.ndim != 2 or arr.shape[1] != 5:
                    return None, PredictionError.INVALID_DATA_FORMAT
                df = pd.DataFrame(arr, columns=['open', 'high', 'low', 'close', 'volume'])
            except:
                return None, PredictionError.INVALID_DATA_FORMAT
        else:
            return None, PredictionError.INVALID_DATA_FORMAT
    except:
        return None, PredictionError.INVALID_DATA_FORMAT
    
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return None, PredictionError.MISSING_OHLCV_COLUMNS
    
    # Check sufficient length
    if len(df) < CONTEXT_LENGTH:
        return None, PredictionError.INSUFFICIENT_DATA
    
    # Convert to float and check for NaN/inf
    try:
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if df[required_cols].isnull().any().any():
            return None, PredictionError.CONTAINS_NAN
        
        if np.isinf(df[required_cols].values).any():
            return None, PredictionError.CONTAINS_NAN
    except:
        return None, PredictionError.CONTAINS_NAN
    
    # Check for zeros (would break normalization)
    price_cols = ['open', 'high', 'low', 'close']
    if (df[price_cols] == 0).any().any():
        return None, PredictionError.CONTAINS_ZEROS
    
    # Check OHLCV validity (high >= low, etc.)
    invalid_rows = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close']) |
        (df['volume'] < 0)
    )
    
    if invalid_rows.any():
        return None, PredictionError.INVALID_OHLCV_VALUES
    
    # Check if data appears normalized (first close should be near 1.0)
    first_close = df['close'].iloc[0]
    if not (0.5 <= first_close <= 2.0):  # Reasonable range for normalized data
        return None, PredictionError.DATA_NOT_NORMALIZED
    
    return df, PredictionError.SUCCESS

def _prepare_kronos_input(df):
    """
    Prepare OHLCV data for Kronos model input.
    
    Args:
        df: Validated OHLCV DataFrame
    
    Returns:
        tuple: (ohlcv_data, timestamps, error_code)
    """
    
    try:
        # Take the last CONTEXT_LENGTH points
        context_df = df.tail(CONTEXT_LENGTH).copy()
        
        # Prepare OHLCV data for Kronos (same format as backtest)
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        ohlcv_data = context_df[ohlcv_columns].copy()
        
        # Add amount column (close * volume)
        ohlcv_data['amount'] = ohlcv_data['close'] * ohlcv_data['volume']
        
        # Create timestamps (Kronos expects pandas Series)
        # Since we don't have actual timestamps, create a sequence
        timestamps = pd.Series(pd.date_range(
            start='2024-01-01 07:00:00',
            periods=len(ohlcv_data),
            freq='1min'
        ))
        
        return ohlcv_data, timestamps, PredictionError.SUCCESS
        
    except Exception as e:
        print(f"❌ Error preparing Kronos input: {e}")
        return None, None, PredictionError.DEVICE_ERROR

# ==============================================================================
# Main Prediction Function
# ==============================================================================

def predict_kronos(data, verbose=False):
    """
    Generate Kronos predictions for OHLCV candlestick data.
    
    Args:
        data: Input OHLCV data (2D array, DataFrame, or list of lists)
              Expected format: m×5 with columns [open, high, low, close, volume]
              Expected to be pre-normalized (prices should start near 1.0)
              Must have at least 416 data points
        verbose: If True, print detailed progress information
    
    Returns:
        tuple: (predictions, error_code)
               predictions: numpy array of predicted close values (96 points) or None if error
               error_code: PredictionError code indicating success or failure
    
    Example:
        >>> # Assuming you have normalized OHLCV data
        >>> ohlcv_data = np.array([[1.0, 1.002, 0.998, 1.001, 1000], ...])  # m×5 array
        >>> predictions, error = predict_kronos(ohlcv_data)
        >>> if error == PredictionError.SUCCESS:
        >>>     print(f"Predicted next 96 minutes: {predictions}")
        >>> else:
        >>>     print(f"Error: {get_error_message(error)}")
    """
    
    if verbose:
        print("🔮 Starting Kronos prediction...")
    
    try:
        # Validate input data
        if verbose:
            print("🔍 Validating input data...")
        df, error_code = _validate_input_data(data)
        if error_code != PredictionError.SUCCESS:
            if verbose:
                print(f"❌ Data validation failed: {get_error_message(error_code)}")
            return None, error_code
        
        if verbose:
            print(f"✅ Data validated: {len(df)} rows")
            print(f"   Close range: [{df['close'].min():.6f}, {df['close'].max():.6f}]")
            print(f"   Volume range: [{df['volume'].min():.0f}, {df['volume'].max():.0f}]")
        
        # Load model
        if verbose:
            print("🤖 Loading model...")
        predictor, error_code = _load_model()
        if error_code != PredictionError.SUCCESS:
            if verbose:
                print(f"❌ Model loading failed: {get_error_message(error_code)}")
            return None, error_code
        
        # Prepare model input
        if verbose:
            print("📊 Preparing model input...")
        ohlcv_data, timestamps, error_code = _prepare_kronos_input(df)
        if error_code != PredictionError.SUCCESS:
            if verbose:
                print(f"❌ Input preparation failed: {get_error_message(error_code)}")
            return None, error_code
        
        # Generate prediction
        if verbose:
            print("🔮 Generating prediction...")
        
        # Create prediction timestamps
        pred_timestamps = []
        pred_start_time = timestamps.iloc[-1] + timedelta(minutes=1)
        for i in range(HORIZON_LENGTH):
            pred_timestamps.append(pred_start_time + timedelta(minutes=i))
        
        # Call Kronos predictor (same as backtest)
        prediction_df = predictor.predict(
            df=ohlcv_data,
            x_timestamp=timestamps,
            y_timestamp=pd.Series(pred_timestamps),
            pred_len=HORIZON_LENGTH,
            T=1.0,
            top_p=0.9,
            sample_count=1,
        )
        
        # Extract predictions
        prediction_values = prediction_df.iloc[:, 0].values  # First column contains predictions
        
        if verbose:
            print(f"✅ Prediction generated: {len(prediction_values)} points")
            print(f"   Range: [{prediction_values.min():.6f}, {prediction_values.max():.6f}]")
            print(f"   First 5: {prediction_values[:5]}")
            print(f"   Last 5: {prediction_values[-5:]}")
        
        return prediction_values, PredictionError.SUCCESS
        
    except torch.cuda.OutOfMemoryError:
        if verbose:
            print("❌ CUDA out of memory error")
        return None, PredictionError.CUDA_ERROR
    except MemoryError:
        if verbose:
            print("❌ System out of memory error")
        return None, PredictionError.MEMORY_ERROR
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
    if _model_loaded and _cached_predictor is not None:
        return {
            'model_loaded': True,
            'device': DEVICE,
            'context_length': CONTEXT_LENGTH,
            'horizon_length': HORIZON_LENGTH,
            'max_context': MAX_CONTEXT,
            'model_name': KRONOS_MODEL,
            'tokenizer_name': KRONOS_TOKENIZER
        }
    else:
        return {
            'model_loaded': False,
            'device': DEVICE,
            'context_length': CONTEXT_LENGTH,
            'horizon_length': HORIZON_LENGTH,
            'max_context': MAX_CONTEXT,
            'model_name': KRONOS_MODEL,
            'tokenizer_name': KRONOS_TOKENIZER
        }

def clear_model_cache():
    """Clear the cached model to free memory"""
    global _cached_predictor, _model_loaded
    if _cached_predictor is not None:
        del _cached_predictor
        _cached_predictor = None
        _model_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 Model cache cleared")

# ==============================================================================
# Test Function
# ==============================================================================

def _test_with_random_csv():
    """Test the prediction function with a random CSV from datasets"""
    
    print("🧪 Testing Kronos prediction with random dataset...")
    
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
            start_idx = 0
        else:
            start_idx = df_7am.index[0]
        
        print(f"🕰️ Starting prediction at index {start_idx} ({df.iloc[start_idx]['timestamp_pt']})")
        
        # Get historical data (416 points before start_idx)
        if start_idx < CONTEXT_LENGTH:
            print(f"❌ Not enough historical data (need {CONTEXT_LENGTH}, have {start_idx})")
            return
        
        # Extract OHLCV data (already normalized in the CSV)
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        historical_data = df[ohlcv_columns].iloc[start_idx - CONTEXT_LENGTH:start_idx]
        
        print(f"📈 Historical OHLCV data: {len(historical_data)} rows")
        print(f"   Close range: [{historical_data['close'].min():.6f}, {historical_data['close'].max():.6f}]")
        print(f"   Volume range: [{historical_data['volume'].min():.0f}, {historical_data['volume'].max():.0f}]")
        print(f"   First close: {historical_data['close'].iloc[0]:.6f}, Last close: {historical_data['close'].iloc[-1]:.6f}")
        
        # Make prediction
        predictions, error_code = predict_kronos(historical_data, verbose=True)
        
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
                
                # Directional accuracy
                pred_direction = np.sign(np.diff(predictions))
                actual_direction = np.sign(np.diff(actual_future))
                dir_accuracy = np.mean(pred_direction == actual_direction) * 100
                print(f"   Directional accuracy: {dir_accuracy:.1f}%")
            else:
                print(f"\n⚠️ Not enough future data for comparison")
            
            # Create visualization plot
            print(f"\n🎨 Creating visualization plot...")
            try:
                # Extract volume data for plotting
                volume_data = historical_data['volume'].values if 'volume' in historical_data.columns else None
                
                plot_result = plot_prediction_results(
                    context_data=historical_data,  # Pass full OHLCV DataFrame
                    prediction_data=predictions,
                    ground_truth_data=actual_future,
                    volume_data=volume_data,
                    title=f"Kronos Prediction - {random_file.stem}",
                    model_name="Kronos",
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
                import traceback
                traceback.print_exc()
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
    print("🚀 Kronos Prediction Interface")
    print("=" * 50)
    
    # Show model info
    info = get_model_info()
    print(f"Device: {info['device']}")
    print(f"Context Length: {info['context_length']} minutes")
    print(f"Horizon Length: {info['horizon_length']} minutes")
    print(f"Model: {info['model_name']}")
    print(f"Tokenizer: {info['tokenizer_name']}")
    print("=" * 50)
    
    # Run test
    _test_with_random_csv()
