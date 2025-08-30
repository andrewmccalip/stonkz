#!/usr/bin/env python3
"""
FFT-Only Prediction Interface - Standalone FFT similarity-based predictions

This module provides a standalone FFT prediction system that integrates seamlessly
with the existing backtest framework. It uses our data-driven frequency bands
and similarity search engine to generate predictions based on historical patterns.

Compatible with backtest_unified.py prediction interface:
- predict_fft(data, verbose=False) -> (predictions, error_code)
- get_model_info() -> model_information_dict

Features:
1. Data-driven frequency band analysis
2. Cosine similarity pattern matching  
3. Adaptive market regime detection
4. Multiple outcome aggregation strategies
5. Comprehensive error handling
6. Performance optimization with caching
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Add project paths
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

# Import our FFT components
from fft_similarity_engine import load_similarity_engine, FFTSimilarityEngine, predict_fft_similarity as engine_predict
from fft_signature_loader import FREQ_BANDS, BAND_WEIGHTS

# Import plotting module for visualization
try:
    from plotting import plot_prediction_results
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️ Plotting module not available")

# ==============================================================================
# Configuration
# ==============================================================================

# Model Configuration
CONTEXT_LENGTH = 416    # Historical context in minutes (~6.9 hours)
HORIZON_LENGTH = 96     # Prediction horizon in minutes (~1.6 hours)

# FFT Similarity Engine Parameters
K_NEIGHBORS = 20               # Number of similar patterns to use
MIN_SIMILARITY_THRESHOLD = 0.25  # Minimum similarity for pattern inclusion
AGGREGATION_METHOD = 'weighted_average'  # How to combine similar outcomes
DATABASE_SIZE = 10000          # Number of signatures to load (balance speed vs accuracy)

# Performance settings
ENABLE_CACHING = True          # Use cached database for faster loading
ADAPTIVE_WEIGHTING = True      # Use market regime detection
EXCLUDE_SAME_SOURCE = True     # Exclude patterns from same source file

# Dataset Configuration
DATASETS_DIR = SCRIPT_DIR / "datasets" / "ES"

# ==============================================================================
# Error Codes (compatible with existing framework)
# ==============================================================================

class PredictionError:
    """Error codes for FFT prediction system"""
    
    # Success
    SUCCESS = 0                    # "Prediction completed successfully"
    
    # Input Data Errors (100-199)
    INSUFFICIENT_DATA = 101        # "Not enough historical data points (need at least 416)"
    INVALID_DATA_FORMAT = 102      # "Input data format is invalid (expected 1D array or list)"
    CONTAINS_NAN = 103            # "Input data contains NaN or infinite values"
    CONTAINS_ZEROS = 104          # "Input data contains zero values (cannot normalize)"
    DATA_NOT_NORMALIZED = 105     # "Input data appears to not be normalized (expected to start near 1.0)"
    
    # FFT Engine Errors (200-299)
    ENGINE_LOAD_FAILED = 201      # "Failed to load FFT similarity engine"
    DATABASE_LOAD_FAILED = 202    # "Failed to load FFT signature database"
    SIMILARITY_SEARCH_FAILED = 203 # "Similarity search failed"
    NO_SIMILAR_PATTERNS = 204     # "No similar patterns found above threshold"
    AGGREGATION_FAILED = 205      # "Failed to aggregate similar outcomes"
    
    # System Errors (300-399)
    MEMORY_ERROR = 301           # "Insufficient memory for FFT analysis"
    UNKNOWN_ERROR = 399          # "Unknown error occurred"

# Error code to message mapping
ERROR_MESSAGES = {
    PredictionError.SUCCESS: "Prediction completed successfully",
    PredictionError.INSUFFICIENT_DATA: "Not enough historical data points (need at least 416)",
    PredictionError.INVALID_DATA_FORMAT: "Input data format is invalid (expected 1D array or list)",
    PredictionError.CONTAINS_NAN: "Input data contains NaN or infinite values",
    PredictionError.CONTAINS_ZEROS: "Input data contains zero values (cannot normalize)",
    PredictionError.DATA_NOT_NORMALIZED: "Input data appears to not be normalized (expected to start near 1.0)",
    PredictionError.ENGINE_LOAD_FAILED: "Failed to load FFT similarity engine",
    PredictionError.DATABASE_LOAD_FAILED: "Failed to load FFT signature database",
    PredictionError.SIMILARITY_SEARCH_FAILED: "Similarity search failed",
    PredictionError.NO_SIMILAR_PATTERNS: "No similar patterns found above threshold",
    PredictionError.AGGREGATION_FAILED: "Failed to aggregate similar outcomes",
    PredictionError.MEMORY_ERROR: "Insufficient memory for FFT analysis",
    PredictionError.UNKNOWN_ERROR: "Unknown error occurred"
}

def get_error_message(error_code):
    """Get human-readable error message for error code"""
    return ERROR_MESSAGES.get(error_code, f"Unknown error code: {error_code}")

# ==============================================================================
# Engine Cache
# ==============================================================================

_cached_engine = None
_engine_loaded = False

def _load_engine():
    """Load and cache the FFT similarity engine"""
    global _cached_engine, _engine_loaded
    
    if _engine_loaded and _cached_engine is not None:
        return _cached_engine, PredictionError.SUCCESS
    
    try:
        print(f"🔮 Loading FFT Similarity Engine...")
        print(f"   Database size: {DATABASE_SIZE:,} signatures")
        print(f"   K-neighbors: {K_NEIGHBORS}")
        print(f"   Min similarity: {MIN_SIMILARITY_THRESHOLD}")
        print(f"   Aggregation: {AGGREGATION_METHOD}")
        
        # Load engine with optimized parameters
        engine = load_similarity_engine(
            database_strategy='full',
            max_signatures=DATABASE_SIZE,
            k_neighbors=K_NEIGHBORS,
            min_similarity=MIN_SIMILARITY_THRESHOLD,
            aggregation_method=AGGREGATION_METHOD
        )
        
        # Cache the engine
        _cached_engine = engine
        _engine_loaded = True
        
        print(f"✅ FFT Similarity Engine loaded successfully")
        print(f"   Loaded signatures: {len(engine.database.signatures):,}")
        print(f"   Memory usage: {engine.database.get_database_stats()['memory_usage_mb']:.1f} MB")
        print(f"   Data-driven bands: {len(FREQ_BANDS)}")
        
        return engine, PredictionError.SUCCESS
        
    except Exception as e:
        print(f"❌ Failed to load FFT Similarity Engine: {e}")
        import traceback
        traceback.print_exc()
        return None, PredictionError.ENGINE_LOAD_FAILED

# ==============================================================================
# Data Validation (reused from TimesFM)
# ==============================================================================

def _validate_input_data(data):
    """
    Validate input data for FFT prediction.
    
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

def predict_fft(data, verbose=False):
    """
    Generate FFT similarity-based predictions.
    
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
        >>> predictions, error = predict_fft(normalized_close_prices)
        >>> if error == PredictionError.SUCCESS:
        >>>     print(f"Predicted next 96 minutes: {predictions}")
        >>> else:
        >>>     print(f"Error: {get_error_message(error)}")
    """
    
    if verbose:
        print("🔮 Starting FFT similarity-based prediction...")
    
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
        
        # Load FFT similarity engine
        if verbose:
            print("🔮 Loading FFT similarity engine...")
        engine, error_code = _load_engine()
        if error_code != PredictionError.SUCCESS:
            if verbose:
                print(f"❌ Engine loading failed: {get_error_message(error_code)}")
            return None, error_code
        
        # Prepare context data
        if verbose:
            print("📊 Preparing context data...")
        
        # Take the last CONTEXT_LENGTH points
        context_data = data_array[-CONTEXT_LENGTH:]
        
        if verbose:
            print(f"   Context length: {len(context_data)} points")
            print(f"   Context range: [{context_data.min():.6f}, {context_data.max():.6f}]")
            print(f"   Context end value: {context_data[-1]:.6f}")
        
        # Generate prediction using FFT similarity engine
        if verbose:
            print("🔍 Finding similar patterns and generating prediction...")
        
        try:
            prediction, confidence, explanation = engine.predict(
                context_data,
                adaptive=ADAPTIVE_WEIGHTING,
                verbose=verbose
            )
            
            if 'error' in explanation:
                if verbose:
                    print(f"❌ Similarity search failed: {explanation['error']}")
                return None, PredictionError.NO_SIMILAR_PATTERNS
            
            if verbose:
                print(f"✅ Prediction generated successfully!")
                print(f"   Similar patterns found: {explanation['num_similar_patterns']}")
                print(f"   Best similarity: {max(explanation['similarity_scores']):.4f}")
                print(f"   Average similarity: {np.mean(explanation['similarity_scores']):.4f}")
                print(f"   Market regime: {explanation['market_regime']}")
                print(f"   Prediction confidence: {confidence:.4f}")
                print(f"   Prediction range: [{prediction.min():.6f}, {prediction.max():.6f}]")
                print(f"   First 5 predictions: {prediction[:5]}")
                print(f"   Last 5 predictions: {prediction[-5:]}")
                
                # Show band contributions
                print(f"   Band similarities:")
                for band_name in FREQ_BANDS.keys():
                    band_sims = explanation['band_similarities'][band_name]
                    avg_band_sim = np.mean(band_sims) if band_sims else 0
                    print(f"     {band_name}: {avg_band_sim:.4f} (weight: {BAND_WEIGHTS[band_name]:.2f})")
            
            return prediction, PredictionError.SUCCESS
            
        except Exception as e:
            if verbose:
                print(f"❌ Similarity search error: {e}")
            return None, PredictionError.SIMILARITY_SEARCH_FAILED
        
    except Exception as e:
        if verbose:
            print(f"❌ Unexpected error during FFT prediction: {e}")
            import traceback
            traceback.print_exc()
        return None, PredictionError.UNKNOWN_ERROR

# ==============================================================================
# Utility Functions
# ==============================================================================

def get_model_info():
    """Get information about the FFT prediction model"""
    if _engine_loaded and _cached_engine is not None:
        engine_stats = _cached_engine.get_engine_stats()
        return {
            'model_loaded': True,
            'model_type': 'FFT Similarity Matching',
            'method': 'Data-Driven Frequency Analysis + Cosine Similarity',
            'context_length': CONTEXT_LENGTH,
            'horizon_length': HORIZON_LENGTH,
            'database_size': engine_stats['database_stats']['total_signatures'],
            'k_neighbors': K_NEIGHBORS,
            'min_similarity': MIN_SIMILARITY_THRESHOLD,
            'aggregation_method': AGGREGATION_METHOD,
            'frequency_bands': len(FREQ_BANDS),
            'data_driven_bands': True,
            'adaptive_weighting': ADAPTIVE_WEIGHTING,
            'memory_usage_mb': engine_stats['database_stats']['memory_usage_mb'],
            'predictions_generated': engine_stats['usage_stats']['predictions_generated'],
            'frequency_band_details': {
                band: {'range': freq_range, 'weight': BAND_WEIGHTS[band]}
                for band, freq_range in FREQ_BANDS.items()
            }
        }
    else:
        return {
            'model_loaded': False,
            'model_type': 'FFT Similarity Matching',
            'method': 'Data-Driven Frequency Analysis + Cosine Similarity',
            'context_length': CONTEXT_LENGTH,
            'horizon_length': HORIZON_LENGTH,
            'database_size': DATABASE_SIZE,
            'k_neighbors': K_NEIGHBORS,
            'frequency_bands': len(FREQ_BANDS),
            'data_driven_bands': True
        }

def clear_model_cache():
    """Clear the cached FFT engine to free memory"""
    global _cached_engine, _engine_loaded
    if _cached_engine is not None:
        del _cached_engine
        _cached_engine = None
        _engine_loaded = False
        print("🧹 FFT engine cache cleared")

# ==============================================================================
# Test Function
# ==============================================================================

def _test_with_random_csv():
    """Test the FFT prediction function with a random CSV from datasets"""
    
    print("🧪 Testing FFT similarity prediction with random dataset...")
    
    # Find all CSV files
    csv_files = list(DATASETS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {DATASETS_DIR}")
        return
    
    # Pick random file
    import random
    random_file = random.choice(csv_files)
    print(f"📁 Selected random file: {random_file.name}")
    
    try:
        # Load data
        df = pd.read_csv(random_file)
        print(f"📊 Loaded {len(df)} rows")
        
        # Convert timestamps
        df['timestamp_pt'] = pd.to_datetime(df['timestamp_pt'])
        
        # Find a good starting point (prefer market hours)
        market_hour_mask = df['timestamp_pt'].dt.hour.between(9, 16)
        if market_hour_mask.any():
            market_indices = df[market_hour_mask].index
            start_idx = random.choice(market_indices[CONTEXT_LENGTH:]) if len(market_indices) > CONTEXT_LENGTH else CONTEXT_LENGTH
        else:
            start_idx = CONTEXT_LENGTH
        
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
        predictions, error_code = predict_fft(historical_data, verbose=True)
        
        if error_code == PredictionError.SUCCESS:
            print(f"\n✅ FFT prediction successful!")
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
                
                # Calculate evaluation metrics
                mse = np.mean((predictions - actual_future) ** 2)
                mae = np.mean(np.abs(predictions - actual_future))
                
                # Directional accuracy
                pred_returns = np.diff(predictions)
                actual_returns = np.diff(actual_future)
                pred_directions = np.sign(pred_returns)
                actual_directions = np.sign(actual_returns)
                correct_directions = pred_directions == actual_directions
                directional_accuracy = np.mean(correct_directions) * 100
                
                # Correlation
                correlation = np.corrcoef(predictions, actual_future)[0, 1]
                
                print(f"\n📈 Evaluation metrics:")
                print(f"   MSE: {mse:.6f}")
                print(f"   MAE: {mae:.6f}")
                print(f"   Directional Accuracy: {directional_accuracy:.1f}%")
                print(f"   Correlation: {correlation:.3f}")
                
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
            if HAS_PLOTTING:
                print(f"\n🎨 Creating visualization plot...")
                try:
                    plot_result = plot_prediction_results(
                        context_data=historical_data,
                        prediction_data=predictions,
                        ground_truth_data=actual_future,
                        title=f"FFT Similarity Prediction - {random_file.stem}",
                        model_name="FFT Similarity",
                        show_plot=False,
                        verbose=False
                    )
                    print(f"📊 Plot saved to: {plot_result['plot_path']}")
                    
                    if plot_result['metrics']:
                        metrics = plot_result['metrics']
                        print(f"📈 Plot metrics:")
                        print(f"   Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
                        print(f"   Correlation: {metrics['correlation']:.3f}")
                        print(f"   MAPE: {metrics['mape']:.2f}%")
                    
                except Exception as e:
                    print(f"⚠️ Failed to create plot: {e}")
        else:
            print(f"\n❌ FFT prediction failed: {get_error_message(error_code)}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# Main Guard
# ==============================================================================

if __name__ == "__main__":
    print("🔮 FFT Similarity Prediction Interface")
    print("=" * 50)
    
    # Show model info
    info = get_model_info()
    print(f"Model Type: {info['model_type']}")
    print(f"Method: {info['method']}")
    print(f"Context Length: {info['context_length']} minutes")
    print(f"Horizon Length: {info['horizon_length']} minutes")
    print(f"Database Size: {info['database_size']}")
    print(f"K-Neighbors: {info['k_neighbors']}")
    print(f"Data-Driven Bands: {info['data_driven_bands']}")
    print(f"Frequency Bands: {info['frequency_bands']}")
    print("=" * 50)
    
    # Run test
    _test_with_random_csv()
