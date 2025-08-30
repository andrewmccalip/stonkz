#!/usr/bin/env python3
"""
FFT+TimesFM Hybrid Prediction Model - Ultimate ensemble prediction system

This module combines FFT similarity-based predictions with TimesFM predictions
to create a superior hybrid model that leverages both:

1. FFT Pattern Matching: Historical similarity in frequency domain
2. TimesFM Deep Learning: Advanced transformer-based time series modeling

The hybrid system uses intelligent ensemble strategies:
- Static weighting (configurable α parameter)
- Dynamic weighting based on prediction confidence
- Adaptive weighting based on market conditions
- Regime-aware ensemble selection

Compatible with backtest_unified.py framework.
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

# Import prediction models
from prediction_fft import predict_fft, get_model_info as get_fft_info, PredictionError as FFTError
from prediction_timesfm_v2 import predict_timesfm_v2, get_model_info as get_timesfm_info, PredictionError as TimesFMError

# Import FFT components for advanced ensemble logic
from fft_similarity_engine import load_similarity_engine
from fft_signature_loader import FREQ_BANDS, BAND_WEIGHTS

# Import plotting module
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

# Ensemble Parameters
DEFAULT_ALPHA = 0.6     # Default weight for TimesFM (0.6) vs FFT (0.4)
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for dynamic weighting

# Ensemble strategies
ENSEMBLE_STRATEGIES = {
    'static': 'Fixed weighting between FFT and TimesFM',
    'confidence': 'Weight by individual model confidence scores',
    'adaptive': 'Adapt weights based on market regime and confidence',
    'selective': 'Select best model based on conditions',
    'average': 'Simple 50/50 average'
}

DEFAULT_ENSEMBLE_STRATEGY = 'adaptive'

# Market regime ensemble weights
REGIME_WEIGHTS = {
    'trending': {'timesfm': 0.7, 'fft': 0.3},    # TimesFM better for trends
    'volatile': {'timesfm': 0.5, 'fft': 0.5},    # Balanced for volatility
    'quiet': {'timesfm': 0.4, 'fft': 0.6},       # FFT better for quiet markets
    'normal': {'timesfm': 0.6, 'fft': 0.4}       # Default weighting
}

# Dataset Configuration
DATASETS_DIR = SCRIPT_DIR / "datasets" / "ES"

# ==============================================================================
# Error Codes
# ==============================================================================

class HybridPredictionError:
    """Error codes for hybrid prediction system"""
    
    # Success
    SUCCESS = 0                    # "Prediction completed successfully"
    
    # Input Data Errors (100-199)
    INSUFFICIENT_DATA = 101        # "Not enough historical data points"
    INVALID_DATA_FORMAT = 102      # "Input data format is invalid"
    CONTAINS_NAN = 103            # "Input data contains NaN or infinite values"
    CONTAINS_ZEROS = 104          # "Input data contains zero values"
    DATA_NOT_NORMALIZED = 105     # "Input data not normalized"
    
    # Model Errors (200-299)
    TIMESFM_FAILED = 201          # "TimesFM prediction failed"
    FFT_FAILED = 202              # "FFT prediction failed"
    BOTH_MODELS_FAILED = 203      # "Both TimesFM and FFT predictions failed"
    ENSEMBLE_FAILED = 204         # "Ensemble combination failed"
    
    # System Errors (300-399)
    MEMORY_ERROR = 301           # "Insufficient memory"
    UNKNOWN_ERROR = 399          # "Unknown error occurred"

# Error messages
HYBRID_ERROR_MESSAGES = {
    HybridPredictionError.SUCCESS: "Prediction completed successfully",
    HybridPredictionError.INSUFFICIENT_DATA: "Not enough historical data points (need at least 416)",
    HybridPredictionError.INVALID_DATA_FORMAT: "Input data format is invalid (expected 1D array or list)",
    HybridPredictionError.CONTAINS_NAN: "Input data contains NaN or infinite values",
    HybridPredictionError.CONTAINS_ZEROS: "Input data contains zero values (cannot normalize)",
    HybridPredictionError.DATA_NOT_NORMALIZED: "Input data appears to not be normalized",
    HybridPredictionError.TIMESFM_FAILED: "TimesFM prediction failed",
    HybridPredictionError.FFT_FAILED: "FFT prediction failed",
    HybridPredictionError.BOTH_MODELS_FAILED: "Both TimesFM and FFT predictions failed",
    HybridPredictionError.ENSEMBLE_FAILED: "Ensemble combination failed",
    HybridPredictionError.MEMORY_ERROR: "Insufficient memory for prediction",
    HybridPredictionError.UNKNOWN_ERROR: "Unknown error occurred"
}

def get_hybrid_error_message(error_code):
    """Get human-readable error message for hybrid error code"""
    return HYBRID_ERROR_MESSAGES.get(error_code, f"Unknown error code: {error_code}")

# ==============================================================================
# Ensemble Functions
# ==============================================================================

def ensemble_static(timesfm_pred: np.ndarray, fft_pred: np.ndarray, 
                   alpha: float = DEFAULT_ALPHA) -> Tuple[np.ndarray, Dict]:
    """
    Static ensemble with fixed weighting.
    
    Args:
        timesfm_pred: TimesFM predictions
        fft_pred: FFT predictions  
        alpha: Weight for TimesFM (1-alpha for FFT)
    
    Returns:
        Tuple of (ensemble_prediction, ensemble_info)
    """
    ensemble_pred = alpha * timesfm_pred + (1 - alpha) * fft_pred
    
    ensemble_info = {
        'strategy': 'static',
        'timesfm_weight': alpha,
        'fft_weight': 1 - alpha,
        'alpha': alpha
    }
    
    return ensemble_pred, ensemble_info

def ensemble_confidence(timesfm_pred: np.ndarray, fft_pred: np.ndarray,
                       timesfm_confidence: float, fft_confidence: float) -> Tuple[np.ndarray, Dict]:
    """
    Confidence-based ensemble weighting.
    
    Args:
        timesfm_pred: TimesFM predictions
        fft_pred: FFT predictions
        timesfm_confidence: TimesFM confidence score
        fft_confidence: FFT confidence score
    
    Returns:
        Tuple of (ensemble_prediction, ensemble_info)
    """
    total_confidence = timesfm_confidence + fft_confidence
    
    if total_confidence > 0:
        timesfm_weight = timesfm_confidence / total_confidence
        fft_weight = fft_confidence / total_confidence
    else:
        # Fallback to equal weighting
        timesfm_weight = 0.5
        fft_weight = 0.5
    
    ensemble_pred = timesfm_weight * timesfm_pred + fft_weight * fft_pred
    
    ensemble_info = {
        'strategy': 'confidence',
        'timesfm_weight': timesfm_weight,
        'fft_weight': fft_weight,
        'timesfm_confidence': timesfm_confidence,
        'fft_confidence': fft_confidence,
        'total_confidence': total_confidence
    }
    
    return ensemble_pred, ensemble_info

def ensemble_adaptive(timesfm_pred: np.ndarray, fft_pred: np.ndarray,
                     market_regime: str, timesfm_confidence: float, fft_confidence: float) -> Tuple[np.ndarray, Dict]:
    """
    Adaptive ensemble based on market regime and confidence.
    
    Args:
        timesfm_pred: TimesFM predictions
        fft_pred: FFT predictions
        market_regime: Detected market regime
        timesfm_confidence: TimesFM confidence score
        fft_confidence: FFT confidence score
    
    Returns:
        Tuple of (ensemble_prediction, ensemble_info)
    """
    # Get regime-based base weights
    regime_weights = REGIME_WEIGHTS.get(market_regime, REGIME_WEIGHTS['normal'])
    base_timesfm_weight = regime_weights['timesfm']
    base_fft_weight = regime_weights['fft']
    
    # Adjust based on confidence if both models have reasonable confidence
    if timesfm_confidence > CONFIDENCE_THRESHOLD and fft_confidence > CONFIDENCE_THRESHOLD:
        # Blend regime weights with confidence weights
        conf_total = timesfm_confidence + fft_confidence
        conf_timesfm_weight = timesfm_confidence / conf_total
        conf_fft_weight = fft_confidence / conf_total
        
        # Weighted combination of regime and confidence
        regime_factor = 0.7  # How much to trust regime vs confidence
        timesfm_weight = regime_factor * base_timesfm_weight + (1 - regime_factor) * conf_timesfm_weight
        fft_weight = regime_factor * base_fft_weight + (1 - regime_factor) * conf_fft_weight
        
    elif timesfm_confidence > CONFIDENCE_THRESHOLD:
        # Only TimesFM has good confidence, increase its weight
        timesfm_weight = min(0.8, base_timesfm_weight + 0.2)
        fft_weight = 1 - timesfm_weight
        
    elif fft_confidence > CONFIDENCE_THRESHOLD:
        # Only FFT has good confidence, increase its weight  
        fft_weight = min(0.8, base_fft_weight + 0.2)
        timesfm_weight = 1 - fft_weight
        
    else:
        # Low confidence from both, use regime weights
        timesfm_weight = base_timesfm_weight
        fft_weight = base_fft_weight
    
    ensemble_pred = timesfm_weight * timesfm_pred + fft_weight * fft_pred
    
    ensemble_info = {
        'strategy': 'adaptive',
        'market_regime': market_regime,
        'timesfm_weight': timesfm_weight,
        'fft_weight': fft_weight,
        'base_regime_weights': regime_weights,
        'timesfm_confidence': timesfm_confidence,
        'fft_confidence': fft_confidence,
        'confidence_adjustment': abs(timesfm_confidence - fft_confidence) > CONFIDENCE_THRESHOLD
    }
    
    return ensemble_pred, ensemble_info

def ensemble_selective(timesfm_pred: np.ndarray, fft_pred: np.ndarray,
                      market_regime: str, timesfm_confidence: float, fft_confidence: float) -> Tuple[np.ndarray, Dict]:
    """
    Selective ensemble - choose the best model for current conditions.
    
    Args:
        timesfm_pred: TimesFM predictions
        fft_pred: FFT predictions
        market_regime: Detected market regime
        timesfm_confidence: TimesFM confidence score
        fft_confidence: FFT confidence score
    
    Returns:
        Tuple of (ensemble_prediction, ensemble_info)
    """
    # Selection logic based on regime and confidence
    if market_regime == 'trending' and timesfm_confidence > fft_confidence:
        # TimesFM for trending markets
        selected_pred = timesfm_pred
        selected_model = 'timesfm'
        selection_reason = 'trending_market_timesfm_advantage'
        
    elif market_regime == 'quiet' and fft_confidence > timesfm_confidence * 0.8:
        # FFT for quiet markets (with some tolerance)
        selected_pred = fft_pred
        selected_model = 'fft'
        selection_reason = 'quiet_market_fft_advantage'
        
    elif fft_confidence > timesfm_confidence * 1.2:
        # FFT has significantly higher confidence
        selected_pred = fft_pred
        selected_model = 'fft'
        selection_reason = 'fft_high_confidence'
        
    elif timesfm_confidence > fft_confidence * 1.2:
        # TimesFM has significantly higher confidence
        selected_pred = timesfm_pred
        selected_model = 'timesfm'
        selection_reason = 'timesfm_high_confidence'
        
    else:
        # Fall back to adaptive ensemble
        return ensemble_adaptive(timesfm_pred, fft_pred, market_regime, timesfm_confidence, fft_confidence)
    
    ensemble_info = {
        'strategy': 'selective',
        'selected_model': selected_model,
        'selection_reason': selection_reason,
        'market_regime': market_regime,
        'timesfm_confidence': timesfm_confidence,
        'fft_confidence': fft_confidence,
        'timesfm_weight': 1.0 if selected_model == 'timesfm' else 0.0,
        'fft_weight': 1.0 if selected_model == 'fft' else 0.0
    }
    
    return selected_pred, ensemble_info

# Ensemble strategy registry
ENSEMBLE_FUNCTIONS = {
    'static': ensemble_static,
    'confidence': ensemble_confidence,
    'adaptive': ensemble_adaptive,
    'selective': ensemble_selective,
    'average': lambda t, f, *args: (0.5 * t + 0.5 * f, {'strategy': 'average', 'timesfm_weight': 0.5, 'fft_weight': 0.5})
}

# ==============================================================================
# Main Hybrid Prediction Function
# ==============================================================================

def predict_fft_timesfm_hybrid(data, verbose=False, ensemble_strategy=DEFAULT_ENSEMBLE_STRATEGY, alpha=DEFAULT_ALPHA):
    """
    Generate hybrid FFT+TimesFM predictions.
    
    Args:
        data: Input time series data (1D array, list, or pandas Series)
              Expected to be pre-normalized close prices
              Must have at least 416 data points
        verbose: If True, print detailed progress information
        ensemble_strategy: Strategy for combining predictions ('static', 'confidence', 'adaptive', 'selective', 'average')
        alpha: Weight for TimesFM in static ensemble (ignored for other strategies)
    
    Returns:
        tuple: (predictions, error_code)
               predictions: numpy array of predicted values (96 points) or None if error
               error_code: HybridPredictionError code indicating success or failure
    
    Example:
        >>> predictions, error = predict_fft_timesfm_hybrid(normalized_close_prices)
        >>> if error == HybridPredictionError.SUCCESS:
        >>>     print(f"Hybrid predicted next 96 minutes: {predictions}")
        >>> else:
        >>>     print(f"Error: {get_hybrid_error_message(error)}")
    """
    
    if verbose:
        print("🚀 Starting FFT+TimesFM hybrid prediction...")
        print(f"   Ensemble strategy: {ensemble_strategy}")
        print(f"   Alpha (static): {alpha}")
    
    try:
        # Validate input data (reuse FFT validation)
        if verbose:
            print("🔍 Validating input data...")
        
        # Use FFT validation (same requirements)
        from prediction_fft import _validate_input_data
        data_array, error_code = _validate_input_data(data)
        if error_code != FFTError.SUCCESS:
            if verbose:
                print(f"❌ Data validation failed: {get_hybrid_error_message(error_code)}")
            return None, error_code
        
        if verbose:
            print(f"✅ Data validated: {len(data_array)} points, range [{data_array.min():.6f}, {data_array.max():.6f}]")
        
        # Prepare context data
        context_data = data_array[-CONTEXT_LENGTH:]
        
        # Generate predictions from both models
        if verbose:
            print("\n🔮 Generating individual model predictions...")
        
        # 1. TimesFM Prediction
        if verbose:
            print("🤖 Running TimesFM prediction...")
        
        timesfm_pred, timesfm_error = predict_timesfm_v2(context_data, verbose=verbose)
        timesfm_success = (timesfm_error == 0)
        timesfm_confidence = 0.8 if timesfm_success else 0.0  # Default confidence for TimesFM
        
        if verbose:
            if timesfm_success:
                print(f"   ✅ TimesFM successful: range [{timesfm_pred.min():.6f}, {timesfm_pred.max():.6f}]")
            else:
                print(f"   ❌ TimesFM failed with error code: {timesfm_error}")
        
        # 2. FFT Prediction  
        if verbose:
            print("🔍 Running FFT similarity prediction...")
        
        fft_pred, fft_error = predict_fft(context_data, verbose=verbose)
        fft_success = (fft_error == 0)
        
        # Get FFT confidence from engine if available
        fft_confidence = 0.0
        market_regime = 'normal'
        
        if fft_success:
            try:
                # Load engine to get detailed prediction info
                from fft_similarity_engine import load_similarity_engine
                engine = load_similarity_engine(max_signatures=5000)
                _, conf, explanation = engine.predict(context_data, adaptive=True, verbose=False)
                fft_confidence = conf
                market_regime = explanation.get('market_regime', 'normal')
            except:
                fft_confidence = 0.5  # Default confidence
        
        if verbose:
            if fft_success:
                print(f"   ✅ FFT successful: range [{fft_pred.min():.6f}, {fft_pred.max():.6f}], confidence: {fft_confidence:.3f}")
                print(f"   🎯 Market regime: {market_regime}")
            else:
                print(f"   ❌ FFT failed with error code: {fft_error}")
        
        # Check if we have at least one successful prediction
        if not timesfm_success and not fft_success:
            if verbose:
                print("❌ Both models failed - cannot generate hybrid prediction")
            return None, HybridPredictionError.BOTH_MODELS_FAILED
        
        # Handle single model success
        if timesfm_success and not fft_success:
            if verbose:
                print("⚠️ Only TimesFM succeeded, using TimesFM prediction")
            return timesfm_pred, HybridPredictionError.SUCCESS
        
        if fft_success and not timesfm_success:
            if verbose:
                print("⚠️ Only FFT succeeded, using FFT prediction")
            return fft_pred, HybridPredictionError.SUCCESS
        
        # Both models succeeded - create ensemble
        if verbose:
            print(f"\n🎯 Creating ensemble prediction (strategy: {ensemble_strategy})...")
        
        try:
            # Get ensemble function
            if ensemble_strategy not in ENSEMBLE_FUNCTIONS:
                if verbose:
                    print(f"⚠️ Unknown ensemble strategy '{ensemble_strategy}', using 'adaptive'")
                ensemble_strategy = 'adaptive'
            
            ensemble_func = ENSEMBLE_FUNCTIONS[ensemble_strategy]
            
            # Apply ensemble strategy
            if ensemble_strategy == 'static':
                ensemble_pred, ensemble_info = ensemble_func(timesfm_pred, fft_pred, alpha)
            elif ensemble_strategy in ['confidence', 'adaptive', 'selective']:
                ensemble_pred, ensemble_info = ensemble_func(
                    timesfm_pred, fft_pred, market_regime, timesfm_confidence, fft_confidence
                )
            else:  # 'average'
                ensemble_pred, ensemble_info = ensemble_func(timesfm_pred, fft_pred)
            
            if verbose:
                print(f"✅ Ensemble prediction generated!")
                print(f"   Strategy: {ensemble_info['strategy']}")
                print(f"   TimesFM weight: {ensemble_info['timesfm_weight']:.3f}")
                print(f"   FFT weight: {ensemble_info['fft_weight']:.3f}")
                print(f"   Ensemble range: [{ensemble_pred.min():.6f}, {ensemble_pred.max():.6f}]")
                print(f"   First 5: {ensemble_pred[:5]}")
                print(f"   Last 5: {ensemble_pred[-5:]}")
                
                # Compare individual predictions
                timesfm_mean = np.mean(timesfm_pred)
                fft_mean = np.mean(fft_pred)
                ensemble_mean = np.mean(ensemble_pred)
                print(f"\n📊 Prediction comparison:")
                print(f"   TimesFM mean: {timesfm_mean:.6f}")
                print(f"   FFT mean: {fft_mean:.6f}")
                print(f"   Ensemble mean: {ensemble_mean:.6f}")
                print(f"   Difference: TimesFM vs FFT = {abs(timesfm_mean - fft_mean):.6f}")
            
            return ensemble_pred, HybridPredictionError.SUCCESS
            
        except Exception as e:
            if verbose:
                print(f"❌ Ensemble creation failed: {e}")
            return None, HybridPredictionError.ENSEMBLE_FAILED
        
    except Exception as e:
        if verbose:
            print(f"❌ Unexpected error during hybrid prediction: {e}")
            import traceback
            traceback.print_exc()
        return None, HybridPredictionError.UNKNOWN_ERROR

# ==============================================================================
# Utility Functions
# ==============================================================================

def get_model_info():
    """Get information about the hybrid model"""
    timesfm_info = get_timesfm_info()
    fft_info = get_fft_info()
    
    return {
        'model_loaded': timesfm_info.get('model_loaded', False) and fft_info.get('model_loaded', False),
        'model_type': 'FFT+TimesFM Hybrid Ensemble',
        'components': {
            'timesfm': timesfm_info,
            'fft': fft_info
        },
        'context_length': CONTEXT_LENGTH,
        'horizon_length': HORIZON_LENGTH,
        'ensemble_strategies': list(ENSEMBLE_STRATEGIES.keys()),
        'default_strategy': DEFAULT_ENSEMBLE_STRATEGY,
        'default_alpha': DEFAULT_ALPHA,
        'market_regimes': list(REGIME_WEIGHTS.keys()),
        'frequency_bands': len(FREQ_BANDS),
        'data_driven_bands': True,
        'adaptive_weighting': True,
        'regime_aware': True,
        'hybrid_features': {
            'static_ensemble': True,
            'confidence_weighting': True,
            'adaptive_regime_detection': True,
            'selective_model_choice': True,
            'fallback_strategies': True
        }
    }

def clear_model_cache():
    """Clear all cached models to free memory"""
    # Clear FFT cache
    from prediction_fft import clear_model_cache as clear_fft_cache
    clear_fft_cache()
    
    # Clear TimesFM cache
    from prediction_timesfm_v2 import clear_model_cache as clear_timesfm_cache
    clear_timesfm_cache()
    
    print("🧹 All hybrid model caches cleared")

# ==============================================================================
# Test Function
# ==============================================================================

def _test_with_random_csv():
    """Test the hybrid prediction function with a random CSV from datasets"""
    
    print("🧪 Testing FFT+TimesFM hybrid prediction with random dataset...")
    
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
        
        # Find a good starting point
        start_idx = max(CONTEXT_LENGTH, len(df) // 2)  # Middle of the data
        
        print(f"🕰️ Starting prediction at index {start_idx} ({df.iloc[start_idx]['timestamp_pt']})")
        
        # Get historical data
        if start_idx < CONTEXT_LENGTH:
            print(f"❌ Not enough historical data (need {CONTEXT_LENGTH}, have {start_idx})")
            return
        
        historical_data = df['close'].iloc[start_idx - CONTEXT_LENGTH:start_idx].values
        
        print(f"📈 Historical data: {len(historical_data)} points")
        print(f"   Range: [{historical_data.min():.6f}, {historical_data.max():.6f}]")
        
        # Test different ensemble strategies
        strategies = ['static', 'adaptive', 'selective']
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"🎯 Testing {strategy} ensemble strategy")
            print(f"{'='*60}")
            
            # Make prediction
            predictions, error_code = predict_fft_timesfm_hybrid(
                historical_data, 
                verbose=True, 
                ensemble_strategy=strategy,
                alpha=0.6
            )
            
            if error_code == HybridPredictionError.SUCCESS:
                print(f"\n✅ {strategy.capitalize()} hybrid prediction successful!")
                
                # Compare with actual if available
                future_end_idx = start_idx + HORIZON_LENGTH
                if future_end_idx <= len(df):
                    actual_future = df['close'].iloc[start_idx:future_end_idx].values
                    
                    # Quick evaluation
                    mae = np.mean(np.abs(predictions - actual_future))
                    pred_returns = np.diff(predictions)
                    actual_returns = np.diff(actual_future)
                    directional_accuracy = np.mean(np.sign(pred_returns) == np.sign(actual_returns)) * 100
                    
                    print(f"   📊 Quick evaluation:")
                    print(f"      MAE: {mae:.6f}")
                    print(f"      Directional Accuracy: {directional_accuracy:.1f}%")
            else:
                print(f"\n❌ {strategy.capitalize()} hybrid prediction failed: {get_hybrid_error_message(error_code)}")
        
        print(f"\n🎉 Hybrid prediction testing complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# Main Guard
# ==============================================================================

if __name__ == "__main__":
    print("🚀 FFT+TimesFM Hybrid Prediction Interface")
    print("=" * 60)
    
    # Show model info
    info = get_model_info()
    print(f"Model Type: {info['model_type']}")
    print(f"Components: TimesFM + FFT Similarity")
    print(f"Context Length: {info['context_length']} minutes")
    print(f"Horizon Length: {info['horizon_length']} minutes")
    print(f"Ensemble Strategies: {info['ensemble_strategies']}")
    print(f"Default Strategy: {info['default_strategy']}")
    print(f"Data-Driven Bands: {info['data_driven_bands']}")
    print(f"Adaptive Weighting: {info['adaptive_weighting']}")
    print("=" * 60)
    
    # Run test
    _test_with_random_csv()
