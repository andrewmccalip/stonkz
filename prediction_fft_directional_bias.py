#!/usr/bin/env python3
"""
FFT Directional Bias Enhanced TimesFM Prediction

This module implements Strategy C: Using FFT similarity to detect directional bias
in historical patterns and apply that bias to enhance TimesFM predictions.

Key Concept:
1. Find historically similar patterns using FFT frequency analysis
2. Calculate the directional bias from those similar patterns
3. Generate base prediction using TimesFM
4. Apply directional bias adjustment to TimesFM prediction
5. Return enhanced prediction with confidence scoring

This approach leverages:
- TimesFM's superior trend prediction capabilities
- FFT's historical pattern recognition
- Directional bias from similar market conditions
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
from prediction_timesfm_v2 import predict_timesfm_v2, get_model_info as get_timesfm_info, PredictionError as TimesFMError
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
CONTEXT_LENGTH = 416    # Keep 416 for compatibility with existing database
HORIZON_LENGTH = 96     # Prediction horizon in minutes

# Directional Bias Parameters
MIN_SIMILARITY_FOR_BIAS = 0.3      # Minimum similarity to consider for bias
MIN_PATTERNS_FOR_BIAS = 5          # Minimum similar patterns needed
BIAS_STRENGTH_THRESHOLD = 0.1      # Minimum bias strength to apply adjustment
MAX_BIAS_ADJUSTMENT = 0.05         # Maximum adjustment factor (5% of prediction range)
BIAS_DECAY_FACTOR = 0.8            # How bias strength decays over prediction horizon

# FFT Analysis Parameters
K_NEIGHBORS = 20                   # Number of similar patterns to analyze
DATABASE_SIZE = 5000               # Size of FFT database to load

# Dataset Configuration
DATASETS_DIR = SCRIPT_DIR / "datasets" / "ES"

# ==============================================================================
# Error Codes
# ==============================================================================

class DirectionalBiasError:
    """Error codes for directional bias prediction system"""
    
    SUCCESS = 0                    # "Prediction completed successfully"
    INSUFFICIENT_DATA = 101        # "Not enough historical data points"
    INVALID_DATA_FORMAT = 102      # "Input data format is invalid"
    CONTAINS_NAN = 103            # "Input data contains NaN or infinite values"
    TIMESFM_FAILED = 201          # "TimesFM prediction failed"
    FFT_ENGINE_FAILED = 202       # "FFT engine failed to load"
    NO_SIMILAR_PATTERNS = 203     # "No similar patterns found for bias detection"
    BIAS_CALCULATION_FAILED = 204 # "Failed to calculate directional bias"
    UNKNOWN_ERROR = 399          # "Unknown error occurred"

ERROR_MESSAGES = {
    DirectionalBiasError.SUCCESS: "Prediction completed successfully",
    DirectionalBiasError.INSUFFICIENT_DATA: "Not enough historical data points (need at least 416)",
    DirectionalBiasError.INVALID_DATA_FORMAT: "Input data format is invalid",
    DirectionalBiasError.CONTAINS_NAN: "Input data contains NaN or infinite values",
    DirectionalBiasError.TIMESFM_FAILED: "TimesFM prediction failed",
    DirectionalBiasError.FFT_ENGINE_FAILED: "FFT engine failed to load",
    DirectionalBiasError.NO_SIMILAR_PATTERNS: "No similar patterns found for bias detection",
    DirectionalBiasError.BIAS_CALCULATION_FAILED: "Failed to calculate directional bias",
    DirectionalBiasError.UNKNOWN_ERROR: "Unknown error occurred"
}

def get_error_message(error_code):
    """Get human-readable error message"""
    return ERROR_MESSAGES.get(error_code, f"Unknown error code: {error_code}")

# ==============================================================================
# Directional Bias Analysis
# ==============================================================================

def calculate_directional_bias(similar_patterns: List, verbose: bool = False) -> Tuple[float, Dict]:
    """
    Calculate directional bias from similar historical patterns.
    
    Args:
        similar_patterns: List of SimilarityResult objects from FFT matching
        verbose: Whether to print detailed analysis
    
    Returns:
        Tuple of (bias_strength, bias_analysis_dict)
        bias_strength: Float indicating directional bias (-1 to +1)
                      Positive = upward bias, Negative = downward bias
    """
    
    if len(similar_patterns) < MIN_PATTERNS_FOR_BIAS:
        if verbose:
            print(f"⚠️ Insufficient patterns for bias calculation: {len(similar_patterns)} < {MIN_PATTERNS_FOR_BIAS}")
        return 0.0, {'error': 'insufficient_patterns'}
    
    directional_signals = []
    pattern_weights = []
    outcome_stats = []
    
    if verbose:
        print(f"🎯 Analyzing directional bias from {len(similar_patterns)} similar patterns...")
    
    for i, pattern_result in enumerate(similar_patterns):
        try:
            # Get outcome data
            outcome = pattern_result.signature.outcome_data
            context_end = pattern_result.signature.context_data[-1]
            
            # Calculate multiple directional signals
            
            # 1. Overall direction (start to end)
            overall_direction = np.sign(outcome[-1] - outcome[0])
            
            # 2. Early direction (first 30 minutes)
            early_direction = np.sign(outcome[29] - outcome[0]) if len(outcome) > 30 else overall_direction
            
            # 3. Late direction (last 30 minutes)  
            late_direction = np.sign(outcome[-1] - outcome[-30]) if len(outcome) > 30 else overall_direction
            
            # 4. Trend strength
            trend_slope = np.polyfit(range(len(outcome)), outcome, 1)[0]
            trend_direction = np.sign(trend_slope)
            
            # Combine directional signals (weighted by importance)
            combined_direction = (
                0.4 * overall_direction +    # Most important
                0.3 * early_direction +      # Important for short-term
                0.2 * late_direction +       # Medium importance
                0.1 * trend_direction        # Trend component
            )
            
            # Weight by similarity score (more similar = more important)
            similarity_weight = pattern_result.similarity_score
            
            directional_signals.append(combined_direction)
            pattern_weights.append(similarity_weight)
            
            # Store outcome statistics
            outcome_stats.append({
                'overall_return': (outcome[-1] - outcome[0]) / outcome[0] * 100,
                'overall_direction': overall_direction,
                'early_direction': early_direction,
                'trend_slope': trend_slope,
                'similarity': similarity_weight,
                'sequence_id': pattern_result.signature.sequence_id
            })
            
            if verbose and i < 5:  # Show first 5 patterns
                print(f"   Pattern {i+1}: {pattern_result.signature.sequence_id}")
                print(f"     Similarity: {similarity_weight:.4f}")
                print(f"     Overall direction: {overall_direction:+.0f}")
                print(f"     Early direction: {early_direction:+.0f}")
                print(f"     Combined direction: {combined_direction:+.2f}")
                print(f"     Return: {outcome_stats[-1]['overall_return']:+.3f}%")
        
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to analyze pattern {i}: {e}")
            continue
    
    if not directional_signals:
        return 0.0, {'error': 'no_valid_patterns'}
    
    # Calculate weighted directional bias
    directional_signals = np.array(directional_signals)
    pattern_weights = np.array(pattern_weights)
    
    # Normalize weights
    if np.sum(pattern_weights) > 0:
        normalized_weights = pattern_weights / np.sum(pattern_weights)
    else:
        normalized_weights = np.ones(len(pattern_weights)) / len(pattern_weights)
    
    # Weighted bias calculation
    bias_strength = np.sum(directional_signals * normalized_weights)
    
    # Calculate confidence metrics
    bias_consistency = 1.0 - np.std(directional_signals)  # How consistent are the directions
    weight_concentration = np.sum(normalized_weights ** 2)  # How concentrated are the weights
    
    # Overall bias confidence
    bias_confidence = bias_consistency * weight_concentration
    
    # Analysis summary
    bias_analysis = {
        'bias_strength': bias_strength,
        'bias_confidence': bias_confidence,
        'num_patterns': len(similar_patterns),
        'directional_consistency': bias_consistency,
        'weight_concentration': weight_concentration,
        'pattern_stats': {
            'mean_similarity': np.mean(pattern_weights),
            'std_similarity': np.std(pattern_weights),
            'upward_patterns': np.sum(np.array(directional_signals) > 0),
            'downward_patterns': np.sum(np.array(directional_signals) < 0),
            'neutral_patterns': np.sum(np.array(directional_signals) == 0)
        },
        'outcome_statistics': outcome_stats
    }
    
    if verbose:
        print(f"📊 Directional Bias Analysis:")
        print(f"   Bias strength: {bias_strength:+.4f} (-1=down, +1=up)")
        print(f"   Bias confidence: {bias_confidence:.4f}")
        print(f"   Pattern consistency: {bias_consistency:.4f}")
        print(f"   Upward patterns: {bias_analysis['pattern_stats']['upward_patterns']}")
        print(f"   Downward patterns: {bias_analysis['pattern_stats']['downward_patterns']}")
        print(f"   Mean similarity: {bias_analysis['pattern_stats']['mean_similarity']:.4f}")
    
    return bias_strength, bias_analysis

def apply_directional_bias(timesfm_prediction: np.ndarray, bias_strength: float, 
                          bias_confidence: float, verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Apply directional bias to TimesFM prediction.
    
    Args:
        timesfm_prediction: Base TimesFM prediction
        bias_strength: Directional bias strength (-1 to +1)
        bias_confidence: Confidence in the bias (0 to 1)
        verbose: Whether to print details
    
    Returns:
        Tuple of (adjusted_prediction, adjustment_info)
    """
    
    if abs(bias_strength) < BIAS_STRENGTH_THRESHOLD:
        if verbose:
            print(f"⚠️ Bias too weak ({bias_strength:+.4f}), no adjustment applied")
        return timesfm_prediction.copy(), {
            'adjustment_applied': False,
            'reason': 'bias_too_weak',
            'bias_strength': bias_strength
        }
    
    if bias_confidence < 0.3:
        if verbose:
            print(f"⚠️ Bias confidence too low ({bias_confidence:.4f}), no adjustment applied")
        return timesfm_prediction.copy(), {
            'adjustment_applied': False,
            'reason': 'confidence_too_low',
            'bias_confidence': bias_confidence
        }
    
    # Calculate adjustment magnitude
    prediction_range = np.max(timesfm_prediction) - np.min(timesfm_prediction)
    base_adjustment = bias_strength * bias_confidence * MAX_BIAS_ADJUSTMENT
    
    if verbose:
        print(f"🎯 Applying directional bias adjustment:")
        print(f"   Bias strength: {bias_strength:+.4f}")
        print(f"   Bias confidence: {bias_confidence:.4f}")
        print(f"   Base adjustment: {base_adjustment:+.6f}")
    
    # Apply bias with decay over time horizon
    adjusted_prediction = timesfm_prediction.copy()
    
    for t in range(len(adjusted_prediction)):
        # Decay factor: stronger bias for near-term, weaker for far-term
        time_decay = BIAS_DECAY_FACTOR ** (t / len(adjusted_prediction))
        adjustment = base_adjustment * time_decay
        
        # Apply adjustment
        if bias_strength > 0:
            # Upward bias - increase prediction
            adjusted_prediction[t] = timesfm_prediction[t] * (1 + adjustment)
        else:
            # Downward bias - decrease prediction
            adjusted_prediction[t] = timesfm_prediction[t] * (1 + adjustment)  # adjustment is negative
    
    adjustment_info = {
        'adjustment_applied': True,
        'bias_strength': bias_strength,
        'bias_confidence': bias_confidence,
        'base_adjustment': base_adjustment,
        'max_adjustment': MAX_BIAS_ADJUSTMENT,
        'decay_factor': BIAS_DECAY_FACTOR,
        'prediction_range_original': prediction_range,
        'mean_adjustment': np.mean(adjusted_prediction - timesfm_prediction)
    }
    
    if verbose:
        print(f"   Mean adjustment: {adjustment_info['mean_adjustment']:+.6f}")
        print(f"   Adjusted range: [{adjusted_prediction.min():.6f}, {adjusted_prediction.max():.6f}]")
    
    return adjusted_prediction, adjustment_info

# ==============================================================================
# Enhanced Prediction Engine
# ==============================================================================

class DirectionalBiasEngine:
    """Enhanced prediction engine using FFT directional bias"""
    
    def __init__(self):
        """Initialize the directional bias engine"""
        self.fft_engine = None
        self.loaded = False
        
        print("🎯 Initializing FFT Directional Bias Engine")
        print(f"   Base model: TimesFM 2.0")
        print(f"   Enhancement: FFT directional bias detection")
        print(f"   Database size: {DATABASE_SIZE}")
    
    def load_fft_engine(self):
        """Load the FFT similarity engine"""
        if self.loaded and self.fft_engine is not None:
            return
        
        print("🔍 Loading FFT similarity engine for bias detection...")
        self.fft_engine = load_similarity_engine(
            max_signatures=DATABASE_SIZE,
            k_neighbors=K_NEIGHBORS,
            min_similarity=MIN_SIMILARITY_FOR_BIAS
        )
        self.loaded = True
        print("✅ FFT engine loaded for bias detection")
    
    def predict_with_bias(self, context_data: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, float, Dict]:
        """
        Generate prediction with FFT directional bias enhancement.
        
        Args:
            context_data: Historical context data (416 minutes)
            verbose: Whether to print detailed information
        
        Returns:
            Tuple of (prediction, confidence, explanation)
        """
        
        if verbose:
            print("🚀 Generating TimesFM prediction with FFT directional bias...")
        
        try:
            # 1. Generate base TimesFM prediction
            if verbose:
                print("🤖 Step 1: Generating base TimesFM prediction...")
            
            timesfm_pred, timesfm_error = predict_timesfm_v2(context_data, verbose=verbose)
            
            if timesfm_error != 0:
                if verbose:
                    print(f"❌ TimesFM failed with error: {timesfm_error}")
                return np.zeros(HORIZON_LENGTH), 0.0, {
                    'error': 'timesfm_failed',
                    'timesfm_error': timesfm_error
                }
            
            if verbose:
                print(f"   ✅ TimesFM prediction: range [{timesfm_pred.min():.6f}, {timesfm_pred.max():.6f}]")
            
            # 2. Load FFT engine and find similar patterns
            if verbose:
                print("🔍 Step 2: Finding similar patterns for bias detection...")
            
            if not self.loaded:
                self.load_fft_engine()
            
            # Find similar patterns
            try:
                _, confidence, explanation = self.fft_engine.predict(
                    context_data, adaptive=True, verbose=False
                )
                
                if 'error' in explanation:
                    if verbose:
                        print("⚠️ No similar patterns found, using TimesFM prediction without bias")
                    return timesfm_pred, 0.8, {
                        'method': 'timesfm_only',
                        'reason': 'no_similar_patterns',
                        'fft_error': explanation['error']
                    }
                
                # Get similar patterns from the explanation
                similar_patterns = []
                if 'sequence_ids' in explanation and 'similarity_scores' in explanation:
                    # Reconstruct similarity results for bias analysis
                    for seq_id, sim_score in zip(explanation['sequence_ids'], explanation['similarity_scores']):
                        # Get the signature from the database
                        signature = self.fft_engine.database.get_signature(seq_id)
                        if signature:
                            # Create a mock SimilarityResult-like object
                            pattern_result = type('SimilarityResult', (), {
                                'signature': signature,
                                'similarity_score': sim_score
                            })()
                            similar_patterns.append(pattern_result)
                
                if verbose:
                    print(f"   Found {len(similar_patterns)} similar patterns for bias analysis")
                
                # 3. Calculate directional bias
                if verbose:
                    print("📊 Step 3: Calculating directional bias...")
                
                bias_strength, bias_analysis = calculate_directional_bias(similar_patterns, verbose=verbose)
                
                if 'error' in bias_analysis:
                    if verbose:
                        print("⚠️ Bias calculation failed, using TimesFM prediction without bias")
                    return timesfm_pred, 0.8, {
                        'method': 'timesfm_only',
                        'reason': 'bias_calculation_failed',
                        'bias_error': bias_analysis['error']
                    }
                
                # 4. Apply directional bias to TimesFM prediction
                if verbose:
                    print("🎯 Step 4: Applying directional bias to TimesFM prediction...")
                
                bias_confidence = bias_analysis['bias_confidence']
                adjusted_prediction, adjustment_info = apply_directional_bias(
                    timesfm_pred, bias_strength, bias_confidence, verbose=verbose
                )
                
                # 5. Calculate overall confidence
                # Combine TimesFM confidence (assume 0.8) with bias confidence
                timesfm_confidence = 0.8
                overall_confidence = 0.7 * timesfm_confidence + 0.3 * bias_confidence
                
                # Create comprehensive explanation
                explanation_dict = {
                    'method': 'timesfm_with_fft_directional_bias',
                    'timesfm_prediction': timesfm_pred.tolist(),
                    'adjusted_prediction': adjusted_prediction.tolist(),
                    'bias_analysis': bias_analysis,
                    'adjustment_info': adjustment_info,
                    'similar_patterns_info': {
                        'num_patterns': len(similar_patterns),
                        'mean_similarity': np.mean([p.similarity_score for p in similar_patterns]),
                        'similarity_range': [min([p.similarity_score for p in similar_patterns]),
                                           max([p.similarity_score for p in similar_patterns])]
                    },
                    'confidence_breakdown': {
                        'timesfm_confidence': timesfm_confidence,
                        'bias_confidence': bias_confidence,
                        'overall_confidence': overall_confidence
                    },
                    'market_regime': explanation.get('market_regime', 'normal')
                }
                
                if verbose:
                    print(f"✅ Enhanced prediction complete!")
                    print(f"   Overall confidence: {overall_confidence:.4f}")
                    print(f"   Adjustment applied: {adjustment_info['adjustment_applied']}")
                    if adjustment_info['adjustment_applied']:
                        print(f"   Bias direction: {'upward' if bias_strength > 0 else 'downward'}")
                        print(f"   Bias strength: {abs(bias_strength):.4f}")
                
                return adjusted_prediction, overall_confidence, explanation_dict
                
            except Exception as e:
                if verbose:
                    print(f"❌ Bias application failed: {e}")
                # Fall back to TimesFM prediction
                return timesfm_pred, 0.8, {
                    'method': 'timesfm_fallback',
                    'error': str(e)
                }
        
        except Exception as e:
            if verbose:
                print(f"❌ Enhanced prediction failed: {e}")
            return np.zeros(HORIZON_LENGTH), 0.0, {
                'error': str(e),
                'method': 'directional_bias_enhanced'
            }

# Global engine instance
_bias_engine = None

def get_bias_engine():
    """Get the global bias engine instance"""
    global _bias_engine
    if _bias_engine is None:
        _bias_engine = DirectionalBiasEngine()
    return _bias_engine

# ==============================================================================
# Main Prediction Function (Compatible with Backtest Framework)
# ==============================================================================

def predict_fft_directional_bias(data, verbose=False):
    """
    Main prediction function compatible with backtest framework.
    
    Args:
        data: Input time series data (416+ minutes)
        verbose: Whether to print detailed information
    
    Returns:
        tuple: (predictions, error_code)
               0 = success, non-zero = error
    """
    
    if verbose:
        print("🎯 Starting FFT Directional Bias Enhanced Prediction...")
    
    try:
        # Basic validation
        if isinstance(data, list):
            data_array = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            data_array = data.astype(np.float32)
        elif isinstance(data, pd.Series):
            data_array = data.values.astype(np.float32)
        else:
            return None, DirectionalBiasError.INVALID_DATA_FORMAT
        
        if len(data_array) < CONTEXT_LENGTH:
            return None, DirectionalBiasError.INSUFFICIENT_DATA
        
        if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
            return None, DirectionalBiasError.CONTAINS_NAN
        
        # Get context data
        context_data = data_array[-CONTEXT_LENGTH:]
        
        # Load and use bias engine
        engine = get_bias_engine()
        prediction, confidence, explanation = engine.predict_with_bias(context_data, verbose=verbose)
        
        if 'error' in explanation:
            if verbose:
                print(f"❌ Bias prediction error: {explanation['error']}")
            return None, DirectionalBiasError.BIAS_CALCULATION_FAILED
        
        if verbose:
            print(f"✅ Enhanced prediction successful!")
            print(f"   Method: {explanation['method']}")
            print(f"   Confidence: {confidence:.4f}")
        
        return prediction, DirectionalBiasError.SUCCESS
        
    except Exception as e:
        if verbose:
            print(f"❌ Prediction failed: {e}")
        return None, DirectionalBiasError.UNKNOWN_ERROR

def get_model_info():
    """Get model information for backtest integration"""
    return {
        'model_type': 'TimesFM + FFT Directional Bias',
        'method': 'TimesFM enhanced with FFT-based directional bias detection',
        'base_model': 'TimesFM 2.0',
        'enhancement': 'FFT directional bias from similar patterns',
        'context_length': CONTEXT_LENGTH,
        'horizon_length': HORIZON_LENGTH,
        'database_size': DATABASE_SIZE,
        'k_neighbors': K_NEIGHBORS,
        'bias_parameters': {
            'min_similarity': MIN_SIMILARITY_FOR_BIAS,
            'min_patterns': MIN_PATTERNS_FOR_BIAS,
            'bias_threshold': BIAS_STRENGTH_THRESHOLD,
            'max_adjustment': MAX_BIAS_ADJUSTMENT,
            'decay_factor': BIAS_DECAY_FACTOR
        },
        'expected_improvement': 'Enhanced directional accuracy through historical bias detection'
    }

# ==============================================================================
# Test Function
# ==============================================================================

def test_directional_bias_prediction():
    """Test the directional bias enhanced prediction"""
    
    print("🧪 Testing FFT Directional Bias Enhanced Prediction")
    print("=" * 60)
    
    # Load test data
    csv_files = list(DATASETS_DIR.glob("*.csv"))
    if not csv_files:
        print("❌ No test data found")
        return
    
    import random
    test_file = random.choice(csv_files)
    df = pd.read_csv(test_file)
    
    # Get test context and actual outcome
    start_idx = len(df) // 2
    context_data = df['close'].iloc[start_idx-CONTEXT_LENGTH:start_idx].values
    
    print(f"📁 Test file: {test_file.name}")
    print(f"📊 Context: {len(context_data)} points")
    print(f"📈 Context range: [{context_data.min():.6f}, {context_data.max():.6f}]")
    
    # Test enhanced prediction
    prediction, error_code = predict_fft_directional_bias(context_data, verbose=True)
    
    if error_code == DirectionalBiasError.SUCCESS:
        print(f"\n✅ Enhanced prediction successful!")
        
        # Compare with actual if available
        if start_idx + HORIZON_LENGTH <= len(df):
            actual = df['close'].iloc[start_idx:start_idx+HORIZON_LENGTH].values
            
            # Calculate metrics
            pred_returns = np.diff(prediction)
            actual_returns = np.diff(actual)
            directional_accuracy = np.mean(np.sign(pred_returns) == np.sign(actual_returns)) * 100
            
            mae = np.mean(np.abs(prediction - actual))
            correlation = np.corrcoef(prediction, actual)[0, 1]
            
            print(f"📊 Performance Metrics:")
            print(f"   Directional accuracy: {directional_accuracy:.1f}%")
            print(f"   MAE: {mae:.6f}")
            print(f"   Correlation: {correlation:.3f}")
            
            # Also test base TimesFM for comparison
            print(f"\n🔄 Comparing with base TimesFM...")
            timesfm_pred, timesfm_error = predict_timesfm_v2(context_data, verbose=False)
            
            if timesfm_error == 0:
                timesfm_returns = np.diff(timesfm_pred)
                timesfm_directional_accuracy = np.mean(np.sign(timesfm_returns) == np.sign(actual_returns)) * 100
                
                improvement = directional_accuracy - timesfm_directional_accuracy
                print(f"   TimesFM directional accuracy: {timesfm_directional_accuracy:.1f}%")
                print(f"   Enhancement improvement: {improvement:+.1f}%")
            
    else:
        print(f"❌ Enhanced prediction failed: {get_error_message(error_code)}")

if __name__ == "__main__":
    print("🎯 FFT Directional Bias Enhanced TimesFM Prediction")
    print("=" * 60)
    
    # Show model info
    info = get_model_info()
    print(f"Model Type: {info['model_type']}")
    print(f"Method: {info['method']}")
    print(f"Base Model: {info['base_model']}")
    print(f"Enhancement: {info['enhancement']}")
    print("=" * 60)
    
    # Run test
    test_directional_bias_prediction()
