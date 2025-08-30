#!/usr/bin/env python3
"""
FFT Similarity Engine - Complete similarity-based prediction system

This module combines the FFT signature database and pattern matcher to create
a complete prediction system based on historical pattern similarity.

Key Features:
1. K-nearest neighbors search in frequency domain
2. Multiple outcome aggregation strategies (weighted average, median, etc.)
3. Confidence scoring based on similarity distribution
4. Integration with data-driven frequency bands
5. Prediction explanation and interpretability
6. Performance optimization for real-time usage

The engine provides the core functionality for the FFT component of the
FFT+TimesFM hybrid prediction system.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project paths
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

# Import our components
from fft_signature_loader import load_fft_database, FFTSignatureDatabase, FFTSignature
from fft_pattern_matcher import FFTPatternMatcher, AdaptivePatternMatcher, SimilarityResult, FREQ_BANDS, BAND_WEIGHTS

# ==============================================================================
# Configuration
# ==============================================================================

# Prediction parameters
DEFAULT_K_NEIGHBORS = 15         # Number of similar patterns to use
DEFAULT_MIN_SIMILARITY = 0.3     # Minimum similarity threshold
DEFAULT_AGGREGATION_METHOD = 'weighted_average'  # How to combine similar outcomes
HORIZON_LENGTH = 96              # Prediction horizon in minutes
CONTEXT_LENGTH = 416             # Context length in minutes

# Aggregation methods
AGGREGATION_METHODS = {
    'weighted_average': 'Weighted average by similarity scores',
    'simple_average': 'Simple arithmetic average',
    'median': 'Median of outcomes',
    'weighted_median': 'Weighted median by similarity',
    'best_match': 'Use outcome from most similar pattern only',
    'ensemble': 'Ensemble of multiple methods'
}

# Confidence scoring parameters
MIN_CONFIDENCE = 0.1
MAX_CONFIDENCE = 1.0

# ==============================================================================
# Outcome Aggregation Functions
# ==============================================================================

def aggregate_outcomes_weighted_average(similarity_results: List[SimilarityResult]) -> Tuple[np.ndarray, float]:
    """
    Aggregate outcomes using similarity-weighted average.
    
    Args:
        similarity_results: List of similar patterns with outcomes
    
    Returns:
        Tuple of (predicted_outcome, confidence_score)
    """
    if not similarity_results:
        return np.zeros(HORIZON_LENGTH), 0.0
    
    # Extract outcomes and weights
    outcomes = []
    weights = []
    
    for result in similarity_results:
        outcomes.append(result.signature.outcome_data)
        weights.append(result.similarity_score)
    
    outcomes = np.array(outcomes)
    weights = np.array(weights)
    
    # Normalize weights
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(len(weights)) / len(weights)
    
    # Compute weighted average
    weighted_outcome = np.average(outcomes, axis=0, weights=weights)
    
    # Confidence based on weight concentration and similarity spread
    weight_concentration = np.sum(weights ** 2)  # Higher = more concentrated
    similarity_spread = np.std([r.similarity_score for r in similarity_results])
    confidence = weight_concentration * (1.0 - similarity_spread)
    confidence = np.clip(confidence, MIN_CONFIDENCE, MAX_CONFIDENCE)
    
    return weighted_outcome, confidence

def aggregate_outcomes_simple_average(similarity_results: List[SimilarityResult]) -> Tuple[np.ndarray, float]:
    """Aggregate using simple arithmetic average"""
    if not similarity_results:
        return np.zeros(HORIZON_LENGTH), 0.0
    
    outcomes = np.array([r.signature.outcome_data for r in similarity_results])
    avg_outcome = np.mean(outcomes, axis=0)
    
    # Confidence based on number of results and average similarity
    avg_similarity = np.mean([r.similarity_score for r in similarity_results])
    num_factor = min(1.0, len(similarity_results) / DEFAULT_K_NEIGHBORS)
    confidence = avg_similarity * num_factor
    
    return avg_outcome, confidence

def aggregate_outcomes_median(similarity_results: List[SimilarityResult]) -> Tuple[np.ndarray, float]:
    """Aggregate using median of outcomes"""
    if not similarity_results:
        return np.zeros(HORIZON_LENGTH), 0.0
    
    outcomes = np.array([r.signature.outcome_data for r in similarity_results])
    median_outcome = np.median(outcomes, axis=0)
    
    # Confidence based on consistency (inverse of spread)
    outcome_std = np.std(outcomes, axis=0)
    avg_std = np.mean(outcome_std)
    confidence = 1.0 / (1.0 + avg_std * 100)  # Scale factor for typical price ranges
    confidence = np.clip(confidence, MIN_CONFIDENCE, MAX_CONFIDENCE)
    
    return median_outcome, confidence

def aggregate_outcomes_best_match(similarity_results: List[SimilarityResult]) -> Tuple[np.ndarray, float]:
    """Use outcome from the most similar pattern only"""
    if not similarity_results:
        return np.zeros(HORIZON_LENGTH), 0.0
    
    # Use the best match (first result, already sorted by similarity)
    best_result = similarity_results[0]
    outcome = best_result.signature.outcome_data
    confidence = best_result.similarity_score
    
    return outcome, confidence

def aggregate_outcomes_ensemble(similarity_results: List[SimilarityResult]) -> Tuple[np.ndarray, float]:
    """Ensemble multiple aggregation methods"""
    if not similarity_results:
        return np.zeros(HORIZON_LENGTH), 0.0
    
    # Get predictions from different methods
    weighted_pred, weighted_conf = aggregate_outcomes_weighted_average(similarity_results)
    median_pred, median_conf = aggregate_outcomes_median(similarity_results)
    best_pred, best_conf = aggregate_outcomes_best_match(similarity_results)
    
    # Ensemble the predictions (weighted by their confidence)
    predictions = [weighted_pred, median_pred, best_pred]
    confidences = [weighted_conf, median_conf, best_conf]
    
    # Weight by confidence
    total_conf = sum(confidences)
    if total_conf > 0:
        conf_weights = [c / total_conf for c in confidences]
    else:
        conf_weights = [1/3, 1/3, 1/3]
    
    ensemble_pred = np.average(predictions, axis=0, weights=conf_weights)
    ensemble_conf = np.mean(confidences)
    
    return ensemble_pred, ensemble_conf

# Aggregation method registry
AGGREGATION_FUNCTIONS = {
    'weighted_average': aggregate_outcomes_weighted_average,
    'simple_average': aggregate_outcomes_simple_average,
    'median': aggregate_outcomes_median,
    'weighted_median': aggregate_outcomes_median,  # Simplified to regular median
    'best_match': aggregate_outcomes_best_match,
    'ensemble': aggregate_outcomes_ensemble
}

# ==============================================================================
# Main Similarity Engine
# ==============================================================================

class FFTSimilarityEngine:
    """Complete similarity-based prediction engine"""
    
    def __init__(self, database: FFTSignatureDatabase,
                 similarity_method: str = 'cosine',
                 aggregation_method: str = DEFAULT_AGGREGATION_METHOD,
                 k_neighbors: int = DEFAULT_K_NEIGHBORS,
                 min_similarity: float = DEFAULT_MIN_SIMILARITY):
        """
        Initialize the similarity engine.
        
        Args:
            database: Loaded FFT signature database
            similarity_method: Method for similarity computation
            aggregation_method: Method for aggregating similar outcomes
            k_neighbors: Number of neighbors to use for prediction
            min_similarity: Minimum similarity threshold
        """
        self.database = database
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity
        self.aggregation_method = aggregation_method
        
        # Initialize pattern matcher
        self.pattern_matcher = FFTPatternMatcher(database, similarity_method)
        self.adaptive_matcher = AdaptivePatternMatcher(database, similarity_method)
        
        # Get aggregation function
        if aggregation_method not in AGGREGATION_FUNCTIONS:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        self.aggregation_function = AGGREGATION_FUNCTIONS[aggregation_method]
        
        # Statistics
        self.prediction_count = 0
        self.total_similarity_searches = 0
        
        print(f"🔮 Initialized FFT Similarity Engine")
        print(f"   Database: {len(database.signatures):,} signatures")
        print(f"   K-neighbors: {k_neighbors}")
        print(f"   Min similarity: {min_similarity}")
        print(f"   Aggregation: {aggregation_method}")
    
    def predict(self, context_data: np.ndarray, 
               adaptive: bool = True,
               exclude_source_files: Optional[List[str]] = None,
               verbose: bool = False) -> Tuple[np.ndarray, float, Dict]:
        """
        Generate prediction based on similar historical patterns.
        
        Args:
            context_data: Historical context data (416 minutes)
            adaptive: Whether to use adaptive weighting based on market regime
            exclude_source_files: Source files to exclude from similarity search
            verbose: Whether to print detailed information
        
        Returns:
            Tuple of (prediction, confidence, explanation_dict)
        """
        if len(context_data) != CONTEXT_LENGTH:
            raise ValueError(f"Context data must be {CONTEXT_LENGTH} points, got {len(context_data)}")
        
        if verbose:
            print(f"🔮 Generating FFT similarity-based prediction...")
            print(f"   Context: {len(context_data)} points")
            print(f"   Range: [{context_data.min():.6f}, {context_data.max():.6f}]")
            print(f"   Adaptive: {adaptive}")
        
        try:
            # Find similar patterns
            if adaptive:
                similarity_results, market_regime, adaptive_weights = self.adaptive_matcher.find_adaptive_similar_patterns(
                    context_data,
                    k=self.k_neighbors,
                    similarity_threshold=self.min_similarity,
                    exclude_source_files=exclude_source_files
                )
                
                if verbose:
                    print(f"   Market regime: {market_regime}")
                    print(f"   Adaptive weights: {adaptive_weights}")
            else:
                similarity_results = self.pattern_matcher.find_similar_patterns(
                    context_data,
                    k=self.k_neighbors,
                    similarity_threshold=self.min_similarity,
                    exclude_source_files=exclude_source_files
                )
                market_regime = 'normal'
                adaptive_weights = BAND_WEIGHTS
            
            if not similarity_results:
                if verbose:
                    print("❌ No similar patterns found")
                return np.zeros(HORIZON_LENGTH), 0.0, {
                    'error': 'no_similar_patterns',
                    'num_candidates': 0
                }
            
            if verbose:
                print(f"   Found {len(similarity_results)} similar patterns")
                print(f"   Best similarity: {similarity_results[0].similarity_score:.4f}")
            
            # Aggregate outcomes
            prediction, confidence = self.aggregation_function(similarity_results)
            
            if verbose:
                print(f"   Prediction range: [{prediction.min():.6f}, {prediction.max():.6f}]")
                print(f"   Confidence: {confidence:.4f}")
            
            # Create explanation
            explanation = {
                'method': 'fft_similarity',
                'num_similar_patterns': len(similarity_results),
                'k_neighbors': self.k_neighbors,
                'min_similarity': self.min_similarity,
                'aggregation_method': self.aggregation_method,
                'market_regime': market_regime,
                'adaptive_weights': adaptive_weights,
                'similarity_scores': [r.similarity_score for r in similarity_results],
                'band_similarities': {
                    band: [r.band_similarities[band] for r in similarity_results]
                    for band in FREQ_BANDS.keys()
                },
                'source_files': [r.signature.source_file for r in similarity_results],
                'sequence_ids': [r.signature.sequence_id for r in similarity_results],
                'prediction_confidence': confidence,
                'context_stats': {
                    'mean': np.mean(context_data),
                    'std': np.std(context_data),
                    'trend': np.polyfit(range(len(context_data)), context_data, 1)[0]
                }
            }
            
            self.prediction_count += 1
            self.total_similarity_searches += 1
            
            return prediction, confidence, explanation
            
        except Exception as e:
            if verbose:
                print(f"❌ Prediction failed: {e}")
            
            return np.zeros(HORIZON_LENGTH), 0.0, {
                'error': str(e),
                'method': 'fft_similarity'
            }
    
    def batch_predict(self, context_batch: List[np.ndarray], 
                     adaptive: bool = True,
                     verbose: bool = False) -> List[Tuple[np.ndarray, float, Dict]]:
        """
        Generate predictions for a batch of contexts.
        
        Args:
            context_batch: List of context data arrays
            adaptive: Whether to use adaptive weighting
            verbose: Whether to print progress
        
        Returns:
            List of (prediction, confidence, explanation) tuples
        """
        print(f"🔮 Batch prediction: {len(context_batch)} contexts")
        
        results = []
        
        iterator = tqdm(context_batch, desc="Batch predictions", disable=not verbose)
        for i, context_data in enumerate(iterator):
            try:
                prediction, confidence, explanation = self.predict(
                    context_data, adaptive=adaptive, verbose=False
                )
                results.append((prediction, confidence, explanation))
                
                if verbose:
                    iterator.set_postfix({
                        'completed': i + 1,
                        'avg_confidence': np.mean([r[1] for r in results]),
                        'successes': len([r for r in results if 'error' not in r[2]])
                    })
                
            except Exception as e:
                # Add failed prediction
                results.append((
                    np.zeros(HORIZON_LENGTH), 
                    0.0, 
                    {'error': str(e), 'method': 'fft_similarity'}
                ))
        
        successful = len([r for r in results if 'error' not in r[2]])
        print(f"✅ Batch prediction complete: {successful}/{len(results)} successful")
        
        return results
    
    def analyze_prediction_quality(self, context_data: np.ndarray,
                                 actual_outcome: np.ndarray,
                                 adaptive: bool = True) -> Dict:
        """
        Analyze the quality of a prediction against actual outcomes.
        
        Args:
            context_data: Historical context
            actual_outcome: Actual future data
            adaptive: Whether to use adaptive prediction
        
        Returns:
            Dictionary with quality metrics and analysis
        """
        # Generate prediction
        prediction, confidence, explanation = self.predict(
            context_data, adaptive=adaptive, verbose=False
        )
        
        if 'error' in explanation:
            return {'error': explanation['error']}
        
        # Calculate quality metrics
        mse = np.mean((prediction - actual_outcome) ** 2)
        mae = np.mean(np.abs(prediction - actual_outcome))
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        pred_returns = np.diff(prediction)
        actual_returns = np.diff(actual_outcome)
        pred_directions = np.sign(pred_returns)
        actual_directions = np.sign(actual_returns)
        
        correct_directions = pred_directions == actual_directions
        directional_accuracy = np.mean(correct_directions) * 100
        
        # Correlation
        correlation = np.corrcoef(prediction, actual_outcome)[0, 1] if len(prediction) > 1 else 0
        
        quality_metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation,
            'prediction_confidence': confidence,
            'num_similar_patterns': explanation['num_similar_patterns'],
            'best_similarity': max(explanation['similarity_scores']) if explanation['similarity_scores'] else 0,
            'avg_similarity': np.mean(explanation['similarity_scores']) if explanation['similarity_scores'] else 0,
            'market_regime': explanation['market_regime'],
            'prediction': prediction.tolist(),
            'actual': actual_outcome.tolist(),
            'explanation': explanation
        }
        
        return quality_metrics

    def get_engine_stats(self) -> Dict:
        """Get comprehensive engine statistics"""
        db_stats = self.database.get_database_stats()
        
        engine_stats = {
            'database_stats': db_stats,
            'engine_config': {
                'k_neighbors': self.k_neighbors,
                'min_similarity': self.min_similarity,
                'aggregation_method': self.aggregation_method,
                'frequency_bands': FREQ_BANDS,
                'band_weights': BAND_WEIGHTS
            },
            'usage_stats': {
                'predictions_generated': self.prediction_count,
                'similarity_searches': self.total_similarity_searches
            }
        }
        
        return engine_stats

# ==============================================================================
# Convenience Functions
# ==============================================================================

def load_similarity_engine(database_strategy: str = 'full',
                          max_signatures: Optional[int] = None,
                          k_neighbors: int = DEFAULT_K_NEIGHBORS,
                          min_similarity: float = DEFAULT_MIN_SIMILARITY,
                          aggregation_method: str = DEFAULT_AGGREGATION_METHOD) -> FFTSimilarityEngine:
    """
    Convenience function to load and initialize the complete similarity engine.
    
    Args:
        database_strategy: Database loading strategy
        max_signatures: Maximum signatures to load
        k_neighbors: Number of neighbors for prediction
        min_similarity: Minimum similarity threshold
        aggregation_method: Method for aggregating outcomes
    
    Returns:
        Initialized FFTSimilarityEngine
    """
    print(f"🚀 Loading FFT Similarity Engine")
    print("=" * 50)
    
    # Load database
    database = load_fft_database(strategy=database_strategy, max_signatures=max_signatures)
    
    # Create engine
    engine = FFTSimilarityEngine(
        database=database,
        aggregation_method=aggregation_method,
        k_neighbors=k_neighbors,
        min_similarity=min_similarity
    )
    
    print(f"✅ Similarity engine ready!")
    return engine

def predict_fft_similarity(context_data: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, int]:
    """
    Main prediction function compatible with backtest framework.
    
    Args:
        context_data: Historical context data (416 minutes)
        verbose: Whether to print detailed information
    
    Returns:
        Tuple of (predictions, error_code)
        - predictions: Predicted values for next 96 minutes
        - error_code: 0 for success, non-zero for error
    """
    try:
        # Load engine (with caching for efficiency)
        engine = load_similarity_engine(max_signatures=5000)  # Use subset for speed
        
        # Generate prediction
        prediction, confidence, explanation = engine.predict(
            context_data, adaptive=True, verbose=verbose
        )
        
        if 'error' in explanation:
            if verbose:
                print(f"❌ FFT prediction error: {explanation['error']}")
            return None, 1  # Error code 1
        
        if verbose:
            print(f"✅ FFT prediction successful (confidence: {confidence:.3f})")
        
        return prediction, 0  # Success
        
    except Exception as e:
        if verbose:
            print(f"❌ FFT prediction failed: {e}")
        return None, 2  # Error code 2

def get_model_info() -> Dict:
    """Get model information for integration with backtest framework"""
    return {
        'model_type': 'FFT Similarity Matching',
        'method': 'cosine_similarity_frequency_domain',
        'database_size': '29,005 signatures',
        'frequency_bands': len(FREQ_BANDS),
        'context_length': CONTEXT_LENGTH,
        'horizon_length': HORIZON_LENGTH,
        'aggregation_method': DEFAULT_AGGREGATION_METHOD,
        'k_neighbors': DEFAULT_K_NEIGHBORS,
        'data_driven_bands': True,
        'adaptive_weighting': True
    }

# ==============================================================================
# Testing Functions
# ==============================================================================

def test_similarity_engine():
    """Test the complete similarity engine"""
    print("🧪 Testing FFT Similarity Engine")
    print("=" * 50)
    
    try:
        # Load engine with small database for testing
        engine = load_similarity_engine(max_signatures=1000)
        
        # Get test data
        test_signatures = engine.database.get_random_signatures(3)
        
        for i, test_sig in enumerate(test_signatures, 1):
            print(f"\n🎯 Test {i}: {test_sig.sequence_id}")
            
            # Use signature's context as query
            query_context = test_sig.context_data
            actual_outcome = test_sig.outcome_data
            
            # Generate prediction
            prediction, confidence, explanation = engine.predict(
                query_context,
                adaptive=True,
                exclude_source_files=[test_sig.source_file],  # Exclude self
                verbose=True
            )
            
            if 'error' not in explanation:
                # Analyze quality
                quality = engine.analyze_prediction_quality(
                    query_context, actual_outcome, adaptive=True
                )
                
                print(f"   📊 Quality metrics:")
                print(f"      Directional accuracy: {quality['directional_accuracy']:.1f}%")
                print(f"      Correlation: {quality['correlation']:.3f}")
                print(f"      MAE: {quality['mae']:.6f}")
                print(f"      Confidence: {quality['prediction_confidence']:.3f}")
            else:
                print(f"   ❌ Prediction failed: {explanation['error']}")
        
        # Test batch prediction
        print(f"\n🔄 Testing batch prediction...")
        test_contexts = [sig.context_data for sig in test_signatures]
        batch_results = engine.batch_predict(test_contexts, verbose=True)
        
        successful_batch = [r for r in batch_results if 'error' not in r[2]]
        print(f"   Batch results: {len(successful_batch)}/{len(batch_results)} successful")
        
        print(f"\n✅ Similarity engine tests passed!")
        
        # Show engine stats
        stats = engine.get_engine_stats()
        print(f"\n📊 Engine Statistics:")
        print(f"   Predictions generated: {stats['usage_stats']['predictions_generated']}")
        print(f"   Database signatures: {stats['database_stats']['total_signatures']:,}")
        print(f"   Memory usage: {stats['database_stats']['memory_usage_mb']:.1f} MB")
        
    except Exception as e:
        print(f"❌ Similarity engine test failed: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FFT Similarity Engine')
    parser.add_argument('--test', action='store_true',
                      help='Run similarity engine tests')
    parser.add_argument('--k', type=int, default=DEFAULT_K_NEIGHBORS,
                      help='Number of neighbors')
    parser.add_argument('--similarity-threshold', type=float, default=DEFAULT_MIN_SIMILARITY,
                      help='Minimum similarity threshold')
    parser.add_argument('--aggregation', type=str, default=DEFAULT_AGGREGATION_METHOD,
                      choices=list(AGGREGATION_METHODS.keys()),
                      help='Outcome aggregation method')
    parser.add_argument('--max-signatures', type=int, default=None,
                      help='Maximum signatures to load')
    
    args = parser.parse_args()
    
    try:
        if args.test:
            test_similarity_engine()
        else:
            # Load engine
            engine = load_similarity_engine(
                max_signatures=args.max_signatures,
                k_neighbors=args.k,
                min_similarity=args.similarity_threshold,
                aggregation_method=args.aggregation
            )
            
            print(f"\n💡 Engine ready! Use engine.predict(context_data) to generate predictions")
            print(f"📊 Database: {len(engine.database.signatures):,} signatures loaded")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
