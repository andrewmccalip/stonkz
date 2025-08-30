#!/usr/bin/env python3
"""
FFT Pattern Matcher - Core similarity search engine using data-driven frequency bands

This module implements the core pattern matching logic for finding historically
similar market conditions using cosine similarity in the frequency domain.

Key Features:
1. Cosine similarity computation using data-driven frequency bands
2. Multi-band similarity with importance weighting  
3. Fast vectorized similarity search across thousands of signatures
4. Configurable similarity thresholds and filtering
5. Integration with FFTSignatureDatabase for efficient access

The matcher supports different similarity strategies:
- Single-band similarity (focus on one frequency range)
- Multi-band weighted similarity (combine all bands)
- Adaptive similarity (adjust weights based on market conditions)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm
from collections import namedtuple

# Add project paths
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

# Import signature loader and frequency bands
from fft_signature_loader import FFTSignatureDatabase, FFTSignature, FREQ_BANDS, BAND_WEIGHTS

# ==============================================================================
# Configuration
# ==============================================================================

# Similarity search parameters
DEFAULT_K = 10                    # Number of nearest neighbors to find
DEFAULT_SIMILARITY_THRESHOLD = 0.1  # Minimum similarity score to consider
MAX_SIMILARITY_RESULTS = 100     # Maximum results to return
CONTEXT_LENGTH = 416

# Similarity computation methods
SIMILARITY_METHODS = {
    'cosine': 'Cosine similarity (recommended)',
    'euclidean': 'Euclidean distance (inverted)',
    'correlation': 'Pearson correlation',
    'manhattan': 'Manhattan distance (inverted)'
}

# ==============================================================================
# Data Structures
# ==============================================================================

SimilarityResult = namedtuple('SimilarityResult', [
    'signature',           # FFTSignature object
    'similarity_score',    # Overall similarity score
    'band_similarities',   # Dict of similarities by band
    'distance',           # Optional distance metric
    'rank'                # Rank in similarity results
])

# ==============================================================================
# Similarity Computation Functions
# ==============================================================================

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1, vec2: Input vectors
    
    Returns:
        Cosine similarity score [0, 1]
    """
    if len(vec1) == 0 or len(vec2) == 0:
        return 0.0
    
    # Ensure same length
    min_len = min(len(vec1), len(vec2))
    vec1 = vec1[:min_len]
    vec2 = vec2[:min_len]
    
    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    
    # Ensure result is in [0, 1] range (cosine can be [-1, 1])
    return max(0.0, similarity)

def compute_band_similarity(sig1: FFTSignature, sig2: FFTSignature, 
                          band_name: str, method: str = 'cosine') -> float:
    """
    Compute similarity between two signatures for a specific frequency band.
    
    Args:
        sig1, sig2: FFT signatures to compare
        band_name: Name of frequency band ('long', 'short', 'medium', 'ultra_short')
        method: Similarity method ('cosine', 'euclidean', 'correlation', 'manhattan')
    
    Returns:
        Similarity score [0, 1]
    """
    # Get band signatures (magnitude spectra)
    mag1, phase1 = sig1.get_band_signature(band_name)
    mag2, phase2 = sig2.get_band_signature(band_name)
    
    if len(mag1) == 0 or len(mag2) == 0:
        return 0.0
    
    if method == 'cosine':
        # Use magnitude for primary similarity
        mag_similarity = compute_cosine_similarity(mag1, mag2)
        
        # Optionally include phase information (experimental)
        if len(phase1) > 0 and len(phase2) > 0:
            phase_similarity = compute_cosine_similarity(np.cos(phase1), np.cos(phase2))
            # Weight magnitude more heavily than phase
            return 0.8 * mag_similarity + 0.2 * phase_similarity
        else:
            return mag_similarity
    
    elif method == 'euclidean':
        # Euclidean distance (inverted to similarity)
        min_len = min(len(mag1), len(mag2))
        mag1_norm = mag1[:min_len] / (np.linalg.norm(mag1[:min_len]) + 1e-10)
        mag2_norm = mag2[:min_len] / (np.linalg.norm(mag2[:min_len]) + 1e-10)
        
        distance = np.linalg.norm(mag1_norm - mag2_norm)
        max_distance = np.sqrt(2)  # Maximum possible distance for normalized vectors
        similarity = 1.0 - (distance / max_distance)
        return max(0.0, similarity)
    
    elif method == 'correlation':
        # Pearson correlation
        if len(mag1) != len(mag2):
            min_len = min(len(mag1), len(mag2))
            mag1 = mag1[:min_len]
            mag2 = mag2[:min_len]
        
        if len(mag1) < 2:
            return 0.0
        
        correlation = np.corrcoef(mag1, mag2)[0, 1]
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0
    
    elif method == 'manhattan':
        # Manhattan distance (inverted to similarity)
        min_len = min(len(mag1), len(mag2))
        mag1_norm = mag1[:min_len] / (np.sum(mag1[:min_len]) + 1e-10)
        mag2_norm = mag2[:min_len] / (np.sum(mag2[:min_len]) + 1e-10)
        
        distance = np.sum(np.abs(mag1_norm - mag2_norm))
        max_distance = 2.0  # Maximum possible L1 distance for normalized vectors
        similarity = 1.0 - (distance / max_distance)
        return max(0.0, similarity)
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def compute_multi_band_similarity(sig1: FFTSignature, sig2: FFTSignature, 
                                method: str = 'cosine', 
                                band_weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
    """
    Compute weighted multi-band similarity between two signatures.
    
    Args:
        sig1, sig2: FFT signatures to compare
        method: Similarity method
        band_weights: Custom band weights (default: use BAND_WEIGHTS)
    
    Returns:
        Tuple of (overall_similarity, band_similarities_dict)
    """
    if band_weights is None:
        band_weights = BAND_WEIGHTS
    
    band_similarities = {}
    weighted_similarity = 0.0
    total_weight = 0.0
    
    # Compute similarity for each frequency band
    for band_name in FREQ_BANDS.keys():
        band_sim = compute_band_similarity(sig1, sig2, band_name, method)
        band_similarities[band_name] = band_sim
        
        # Weight by band importance
        weight = band_weights.get(band_name, 0.0)
        weighted_similarity += band_sim * weight
        total_weight += weight
    
    # Normalize by total weight
    if total_weight > 0:
        overall_similarity = weighted_similarity / total_weight
    else:
        overall_similarity = 0.0
    
    return overall_similarity, band_similarities

# ==============================================================================
# Pattern Matcher Class
# ==============================================================================

class FFTPatternMatcher:
    """Core pattern matching engine using FFT signatures"""
    
    def __init__(self, database: FFTSignatureDatabase, 
                 similarity_method: str = 'cosine',
                 band_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the pattern matcher.
        
        Args:
            database: Loaded FFT signature database
            similarity_method: Method for similarity computation
            band_weights: Custom weights for frequency bands
        """
        self.database = database
        self.similarity_method = similarity_method
        self.band_weights = band_weights or BAND_WEIGHTS
        
        if similarity_method not in SIMILARITY_METHODS:
            raise ValueError(f"Unknown similarity method: {similarity_method}")
        
        print(f"🔍 Initialized FFT Pattern Matcher")
        print(f"   Database: {len(database.signatures):,} signatures")
        print(f"   Method: {similarity_method}")
        print(f"   Bands: {list(FREQ_BANDS.keys())}")
        print(f"   Weights: {self.band_weights}")
    
    def find_similar_patterns(self, query_context: np.ndarray, 
                            k: int = DEFAULT_K,
                            similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                            priority_filter: Optional[str] = None,
                            exclude_source_files: Optional[List[str]] = None) -> List[SimilarityResult]:
        """
        Find k most similar patterns to the query context.
        
        Args:
            query_context: Input context data (416 minutes of price data)
            k: Number of most similar patterns to return
            similarity_threshold: Minimum similarity score to consider
            priority_filter: Filter by priority ('high', 'medium', 'low')
            exclude_source_files: Source files to exclude from search
        
        Returns:
            List of SimilarityResult objects sorted by similarity (highest first)
        """
        if len(query_context) != CONTEXT_LENGTH:
            raise ValueError(f"Query context must be {CONTEXT_LENGTH} points, got {len(query_context)}")
        
        print(f"🔍 Searching for similar patterns...")
        print(f"   Query length: {len(query_context)} points")
        print(f"   Target results: {k}")
        print(f"   Similarity threshold: {similarity_threshold}")
        print(f"   Priority filter: {priority_filter or 'none'}")
        
        # Create query signature for comparison
        query_signature = self._create_query_signature(query_context)
        
        # Get candidate signatures
        if priority_filter:
            candidates = self.database.get_signatures_by_priority(priority_filter)
        else:
            candidates = list(self.database.signatures.values())
        
        # Filter out excluded source files
        if exclude_source_files:
            candidates = [sig for sig in candidates if isinstance(sig, FFTSignature) and sig.source_file not in exclude_source_files]
        
        print(f"   Candidate signatures: {len(candidates):,}")
        
        if not candidates:
            print("❌ No candidate signatures found")
            return []
        
        # Compute similarities
        similarities = []
        
        # Use tqdm for progress tracking
        for candidate in tqdm(candidates, desc="Computing similarities", leave=False):
            if not isinstance(candidate, FFTSignature):
                continue
            
            try:
                # Compute multi-band similarity
                overall_sim, band_sims = compute_multi_band_similarity(
                    query_signature, candidate, self.similarity_method, self.band_weights
                )
                
                # Filter by threshold
                if overall_sim >= similarity_threshold:
                    similarities.append(SimilarityResult(
                        signature=candidate,
                        similarity_score=overall_sim,
                        band_similarities=band_sims,
                        distance=1.0 - overall_sim,  # Convert to distance
                        rank=0  # Will be set after sorting
                    ))
                
            except Exception as e:
                continue
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Limit results and add ranks
        results = []
        for i, sim_result in enumerate(similarities[:min(k, MAX_SIMILARITY_RESULTS)]):
            # Create new result with rank
            ranked_result = SimilarityResult(
                signature=sim_result.signature,
                similarity_score=sim_result.similarity_score,
                band_similarities=sim_result.band_similarities,
                distance=sim_result.distance,
                rank=i + 1
            )
            results.append(ranked_result)
        
        print(f"✅ Found {len(results)} similar patterns")
        if results:
            print(f"   Best similarity: {results[0].similarity_score:.4f}")
            print(f"   Worst similarity: {results[-1].similarity_score:.4f}")
            print(f"   Average similarity: {np.mean([r.similarity_score for r in results]):.4f}")
        
        return results
    
    def _create_query_signature(self, context_data: np.ndarray) -> FFTSignature:
        """Create a temporary FFT signature from query context data"""
        
        # Apply window function and compute FFT
        windowed_data = context_data * np.hanning(len(context_data))
        fft_complex = np.fft.fft(windowed_data)
        fft_magnitude = np.abs(fft_complex)
        fft_phase = np.angle(fft_complex)
        
        # Create signature data dictionary
        query_data = {
            'sequence_id': 'query_context',
            'source_file': 'query',
            'priority_score': 1.0,
            'timestamp_start': None,
            'timestamp_end': None,
            'start_idx': 0,
            'end_idx': len(context_data),
            'context_data': context_data,
            'outcome_data': np.zeros(96),  # Dummy outcome data
            'fft_magnitude': fft_magnitude,
            'fft_phase': fft_phase,
            'total_energy': np.sum(fft_magnitude ** 2),
            'processing_timestamp': None
        }
        
        return FFTSignature(query_data)
    
    def analyze_similarity_distribution(self, query_context: np.ndarray,
                                      sample_size: int = 1000) -> Dict:
        """
        Analyze the distribution of similarities for a query context.
        Useful for understanding similarity thresholds and patterns.
        
        Args:
            query_context: Query context data
            sample_size: Number of signatures to sample for analysis
        
        Returns:
            Dictionary with similarity distribution statistics
        """
        print(f"📊 Analyzing similarity distribution (sample size: {sample_size})...")
        
        # Get random sample of signatures
        sample_signatures = self.database.get_random_signatures(sample_size)
        
        if not sample_signatures:
            return {'error': 'No signatures available for analysis'}
        
        # Create query signature
        query_signature = self._create_query_signature(query_context)
        
        # Compute similarities
        similarities = []
        band_similarities = {band: [] for band in FREQ_BANDS.keys()}
        
        for candidate in tqdm(sample_signatures, desc="Analyzing similarities", leave=False):
            try:
                overall_sim, band_sims = compute_multi_band_similarity(
                    query_signature, candidate, self.similarity_method, self.band_weights
                )
                
                similarities.append(overall_sim)
                
                for band_name, band_sim in band_sims.items():
                    band_similarities[band_name].append(band_sim)
                
            except Exception as e:
                continue
        
        if not similarities:
            return {'error': 'No valid similarities computed'}
        
        # Compute statistics
        similarities = np.array(similarities)
        
        stats = {
            'overall_stats': {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities),
                'median': np.median(similarities),
                'percentiles': {
                    '25': np.percentile(similarities, 25),
                    '75': np.percentile(similarities, 75),
                    '90': np.percentile(similarities, 90),
                    '95': np.percentile(similarities, 95),
                    '99': np.percentile(similarities, 99)
                }
            },
            'band_stats': {},
            'recommended_thresholds': {
                'conservative': np.percentile(similarities, 95),  # Top 5%
                'moderate': np.percentile(similarities, 90),     # Top 10%
                'liberal': np.percentile(similarities, 75)       # Top 25%
            },
            'sample_size': len(similarities)
        }
        
        # Band-specific statistics
        for band_name, band_sims in band_similarities.items():
            if band_sims:
                band_array = np.array(band_sims)
                stats['band_stats'][band_name] = {
                    'mean': np.mean(band_array),
                    'std': np.std(band_array),
                    'min': np.min(band_array),
                    'max': np.max(band_array)
                }
        
        print(f"✅ Analysis complete:")
        print(f"   Mean similarity: {stats['overall_stats']['mean']:.4f}")
        print(f"   Recommended thresholds:")
        print(f"     Conservative (top 5%): {stats['recommended_thresholds']['conservative']:.4f}")
        print(f"     Moderate (top 10%): {stats['recommended_thresholds']['moderate']:.4f}")
        print(f"     Liberal (top 25%): {stats['recommended_thresholds']['liberal']:.4f}")
        
        return stats

# ==============================================================================
# Advanced Pattern Matching
# ==============================================================================

class AdaptivePatternMatcher(FFTPatternMatcher):
    """Advanced pattern matcher with adaptive weighting and market regime detection"""
    
    def __init__(self, database: FFTSignatureDatabase, 
                 similarity_method: str = 'cosine'):
        super().__init__(database, similarity_method)
        
        # Market regime detection parameters
        self.regime_thresholds = {
            'trending': 0.01,    # Minimum trend strength
            'volatile': 0.02,    # Minimum volatility level
            'quiet': 0.005       # Maximum volatility for quiet markets
        }
    
    def detect_market_regime(self, context_data: np.ndarray) -> str:
        """
        Detect market regime from context data to adapt similarity weights.
        
        Args:
            context_data: Context price data
        
        Returns:
            Market regime: 'trending', 'volatile', 'quiet', or 'normal'
        """
        # Calculate price statistics
        returns = np.diff(context_data)
        volatility = np.std(returns)
        trend_strength = abs(np.polyfit(range(len(context_data)), context_data, 1)[0])
        
        # Classify regime
        if trend_strength > self.regime_thresholds['trending']:
            return 'trending'
        elif volatility > self.regime_thresholds['volatile']:
            return 'volatile'
        elif volatility < self.regime_thresholds['quiet']:
            return 'quiet'
        else:
            return 'normal'
    
    def get_adaptive_weights(self, market_regime: str) -> Dict[str, float]:
        """
        Get adaptive frequency band weights based on market regime.
        
        Args:
            market_regime: Detected market regime
        
        Returns:
            Adaptive band weights
        """
        base_weights = BAND_WEIGHTS.copy()
        
        if market_regime == 'trending':
            # Emphasize long-term patterns for trending markets
            base_weights['long'] *= 1.3
            base_weights['short'] *= 0.8
            
        elif market_regime == 'volatile':
            # Emphasize short-term patterns for volatile markets
            base_weights['short'] *= 1.4
            base_weights['ultra_short'] *= 1.2
            base_weights['long'] *= 0.7
            
        elif market_regime == 'quiet':
            # Balanced approach for quiet markets
            base_weights['medium'] *= 1.2
            base_weights['ultra_short'] *= 0.6
        
        # Normalize weights to sum to 1.0
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            base_weights = {k: v / total_weight for k, v in base_weights.items()}
        
        return base_weights
    
    def find_adaptive_similar_patterns(self, query_context: np.ndarray, 
                                     k: int = DEFAULT_K,
                                     similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                                     **kwargs) -> Tuple[List[SimilarityResult], str, Dict[str, float]]:
        """
        Find similar patterns with adaptive weighting based on market regime.
        
        Returns:
            Tuple of (similarity_results, detected_regime, adaptive_weights)
        """
        # Detect market regime
        regime = self.detect_market_regime(query_context)
        
        # Get adaptive weights
        adaptive_weights = self.get_adaptive_weights(regime)
        
        print(f"🎯 Adaptive search:")
        print(f"   Detected regime: {regime}")
        print(f"   Adaptive weights: {adaptive_weights}")
        
        # Temporarily override band weights
        original_weights = self.band_weights
        self.band_weights = adaptive_weights
        
        try:
            # Find similar patterns with adaptive weights
            results = self.find_similar_patterns(
                query_context, k, similarity_threshold, **kwargs
            )
            
            return results, regime, adaptive_weights
            
        finally:
            # Restore original weights
            self.band_weights = original_weights

# ==============================================================================
# Utility Functions
# ==============================================================================

def load_pattern_matcher(database_strategy: str = 'full', 
                        similarity_method: str = 'cosine',
                        max_signatures: Optional[int] = None) -> FFTPatternMatcher:
    """
    Convenience function to load database and create pattern matcher.
    
    Args:
        database_strategy: Database loading strategy
        similarity_method: Similarity computation method
        max_signatures: Maximum signatures to load
    
    Returns:
        Initialized FFTPatternMatcher
    """
    print("🚀 Loading FFT Pattern Matcher")
    print("=" * 50)
    
    # Load database
    from fft_signature_loader import load_fft_database
    database = load_fft_database(strategy=database_strategy, max_signatures=max_signatures)
    
    # Create pattern matcher
    matcher = FFTPatternMatcher(database, similarity_method)
    
    print(f"✅ Pattern matcher ready!")
    return matcher

def test_pattern_matching():
    """Test the pattern matching functionality"""
    print("🧪 Testing FFT Pattern Matching")
    print("=" * 50)
    
    try:
        # Load small database for testing
        matcher = load_pattern_matcher(max_signatures=500)
        
        # Get a random signature as query
        random_sigs = matcher.database.get_random_signatures(1)
        if not random_sigs:
            print("❌ No signatures available for testing")
            return
        
        test_signature = random_sigs[0]
        query_context = test_signature.context_data
        
        print(f"\n🎯 Test query: {test_signature.sequence_id}")
        print(f"   Source: {test_signature.source_file}")
        print(f"   Priority: {test_signature.priority_score:.3f}")
        
        # Test similarity search
        results = matcher.find_similar_patterns(
            query_context, 
            k=5, 
            similarity_threshold=0.1,
            exclude_source_files=[test_signature.source_file]  # Exclude self
        )
        
        print(f"\n📊 Similarity results:")
        for i, result in enumerate(results):
            print(f"   {i+1}. {result.signature.sequence_id}")
            print(f"      Similarity: {result.similarity_score:.4f}")
            print(f"      Source: {result.signature.source_file}")
            print(f"      Band similarities: {result.band_similarities}")
            print()
        
        # Test adaptive matching
        print(f"🎯 Testing adaptive matching...")
        adaptive_matcher = AdaptivePatternMatcher(matcher.database)
        
        adaptive_results, regime, weights = adaptive_matcher.find_adaptive_similar_patterns(
            query_context, k=3, exclude_source_files=[test_signature.source_file]
        )
        
        print(f"   Detected regime: {regime}")
        print(f"   Adaptive results: {len(adaptive_results)}")
        
        print(f"\n✅ Pattern matching tests passed!")
        
    except Exception as e:
        print(f"❌ Pattern matching test failed: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FFT Pattern Matcher')
    parser.add_argument('--test', action='store_true',
                      help='Run pattern matching tests')
    parser.add_argument('--method', type=str, default='cosine',
                      choices=list(SIMILARITY_METHODS.keys()),
                      help='Similarity method')
    parser.add_argument('--max-signatures', type=int, default=None,
                      help='Maximum signatures to load')
    parser.add_argument('--database-strategy', type=str, default='full',
                      choices=['full', 'lazy', 'streaming'],
                      help='Database loading strategy')
    
    args = parser.parse_args()
    
    try:
        if args.test:
            test_pattern_matching()
        else:
            # Load pattern matcher
            matcher = load_pattern_matcher(
                database_strategy=args.database_strategy,
                similarity_method=args.method,
                max_signatures=args.max_signatures
            )
            
            # Show database stats
            stats = matcher.database.get_database_stats()
            print(f"\n📊 Database loaded:")
            print(f"   Signatures: {stats['total_signatures']:,}")
            print(f"   Memory: {stats['memory_usage_mb']:.1f} MB")
            print(f"   Strategy: {stats['load_strategy']}")
            
            print(f"\n💡 Use matcher.find_similar_patterns(context_data) to search for patterns!")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
