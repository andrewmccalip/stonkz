#!/usr/bin/env python3
"""
FFT Similarity Improvement Analyzer

This script analyzes why FFT similarity matching is performing close to random (50%)
and implements several improvement strategies:

1. Frequency band optimization
2. Alternative similarity metrics
3. Context window optimization  
4. Better outcome aggregation
5. Feature engineering improvements
6. Noise reduction techniques

The goal is to improve directional accuracy significantly above 50%.
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
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# Add project paths
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

# Import our FFT components
from fft_signature_loader import load_fft_database, FFTSignature, FREQ_BANDS
from fft_pattern_matcher import FFTPatternMatcher, compute_cosine_similarity

# ==============================================================================
# Configuration
# ==============================================================================

CONTEXT_LENGTH = 416
HORIZON_LENGTH = 96
ANALYSIS_DIR = SCRIPT_DIR / "datasets" / "fft" / "analysis"

# ==============================================================================
# Improvement Strategies
# ==============================================================================

def analyze_current_performance(num_test_cases: int = 50):
    """Analyze current FFT similarity performance to identify issues"""
    
    print("🔍 Analyzing Current FFT Similarity Performance")
    print("=" * 60)
    
    # Load database
    print("📊 Loading FFT database...")
    database = load_fft_database(max_signatures=2000)
    matcher = FFTPatternMatcher(database)
    
    # Get random test cases
    test_signatures = database.get_random_signatures(num_test_cases)
    
    results = []
    
    print(f"🧪 Testing {len(test_signatures)} cases...")
    
    for test_sig in tqdm(test_signatures, desc="Testing cases"):
        try:
            # Use signature's context as query
            query_context = test_sig.context_data
            actual_outcome = test_sig.outcome_data
            
            # Find similar patterns (exclude same source file)
            similar_patterns = matcher.find_similar_patterns(
                query_context,
                k=10,
                similarity_threshold=0.1,
                exclude_source_files=[test_sig.source_file]
            )
            
            if not similar_patterns:
                continue
            
            # Aggregate outcomes (simple weighted average)
            weights = [r.similarity_score for r in similar_patterns]
            outcomes = [r.signature.outcome_data for r in similar_patterns]
            
            if len(outcomes) > 0:
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                predicted_outcome = np.average(outcomes, axis=0, weights=weights)
                
                # Calculate directional accuracy
                pred_returns = np.diff(predicted_outcome)
                actual_returns = np.diff(actual_outcome)
                pred_directions = np.sign(pred_returns)
                actual_directions = np.sign(actual_returns)
                
                correct_directions = pred_directions == actual_directions
                directional_accuracy = np.mean(correct_directions) * 100
                
                # Store results
                results.append({
                    'sequence_id': test_sig.sequence_id,
                    'directional_accuracy': directional_accuracy,
                    'num_similar_patterns': len(similar_patterns),
                    'best_similarity': similar_patterns[0].similarity_score,
                    'avg_similarity': np.mean([r.similarity_score for r in similar_patterns]),
                    'similarity_spread': np.std([r.similarity_score for r in similar_patterns]),
                    'mae': np.mean(np.abs(predicted_outcome - actual_outcome)),
                    'correlation': np.corrcoef(predicted_outcome, actual_outcome)[0, 1]
                })
        
        except Exception as e:
            continue
    
    if not results:
        print("❌ No valid test results")
        return None
    
    # Analyze results
    results_df = pd.DataFrame(results)
    
    print(f"\n📊 Current Performance Analysis:")
    print(f"   Test cases: {len(results)}")
    print(f"   Mean directional accuracy: {results_df['directional_accuracy'].mean():.1f}%")
    print(f"   Median directional accuracy: {results_df['directional_accuracy'].median():.1f}%")
    print(f"   Std directional accuracy: {results_df['directional_accuracy'].std():.1f}%")
    print(f"   Cases above 50%: {(results_df['directional_accuracy'] > 50).sum()}/{len(results)} ({(results_df['directional_accuracy'] > 50).mean()*100:.1f}%)")
    
    print(f"\n🔍 Similarity Analysis:")
    print(f"   Mean best similarity: {results_df['best_similarity'].mean():.4f}")
    print(f"   Mean avg similarity: {results_df['avg_similarity'].mean():.4f}")
    print(f"   Mean similarity spread: {results_df['similarity_spread'].mean():.4f}")
    print(f"   Mean similar patterns found: {results_df['num_similar_patterns'].mean():.1f}")
    
    print(f"\n📈 Error Analysis:")
    print(f"   Mean MAE: {results_df['mae'].mean():.6f}")
    print(f"   Mean correlation: {results_df['correlation'].mean():.3f}")
    
    return results_df

def test_improved_similarity_metrics():
    """Test different similarity metrics and feature engineering approaches"""
    
    print("\n🔧 Testing Improved Similarity Metrics")
    print("=" * 60)
    
    # Load small database for testing
    database = load_fft_database(max_signatures=500)
    test_signatures = database.get_random_signatures(20)
    
    # Test different similarity approaches
    improvements = {
        'original_cosine': 'Current cosine similarity on FFT magnitude',
        'normalized_cosine': 'Cosine similarity on normalized FFT magnitude',
        'log_magnitude': 'Cosine similarity on log-transformed magnitude',
        'spectral_features': 'Similarity on derived spectral features',
        'phase_aware': 'Combined magnitude + phase similarity',
        'band_weighted': 'Weighted similarity by frequency band importance'
    }
    
    results = defaultdict(list)
    
    print(f"🧪 Testing {len(improvements)} similarity approaches...")
    
    for test_sig in tqdm(test_signatures[:10], desc="Testing improvements"):  # Smaller sample for speed
        query_context = test_sig.context_data
        actual_outcome = test_sig.outcome_data
        
        # Get candidate signatures (exclude same source)
        candidates = [sig for sig in database.get_random_signatures(100) 
                     if sig.source_file != test_sig.source_file]
        
        if len(candidates) < 10:
            continue
        
        # Test each improvement
        for improvement_name, description in improvements.items():
            try:
                similarities = []
                outcomes = []
                
                # Create query signature
                query_sig = create_test_signature(query_context)
                
                for candidate in candidates:
                    if improvement_name == 'original_cosine':
                        # Current approach
                        sim = compute_improved_similarity(query_sig, candidate, 'original')
                    elif improvement_name == 'normalized_cosine':
                        sim = compute_improved_similarity(query_sig, candidate, 'normalized')
                    elif improvement_name == 'log_magnitude':
                        sim = compute_improved_similarity(query_sig, candidate, 'log_magnitude')
                    elif improvement_name == 'spectral_features':
                        sim = compute_improved_similarity(query_sig, candidate, 'spectral_features')
                    elif improvement_name == 'phase_aware':
                        sim = compute_improved_similarity(query_sig, candidate, 'phase_aware')
                    elif improvement_name == 'band_weighted':
                        sim = compute_improved_similarity(query_sig, candidate, 'band_weighted')
                    else:
                        sim = 0.0
                    
                    if sim > 0.1:  # Minimum threshold
                        similarities.append(sim)
                        outcomes.append(candidate.outcome_data)
                
                if len(outcomes) >= 5:  # Need minimum patterns
                    # Aggregate outcomes
                    weights = np.array(similarities)
                    weights = weights / np.sum(weights)
                    predicted_outcome = np.average(outcomes, axis=0, weights=weights)
                    
                    # Calculate directional accuracy
                    pred_returns = np.diff(predicted_outcome)
                    actual_returns = np.diff(actual_outcome)
                    directional_accuracy = np.mean(np.sign(pred_returns) == np.sign(actual_returns)) * 100
                    
                    results[improvement_name].append({
                        'directional_accuracy': directional_accuracy,
                        'num_patterns': len(outcomes),
                        'best_similarity': max(similarities),
                        'avg_similarity': np.mean(similarities)
                    })
            
            except Exception as e:
                continue
    
    # Analyze improvement results
    print(f"\n📊 Similarity Improvement Results:")
    for improvement_name, improvement_results in results.items():
        if improvement_results:
            accuracies = [r['directional_accuracy'] for r in improvement_results]
            similarities = [r['avg_similarity'] for r in improvement_results]
            
            print(f"\n🎯 {improvement_name}:")
            print(f"   Mean accuracy: {np.mean(accuracies):.1f}%")
            print(f"   Cases above 50%: {(np.array(accuracies) > 50).sum()}/{len(accuracies)}")
            print(f"   Mean similarity: {np.mean(similarities):.4f}")
            print(f"   Test cases: {len(improvement_results)}")
    
    return results

def create_test_signature(context_data: np.ndarray) -> FFTSignature:
    """Create a test signature from context data"""
    # Apply window and compute FFT
    windowed_data = context_data * np.hanning(len(context_data))
    fft_complex = np.fft.fft(windowed_data)
    
    data_dict = {
        'sequence_id': 'test_query',
        'source_file': 'test',
        'priority_score': 1.0,
        'timestamp_start': None,
        'timestamp_end': None,
        'start_idx': 0,
        'end_idx': len(context_data),
        'context_data': context_data,
        'outcome_data': np.zeros(96),
        'fft_magnitude': np.abs(fft_complex),
        'fft_phase': np.angle(fft_complex),
        'total_energy': np.sum(np.abs(fft_complex) ** 2),
        'processing_timestamp': None
    }
    
    return FFTSignature(data_dict)

def compute_improved_similarity(sig1: FFTSignature, sig2: FFTSignature, method: str) -> float:
    """Compute similarity using improved methods"""
    
    try:
        if method == 'original':
            # Current approach - cosine similarity on magnitude
            mag1 = sig1.fft_magnitude
            mag2 = sig2.fft_magnitude
            return compute_cosine_similarity(mag1, mag2)
        
        elif method == 'normalized':
            # Normalize magnitude by total energy
            mag1 = sig1.fft_magnitude / (sig1.total_energy + 1e-10)
            mag2 = sig2.fft_magnitude / (sig2.total_energy + 1e-10)
            return compute_cosine_similarity(mag1, mag2)
        
        elif method == 'log_magnitude':
            # Use log-transformed magnitude to reduce dynamic range
            mag1 = np.log10(sig1.fft_magnitude + 1e-10)
            mag2 = np.log10(sig2.fft_magnitude + 1e-10)
            return compute_cosine_similarity(mag1, mag2)
        
        elif method == 'spectral_features':
            # Use derived spectral features instead of raw FFT
            features1 = extract_spectral_features(sig1)
            features2 = extract_spectral_features(sig2)
            return compute_cosine_similarity(features1, features2)
        
        elif method == 'phase_aware':
            # Combine magnitude and phase information
            mag_sim = compute_cosine_similarity(sig1.fft_magnitude, sig2.fft_magnitude)
            
            # Phase similarity using circular correlation
            phase1 = sig1.fft_phase
            phase2 = sig2.fft_phase
            phase_sim = compute_phase_similarity(phase1, phase2)
            
            # Combine (weight magnitude more heavily)
            return 0.7 * mag_sim + 0.3 * phase_sim
        
        elif method == 'band_weighted':
            # Compute similarity for each band and weight by importance
            total_sim = 0.0
            total_weight = 0.0
            
            for band_name, weight in BAND_WEIGHTS.items():
                mag1, _ = sig1.get_band_signature(band_name)
                mag2, _ = sig2.get_band_signature(band_name)
                
                if len(mag1) > 0 and len(mag2) > 0:
                    band_sim = compute_cosine_similarity(mag1, mag2)
                    total_sim += band_sim * weight
                    total_weight += weight
            
            return total_sim / total_weight if total_weight > 0 else 0.0
        
        else:
            return 0.0
    
    except Exception as e:
        return 0.0

def extract_spectral_features(signature: FFTSignature) -> np.ndarray:
    """Extract meaningful spectral features instead of raw FFT"""
    
    magnitude = signature.fft_magnitude
    phase = signature.fft_phase
    
    # Get frequency bins
    freq_bins = np.fft.fftfreq(len(magnitude), d=1.0)[:len(magnitude)//2]
    magnitude_positive = magnitude[:len(freq_bins)]
    
    features = []
    
    # 1. Spectral centroid (center of mass of spectrum)
    if np.sum(magnitude_positive) > 0:
        spectral_centroid = np.sum(freq_bins * magnitude_positive) / np.sum(magnitude_positive)
        features.append(spectral_centroid)
    else:
        features.append(0.0)
    
    # 2. Spectral rolloff (frequency below which 85% of energy lies)
    cumulative_energy = np.cumsum(magnitude_positive ** 2)
    total_energy = cumulative_energy[-1]
    rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
    spectral_rolloff = freq_bins[rolloff_idx[0]] if len(rolloff_idx) > 0 else freq_bins[-1]
    features.append(spectral_rolloff)
    
    # 3. Spectral spread (variance around centroid)
    if len(features) > 0:
        centroid = features[0]
        spectral_spread = np.sqrt(np.sum(((freq_bins - centroid) ** 2) * magnitude_positive) / np.sum(magnitude_positive))
        features.append(spectral_spread)
    else:
        features.append(0.0)
    
    # 4. Spectral flux (rate of change of spectrum)
    if len(magnitude_positive) > 1:
        spectral_flux = np.sum(np.diff(magnitude_positive) ** 2)
        features.append(spectral_flux)
    else:
        features.append(0.0)
    
    # 5. Zero crossing rate of the time series
    context_data = signature.context_data
    zero_crossings = np.sum(np.diff(np.sign(context_data - np.mean(context_data))) != 0)
    features.append(zero_crossings)
    
    # 6. Energy in each data-driven frequency band
    for band_name in FREQ_BANDS.keys():
        band_energy = signature.band_features.get(f'{band_name}_energy', 0.0)
        features.append(band_energy)
    
    # 7. Peak frequencies in each band
    for band_name in FREQ_BANDS.keys():
        peak_freq = signature.band_features.get(f'{band_name}_peak_freq', 0.0)
        features.append(peak_freq)
    
    return np.array(features)

def compute_phase_similarity(phase1: np.ndarray, phase2: np.ndarray) -> float:
    """Compute phase similarity accounting for circular nature of phase"""
    
    if len(phase1) == 0 or len(phase2) == 0:
        return 0.0
    
    # Ensure same length
    min_len = min(len(phase1), len(phase2))
    phase1 = phase1[:min_len]
    phase2 = phase2[:min_len]
    
    # Compute phase difference
    phase_diff = np.angle(np.exp(1j * (phase1 - phase2)))
    
    # Convert to similarity (smaller differences = higher similarity)
    phase_similarity = 1.0 - np.mean(np.abs(phase_diff)) / np.pi
    
    return max(0.0, phase_similarity)

def test_context_window_optimization():
    """Test different context window sizes for better pattern matching"""
    
    print("\n🪟 Testing Context Window Optimization")
    print("=" * 60)
    
    # Test different window sizes
    window_sizes = [208, 312, 416, 520, 624]  # Half, 3/4, current, 1.25x, 1.5x
    
    database = load_fft_database(max_signatures=1000)
    test_signatures = database.get_random_signatures(10)
    
    window_results = {}
    
    for window_size in window_sizes:
        print(f"\n📏 Testing window size: {window_size} minutes")
        
        accuracies = []
        
        for test_sig in test_signatures:
            try:
                # Extract context of specified size
                full_context = test_sig.context_data
                if len(full_context) < window_size:
                    continue
                
                # Take last window_size points
                context_window = full_context[-window_size:]
                actual_outcome = test_sig.outcome_data
                
                # Create temporary signature for this window
                test_window_sig = create_test_signature(context_window)
                
                # Find similar patterns
                similarities = []
                outcomes = []
                
                for candidate in database.get_random_signatures(200):
                    if candidate.source_file == test_sig.source_file:
                        continue
                    
                    # Extract same window size from candidate
                    candidate_context = candidate.context_data
                    if len(candidate_context) < window_size:
                        continue
                    
                    candidate_window = candidate_context[-window_size:]
                    candidate_window_sig = create_test_signature(candidate_window)
                    
                    # Compute similarity
                    sim = compute_improved_similarity(test_window_sig, candidate_window_sig, 'normalized')
                    
                    if sim > 0.2:  # Threshold
                        similarities.append(sim)
                        outcomes.append(candidate.outcome_data)
                
                # Aggregate and evaluate
                if len(outcomes) >= 5:
                    weights = np.array(similarities)
                    weights = weights / np.sum(weights)
                    predicted_outcome = np.average(outcomes, axis=0, weights=weights)
                    
                    # Directional accuracy
                    pred_returns = np.diff(predicted_outcome)
                    actual_returns = np.diff(actual_outcome)
                    directional_accuracy = np.mean(np.sign(pred_returns) == np.sign(actual_returns)) * 100
                    
                    accuracies.append(directional_accuracy)
            
            except Exception as e:
                continue
        
        if accuracies:
            window_results[window_size] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'cases': len(accuracies)
            }
            
            print(f"   Mean accuracy: {np.mean(accuracies):.1f}% (±{np.std(accuracies):.1f}%, n={len(accuracies)})")
    
    # Find best window size
    if window_results:
        best_window = max(window_results.keys(), key=lambda w: window_results[w]['mean_accuracy'])
        print(f"\n🎯 Best window size: {best_window} minutes ({window_results[best_window]['mean_accuracy']:.1f}% accuracy)")
    
    return window_results

def test_noise_reduction_techniques():
    """Test noise reduction and signal processing improvements"""
    
    print("\n🔇 Testing Noise Reduction Techniques")
    print("=" * 60)
    
    techniques = {
        'raw': 'No preprocessing',
        'detrend': 'Remove linear trend',
        'highpass_filter': 'High-pass filter to remove low-frequency drift',
        'moving_average_detrend': 'Remove moving average trend',
        'standardize': 'Z-score standardization',
        'robust_scale': 'Robust scaling (median/MAD)'
    }
    
    database = load_fft_database(max_signatures=500)
    test_signatures = database.get_random_signatures(10)
    
    technique_results = defaultdict(list)
    
    for test_sig in test_signatures[:5]:  # Small sample for speed
        query_context = test_sig.context_data
        actual_outcome = test_sig.outcome_data
        
        for technique_name, description in techniques.items():
            try:
                # Preprocess context
                processed_context = apply_preprocessing(query_context, technique_name)
                
                # Find similar patterns with preprocessing
                similarities = []
                outcomes = []
                
                for candidate in database.get_random_signatures(100):
                    if candidate.source_file == test_sig.source_file:
                        continue
                    
                    # Preprocess candidate context
                    processed_candidate = apply_preprocessing(candidate.context_data, technique_name)
                    
                    # Create signatures and compute similarity
                    query_sig = create_test_signature(processed_context)
                    candidate_sig = create_test_signature(processed_candidate)
                    
                    sim = compute_improved_similarity(query_sig, candidate_sig, 'normalized')
                    
                    if sim > 0.2:
                        similarities.append(sim)
                        outcomes.append(candidate.outcome_data)
                
                # Evaluate
                if len(outcomes) >= 5:
                    weights = np.array(similarities)
                    weights = weights / np.sum(weights)
                    predicted_outcome = np.average(outcomes, axis=0, weights=weights)
                    
                    pred_returns = np.diff(predicted_outcome)
                    actual_returns = np.diff(actual_outcome)
                    directional_accuracy = np.mean(np.sign(pred_returns) == np.sign(actual_returns)) * 100
                    
                    technique_results[technique_name].append(directional_accuracy)
            
            except Exception as e:
                continue
    
    # Report results
    print(f"\n📊 Noise Reduction Results:")
    for technique_name, accuracies in technique_results.items():
        if accuracies:
            print(f"   {technique_name:20}: {np.mean(accuracies):5.1f}% (±{np.std(accuracies):4.1f}%, n={len(accuracies)})")
    
    return technique_results

def apply_preprocessing(data: np.ndarray, technique: str) -> np.ndarray:
    """Apply preprocessing technique to time series data"""
    
    if technique == 'raw':
        return data.copy()
    
    elif technique == 'detrend':
        # Remove linear trend
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        trend = np.polyval(coeffs, x)
        return data - trend
    
    elif technique == 'highpass_filter':
        # Simple high-pass filter (remove low frequencies)
        # Use moving average as low-pass, subtract to get high-pass
        window = min(60, len(data) // 4)  # 1-hour window or 1/4 of data
        moving_avg = np.convolve(data, np.ones(window)/window, mode='same')
        return data - moving_avg
    
    elif technique == 'moving_average_detrend':
        # Remove long-term moving average trend
        window = min(120, len(data) // 2)  # 2-hour window
        moving_avg = np.convolve(data, np.ones(window)/window, mode='same')
        return data - moving_avg
    
    elif technique == 'standardize':
        # Z-score standardization
        return (data - np.mean(data)) / (np.std(data) + 1e-10)
    
    elif technique == 'robust_scale':
        # Robust scaling using median and MAD
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return (data - median) / (mad + 1e-10)
    
    else:
        return data.copy()

# ==============================================================================
# Main Analysis Function
# ==============================================================================

def run_similarity_improvement_analysis():
    """Run comprehensive analysis to improve FFT similarity matching"""
    
    print("🚀 FFT Similarity Improvement Analysis")
    print("=" * 70)
    print("Goal: Improve directional accuracy significantly above 50%")
    print()
    
    try:
        # 1. Analyze current performance
        current_results = analyze_current_performance(num_test_cases=30)
        
        if current_results is not None:
            baseline_accuracy = current_results['directional_accuracy'].mean()
            print(f"\n📊 Baseline Performance: {baseline_accuracy:.1f}%")
        else:
            baseline_accuracy = 50.0
            print(f"\n⚠️ Could not establish baseline, assuming 50%")
        
        # 2. Test improved similarity metrics
        similarity_results = test_improved_similarity_metrics()
        
        # 3. Test context window optimization
        window_results = test_context_window_optimization()
        
        # 4. Test noise reduction
        noise_results = test_noise_reduction_techniques()
        
        # 5. Generate recommendations
        print(f"\n🎯 IMPROVEMENT RECOMMENDATIONS")
        print("=" * 50)
        
        # Find best similarity method
        best_similarity_method = None
        best_similarity_accuracy = baseline_accuracy
        
        for method, results in similarity_results.items():
            if results:
                mean_acc = np.mean([r['directional_accuracy'] for r in results])
                if mean_acc > best_similarity_accuracy:
                    best_similarity_accuracy = mean_acc
                    best_similarity_method = method
        
        if best_similarity_method:
            improvement = best_similarity_accuracy - baseline_accuracy
            print(f"1. 🎯 Best Similarity Method: {best_similarity_method}")
            print(f"   Improvement: +{improvement:.1f}% (from {baseline_accuracy:.1f}% to {best_similarity_accuracy:.1f}%)")
        
        # Find best window size
        if window_results:
            best_window = max(window_results.keys(), key=lambda w: window_results[w]['mean_accuracy'])
            best_window_accuracy = window_results[best_window]['mean_accuracy']
            window_improvement = best_window_accuracy - baseline_accuracy
            
            print(f"2. 🪟 Best Window Size: {best_window} minutes")
            print(f"   Improvement: +{window_improvement:.1f}% (from {baseline_accuracy:.1f}% to {best_window_accuracy:.1f}%)")
        
        # Find best noise reduction
        best_noise_technique = None
        best_noise_accuracy = baseline_accuracy
        
        for technique, accuracies in noise_results.items():
            if accuracies:
                mean_acc = np.mean(accuracies)
                if mean_acc > best_noise_accuracy:
                    best_noise_accuracy = mean_acc
                    best_noise_technique = technique
        
        if best_noise_technique:
            noise_improvement = best_noise_accuracy - baseline_accuracy
            print(f"3. 🔇 Best Noise Reduction: {best_noise_technique}")
            print(f"   Improvement: +{noise_improvement:.1f}% (from {baseline_accuracy:.1f}% to {best_noise_accuracy:.1f}%)")
        
        print(f"\n💡 IMPLEMENTATION SUGGESTIONS:")
        print(f"   • Update similarity computation to use: {best_similarity_method or 'normalized_cosine'}")
        print(f"   • Consider context window of: {best_window if 'best_window' in locals() else 416} minutes")
        print(f"   • Apply preprocessing: {best_noise_technique or 'detrend'}")
        print(f"   • Lower similarity threshold to find more patterns")
        print(f"   • Use more sophisticated outcome aggregation")
        print(f"   • Consider ensemble of multiple similarity methods")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    try:
        run_similarity_improvement_analysis()
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
