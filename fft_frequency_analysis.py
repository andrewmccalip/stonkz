#!/usr/bin/env python3
"""
FFT Frequency Analysis - Discover Dominant Frequencies Across Entire Database

This script analyzes all 29,005+ FFT signatures to find the most dominant frequencies
in ES futures data. Instead of using hard-coded frequency bands, this discovers
natural frequency patterns from the data itself.

Key Features:
1. Load and analyze all FFT signatures in the database
2. Focus on 5-120 minute patterns (practical trading timeframes)
3. Find peak frequencies across the entire dataset
4. Cluster similar frequency patterns
5. Generate data-driven frequency bands
6. Create comprehensive visualizations and statistics
7. Export results for use in prediction models

This will replace hard-coded frequency bands with emergent, data-driven ones.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from tqdm import tqdm
from collections import defaultdict, Counter

# Try to import scipy for advanced analysis
try:
    from scipy import stats, signal, cluster
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy not available, some advanced analysis will be simplified")

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

# Add project paths
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

# ==============================================================================
# Configuration
# ==============================================================================

# Paths
FFT_DIR = SCRIPT_DIR / "datasets" / "fft"
SIGNATURES_DIR = FFT_DIR / "signatures"
METADATA_DIR = FFT_DIR / "metadata"
ANALYSIS_DIR = FFT_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

# Analysis parameters
CONTEXT_LENGTH = 416
SAMPLING_RATE = 1.0  # 1 sample per minute
NYQUIST_FREQ = SAMPLING_RATE / 2

# Focus on practical trading timeframes: 5 minutes to 120 minutes
MIN_PERIOD_MINUTES = 5
MAX_PERIOD_MINUTES = 120
MIN_FREQUENCY = 1.0 / MAX_PERIOD_MINUTES  # 0.0083 cycles/min
MAX_FREQUENCY = 1.0 / MIN_PERIOD_MINUTES  # 0.2 cycles/min

# Analysis settings
BATCH_SIZE = 500  # Process signatures in batches to manage memory
TOP_N_FREQUENCIES = 20  # Number of top frequencies to analyze in detail
FREQUENCY_RESOLUTION = 100  # Number of frequency bins for analysis
MIN_ENERGY_THRESHOLD = 1e-10  # Minimum energy to consider a frequency significant

# Clustering parameters
NUM_CLUSTERS = 8  # Number of frequency clusters to discover
CLUSTER_METHOD = 'ward'  # Hierarchical clustering method

# ==============================================================================
# Data Loading Functions
# ==============================================================================

def load_database_metadata() -> Dict:
    """Load database metadata"""
    metadata_path = METADATA_DIR / "database_info.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Database metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)

def get_all_signature_files() -> List[Path]:
    """Get all signature files from the database"""
    signature_files = list(SIGNATURES_DIR.glob("*.csv"))
    
    if not signature_files:
        raise FileNotFoundError(f"No signature files found in {SIGNATURES_DIR}")
    
    return sorted(signature_files)

def load_full_timeseries_batch(signature_files: List[Path], start_idx: int, batch_size: int, 
                              show_progress: bool = True) -> List[Dict]:
    """Load a batch of full time series data for comprehensive frequency analysis"""
    end_idx = min(start_idx + batch_size, len(signature_files))
    batch_files = signature_files[start_idx:end_idx]
    
    signatures = []
    failed_count = 0
    
    # Use tqdm for progress tracking within batch
    iterator = tqdm(batch_files, desc="  Loading full time series", leave=False, disable=not show_progress)
    
    for file_path in iterator:
        try:
            # Get the source CSV file name from the signature file name
            signature_name = file_path.stem  # e.g., "ES_Apr_2024_01_0000"
            source_csv_name = "_".join(signature_name.split("_")[:-1]) + ".csv"  # e.g., "ES_Apr_2024_01.csv"
            source_csv_path = SCRIPT_DIR / "datasets" / "ES" / source_csv_name
            
            if not source_csv_path.exists():
                failed_count += 1
                continue
            
            # Load the full source CSV file
            df_full = pd.read_csv(source_csv_path)
            
            if len(df_full) < MIN_PERIOD_MINUTES:  # Need at least minimum period length
                failed_count += 1
                continue
            
            # Extract the full close price time series
            close_prices = df_full['close'].values
            
            # Remove any NaN or infinite values
            close_prices = close_prices[np.isfinite(close_prices)]
            
            if len(close_prices) < MIN_PERIOD_MINUTES:
                failed_count += 1
                continue
            
            # Compute FFT of the entire time series
            # Apply window function to reduce spectral leakage
            windowed_data = close_prices * np.hanning(len(close_prices))
            fft_complex = np.fft.fft(windowed_data)
            fft_magnitude = np.abs(fft_complex)
            
            # Calculate total energy
            total_energy = np.sum(fft_magnitude ** 2)
            
            # Get basic metadata
            timestamp_start = df_full.iloc[0].get('timestamp_pt', None)
            timestamp_end = df_full.iloc[-1].get('timestamp_pt', None)
            
            signature = {
                'file_path': file_path,
                'source_csv_path': source_csv_path,
                'sequence_id': source_csv_name.replace('.csv', '_full_series'),
                'source_file': source_csv_name,
                'full_length': len(close_prices),
                'fft_magnitude': fft_magnitude,
                'total_energy': total_energy,
                'timestamp_start': timestamp_start,
                'timestamp_end': timestamp_end,
                'close_prices': close_prices  # Keep for additional analysis if needed
            }
            
            signatures.append(signature)
            
            # Update progress description with current stats
            if show_progress:
                iterator.set_postfix({
                    'loaded': len(signatures),
                    'failed': failed_count,
                    'avg_length': int(np.mean([s['full_length'] for s in signatures])) if signatures else 0
                })
            
        except Exception as e:
            failed_count += 1
            continue
    
    if failed_count > 0:
        print(f"⚠️  Failed to load {failed_count}/{len(batch_files)} full time series in batch")
    
    return signatures

def get_frequency_bins(n_samples: int) -> np.ndarray:
    """Get frequency bins for FFT analysis"""
    return np.fft.fftfreq(n_samples, d=1.0)[:n_samples//2]

def frequency_to_period(frequency: float) -> float:
    """Convert frequency (cycles/min) to period (minutes)"""
    return 1.0 / frequency if frequency > 0 else np.inf

def period_to_frequency(period: float) -> float:
    """Convert period (minutes) to frequency (cycles/min)"""
    return 1.0 / period if period > 0 else 0

# ==============================================================================
# Frequency Analysis Functions
# ==============================================================================

def analyze_frequency_spectrum(signatures: List[Dict], show_progress: bool = True) -> Dict:
    """Analyze frequency spectrum across multiple full time series"""
    print(f"🔍 Analyzing frequency spectrum for {len(signatures)} full time series...")
    
    # Since we're analyzing full time series of different lengths, we need a common frequency grid
    # Use a high-resolution frequency grid that covers our range of interest
    common_freq_resolution = 1000  # Number of frequency bins in our range
    freq_range = np.linspace(MIN_FREQUENCY, MAX_FREQUENCY, common_freq_resolution)
    
    print(f"📊 Using common frequency grid: {common_freq_resolution} bins from {MIN_FREQUENCY:.6f} to {MAX_FREQUENCY:.6f} cycles/min")
    print(f"   Period range: {frequency_to_period(MAX_FREQUENCY):.1f} to {frequency_to_period(MIN_FREQUENCY):.1f} minutes")
    
    # Accumulate spectral data
    spectral_accumulator = np.zeros(len(freq_range))
    energy_weighted_accumulator = np.zeros(len(freq_range))
    total_signatures = 0
    total_energy = 0
    
    frequency_peaks = []  # Store individual peak frequencies
    length_stats = []  # Track time series lengths
    
    # Use tqdm for progress tracking
    iterator = tqdm(signatures, desc="  Analyzing full spectra", leave=False, disable=not show_progress)
    
    for signature in iterator:
        try:
            fft_magnitude = signature['fft_magnitude']
            series_length = signature['full_length']
            length_stats.append(series_length)
            
            # Get frequency bins for this specific time series
            series_freq_bins = get_frequency_bins(series_length)
            
            # Filter to our frequency range of interest
            freq_mask = (series_freq_bins >= MIN_FREQUENCY) & (series_freq_bins <= MAX_FREQUENCY)
            filtered_series_freq_bins = series_freq_bins[freq_mask]
            magnitude_filtered = fft_magnitude[:len(series_freq_bins)][freq_mask]
            
            if len(magnitude_filtered) == 0:
                continue
            
            # Interpolate to common frequency grid
            interpolated_magnitude = np.interp(freq_range, filtered_series_freq_bins, magnitude_filtered)
            
            # Accumulate raw magnitudes
            spectral_accumulator += interpolated_magnitude
            
            # Weight by total energy of the signature
            signature_energy = signature['total_energy']
            energy_weighted_accumulator += interpolated_magnitude * signature_energy
            total_energy += signature_energy
            
            # Find peak frequency in this signature
            peak_idx = np.argmax(magnitude_filtered)
            peak_frequency = filtered_series_freq_bins[peak_idx]
            peak_magnitude = magnitude_filtered[peak_idx]
            
            if peak_magnitude > MIN_ENERGY_THRESHOLD:
                frequency_peaks.append({
                    'frequency': peak_frequency,
                    'period': frequency_to_period(peak_frequency),
                    'magnitude': peak_magnitude,
                    'energy': signature_energy,
                    'sequence_id': signature['sequence_id'],
                    'series_length': series_length
                })
            
            total_signatures += 1
            
            # Update progress with current stats
            if show_progress:
                iterator.set_postfix({
                    'processed': total_signatures,
                    'peaks': len(frequency_peaks),
                    'avg_length': int(np.mean(length_stats)),
                    'energy': f'{total_energy:.2e}'
                })
            
        except Exception as e:
            continue
    
    if total_signatures == 0:
        raise ValueError("No valid signatures processed")
    
    # Calculate average spectra
    avg_spectrum = spectral_accumulator / total_signatures
    energy_weighted_spectrum = energy_weighted_accumulator / total_energy if total_energy > 0 else avg_spectrum
    
    print(f"✅ Processed {total_signatures} full time series successfully")
    print(f"📈 Found {len(frequency_peaks)} significant frequency peaks")
    print(f"📏 Time series lengths: {np.min(length_stats):.0f} to {np.max(length_stats):.0f} minutes (avg: {np.mean(length_stats):.0f})")
    
    return {
        'freq_bins': freq_range,
        'avg_spectrum': avg_spectrum,
        'energy_weighted_spectrum': energy_weighted_spectrum,
        'frequency_peaks': frequency_peaks,
        'total_signatures': total_signatures,
        'total_energy': total_energy,
        'length_stats': length_stats
    }

def find_dominant_frequencies(spectrum_data: Dict, top_n: int = TOP_N_FREQUENCIES) -> List[Dict]:
    """Find the most dominant frequencies across the dataset"""
    print(f"🎯 Finding top {top_n} dominant frequencies...")
    
    freq_bins = spectrum_data['freq_bins']
    avg_spectrum = spectrum_data['avg_spectrum']
    energy_weighted_spectrum = spectrum_data['energy_weighted_spectrum']
    frequency_peaks = spectrum_data['frequency_peaks']
    
    # Method 1: Peaks in average spectrum
    peak_indices = signal.find_peaks(avg_spectrum, height=np.mean(avg_spectrum))[0] if HAS_SCIPY else np.argsort(avg_spectrum)[-top_n:]
    
    spectrum_peaks = []
    for idx in peak_indices:
        if idx < len(freq_bins):
            freq = freq_bins[idx]
            magnitude = avg_spectrum[idx]
            energy_weight = energy_weighted_spectrum[idx]
            
            spectrum_peaks.append({
                'frequency': freq,
                'period': frequency_to_period(freq),
                'avg_magnitude': magnitude,
                'energy_weighted_magnitude': energy_weight,
                'source': 'spectrum_peak'
            })
    
    # Method 2: Most common individual peak frequencies
    peak_counter = Counter()
    for peak in frequency_peaks:
        # Round to reasonable precision for counting
        rounded_freq = round(peak['frequency'], 4)
        peak_counter[rounded_freq] += 1
    
    common_peaks = []
    for freq, count in peak_counter.most_common(top_n):
        # Find representative peak data
        matching_peaks = [p for p in frequency_peaks if abs(p['frequency'] - freq) < 1e-4]
        if matching_peaks:
            avg_magnitude = np.mean([p['magnitude'] for p in matching_peaks])
            avg_energy = np.mean([p['energy'] for p in matching_peaks])
            
            common_peaks.append({
                'frequency': freq,
                'period': frequency_to_period(freq),
                'count': count,
                'avg_magnitude': avg_magnitude,
                'avg_energy': avg_energy,
                'source': 'individual_peaks'
            })
    
    # Method 3: Energy-weighted dominant frequencies
    energy_peaks = []
    sorted_indices = np.argsort(energy_weighted_spectrum)[-top_n:]
    for idx in reversed(sorted_indices):
        if idx < len(freq_bins):
            freq = freq_bins[idx]
            energy_weight = energy_weighted_spectrum[idx]
            avg_magnitude = avg_spectrum[idx]
            
            energy_peaks.append({
                'frequency': freq,
                'period': frequency_to_period(freq),
                'energy_weighted_magnitude': energy_weight,
                'avg_magnitude': avg_magnitude,
                'source': 'energy_weighted'
            })
    
    print(f"📊 Found {len(spectrum_peaks)} spectrum peaks")
    print(f"📊 Found {len(common_peaks)} common individual peaks")
    print(f"📊 Found {len(energy_peaks)} energy-weighted peaks")
    
    return {
        'spectrum_peaks': spectrum_peaks,
        'common_peaks': common_peaks,
        'energy_peaks': energy_peaks,
        'all_individual_peaks': frequency_peaks
    }

def cluster_frequencies(dominant_frequencies: Dict, num_clusters: int = NUM_CLUSTERS) -> Dict:
    """Cluster similar frequencies to discover natural groupings"""
    print(f"🔗 Clustering frequencies into {num_clusters} natural groups...")
    
    if not HAS_SCIPY:
        print("⚠️  Scipy not available, skipping clustering analysis")
        return {'clusters': [], 'cluster_centers': [], 'method': 'none'}
    
    # Combine all significant frequencies
    all_frequencies = []
    
    # Add spectrum peaks
    for peak in dominant_frequencies['spectrum_peaks']:
        all_frequencies.append({
            'frequency': peak['frequency'],
            'period': peak['period'],
            'weight': peak['avg_magnitude'],
            'source': 'spectrum'
        })
    
    # Add common individual peaks
    for peak in dominant_frequencies['common_peaks']:
        all_frequencies.append({
            'frequency': peak['frequency'],
            'period': peak['period'],
            'weight': peak['count'] * peak['avg_magnitude'],
            'source': 'individual'
        })
    
    if len(all_frequencies) < num_clusters:
        print(f"⚠️  Only {len(all_frequencies)} frequencies found, reducing clusters to {len(all_frequencies)}")
        num_clusters = len(all_frequencies)
    
    # Prepare data for clustering (use log-space for frequencies)
    frequencies_array = np.array([f['frequency'] for f in all_frequencies])
    weights_array = np.array([f['weight'] for f in all_frequencies])
    
    # Create feature matrix (frequency + weight)
    X = np.column_stack([
        np.log10(frequencies_array + 1e-10),  # Log frequency
        np.log10(weights_array + 1e-10)       # Log weight
    ])
    
    # Perform hierarchical clustering
    try:
        linkage_matrix = linkage(X, method=CLUSTER_METHOD)
        cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        
        # Analyze clusters
        clusters = []
        cluster_centers = []
        
        for cluster_id in range(1, num_clusters + 1):
            cluster_mask = cluster_labels == cluster_id
            cluster_frequencies = frequencies_array[cluster_mask]
            cluster_weights = weights_array[cluster_mask]
            
            if len(cluster_frequencies) > 0:
                # Calculate cluster statistics
                center_freq = np.average(cluster_frequencies, weights=cluster_weights)
                center_period = frequency_to_period(center_freq)
                freq_std = np.sqrt(np.average((cluster_frequencies - center_freq)**2, weights=cluster_weights))
                total_weight = np.sum(cluster_weights)
                
                cluster_info = {
                    'cluster_id': cluster_id,
                    'center_frequency': center_freq,
                    'center_period': center_period,
                    'frequency_std': freq_std,
                    'frequency_range': (np.min(cluster_frequencies), np.max(cluster_frequencies)),
                    'period_range': (frequency_to_period(np.max(cluster_frequencies)), 
                                   frequency_to_period(np.min(cluster_frequencies))),
                    'total_weight': total_weight,
                    'num_frequencies': len(cluster_frequencies),
                    'frequencies': cluster_frequencies.tolist(),
                    'weights': cluster_weights.tolist()
                }
                
                clusters.append(cluster_info)
                cluster_centers.append(center_freq)
        
        # Sort clusters by center frequency
        clusters.sort(key=lambda x: x['center_frequency'])
        cluster_centers = [c['center_frequency'] for c in clusters]
        
        print(f"✅ Successfully clustered into {len(clusters)} groups")
        for i, cluster in enumerate(clusters):
            print(f"   Cluster {i+1}: {cluster['center_period']:.1f}min ± {frequency_to_period(cluster['center_frequency'] - cluster['frequency_std']):.1f}min "
                  f"({cluster['num_frequencies']} frequencies)")
        
        return {
            'clusters': clusters,
            'cluster_centers': cluster_centers,
            'linkage_matrix': linkage_matrix,
            'cluster_labels': cluster_labels,
            'method': CLUSTER_METHOD
        }
        
    except Exception as e:
        print(f"❌ Clustering failed: {e}")
        return {'clusters': [], 'cluster_centers': [], 'method': 'failed'}

def generate_data_driven_bands(clustering_results: Dict) -> List[Dict]:
    """Generate frequency bands based on clustering results"""
    print("🎯 Generating data-driven frequency bands...")
    
    if not clustering_results['clusters']:
        print("⚠️  No clusters available, using fallback method")
        return []
    
    bands = []
    clusters = clustering_results['clusters']
    
    for i, cluster in enumerate(clusters):
        # Define band boundaries with some overlap
        center_freq = cluster['center_frequency']
        freq_std = cluster['frequency_std']
        
        # Band extends ±2 standard deviations from center
        min_freq = max(MIN_FREQUENCY, center_freq - 2 * freq_std)
        max_freq = min(MAX_FREQUENCY, center_freq + 2 * freq_std)
        
        # Convert to periods for intuitive naming
        center_period = frequency_to_period(center_freq)
        min_period = frequency_to_period(max_freq)
        max_period = frequency_to_period(min_freq)
        
        # Generate band name based on period
        if center_period < 10:
            band_name = f"ultra_short_{i+1}"
            display_name = f"Ultra-Short ({center_period:.1f}min)"
        elif center_period < 30:
            band_name = f"short_{i+1}"
            display_name = f"Short ({center_period:.1f}min)"
        elif center_period < 60:
            band_name = f"medium_{i+1}"
            display_name = f"Medium ({center_period:.1f}min)"
        else:
            band_name = f"long_{i+1}"
            display_name = f"Long ({center_period:.1f}min)"
        
        band = {
            'band_name': band_name,
            'display_name': display_name,
            'frequency_range': (min_freq, max_freq),
            'period_range': (min_period, max_period),
            'center_frequency': center_freq,
            'center_period': center_period,
            'num_source_frequencies': cluster['num_frequencies'],
            'total_weight': cluster['total_weight'],
            'cluster_id': cluster['cluster_id']
        }
        
        bands.append(band)
        print(f"   {display_name}: {min_period:.1f}-{max_period:.1f}min ({min_freq:.6f}-{max_freq:.6f} cyc/min)")
    
    return bands

# ==============================================================================
# Visualization Functions
# ==============================================================================

def create_comprehensive_frequency_analysis_plot(spectrum_data: Dict, dominant_frequencies: Dict, 
                                                clustering_results: Dict, data_driven_bands: List[Dict],
                                                save_path: Optional[Path] = None):
    """Create comprehensive visualization of frequency analysis"""
    
    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Title
    fig.suptitle('Comprehensive Frequency Analysis - ES Futures Database (Full Time Series)\n' + 
                 f'Analysis of {spectrum_data["total_signatures"]:,} Complete CSV Files (5-120 minute patterns)',
                 fontsize=18, fontweight='bold')
    
    freq_bins = spectrum_data['freq_bins']
    avg_spectrum = spectrum_data['avg_spectrum']
    energy_weighted_spectrum = spectrum_data['energy_weighted_spectrum']
    
    # Convert frequency bins to periods for better interpretation
    period_bins = [frequency_to_period(f) for f in freq_bins]
    
    # ==============================================================================
    # 1. Average Spectrum across all signatures
    # ==============================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot average spectrum
    ax1.plot(period_bins, avg_spectrum, color='#2E86AB', linewidth=2, label='Average Spectrum', alpha=0.8)
    ax1.fill_between(period_bins, 0, avg_spectrum, color='#2E86AB', alpha=0.3)
    
    # Overlay energy-weighted spectrum
    ax1.plot(period_bins, energy_weighted_spectrum, color='#F18F01', linewidth=2, 
             label='Energy-Weighted Spectrum', alpha=0.8)
    
    # Mark dominant frequencies
    for peak in dominant_frequencies['spectrum_peaks'][:10]:  # Top 10
        period = peak['period']
        magnitude = peak['avg_magnitude']
        ax1.plot(period, magnitude, 'ro', markersize=8, markeredgecolor='black', 
                label='Spectrum Peak' if peak == dominant_frequencies['spectrum_peaks'][0] else "")
    
    # Highlight data-driven bands
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_driven_bands)))
    for band, color in zip(data_driven_bands, colors):
        min_period, max_period = band['period_range']
        ax1.axvspan(min_period, max_period, alpha=0.2, color=color, 
                   label=f"{band['display_name']}")
    
    ax1.set_xlabel('Period (minutes)', fontsize=14)
    ax1.set_ylabel('Average Magnitude', fontsize=14)
    ax1.set_title('Frequency Spectrum Analysis - All Signatures', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(MIN_PERIOD_MINUTES, MAX_PERIOD_MINUTES)
    
    # ==============================================================================
    # 2. Individual Peak Frequency Distribution
    # ==============================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    individual_periods = [frequency_to_period(p['frequency']) for p in spectrum_data['frequency_peaks']]
    
    # Create histogram
    n, bins, patches = ax2.hist(individual_periods, bins=50, alpha=0.7, color='#A23B72', 
                               edgecolor='black', density=True)
    
    # Add KDE if scipy available
    if HAS_SCIPY and len(individual_periods) > 10:
        try:
            kde = stats.gaussian_kde(individual_periods)
            x_range = np.linspace(MIN_PERIOD_MINUTES, MAX_PERIOD_MINUTES, 200)
            ax2.plot(x_range, kde(x_range), 'r-', linewidth=3, label='KDE', alpha=0.8)
        except:
            pass
    
    # Mark cluster centers
    if clustering_results['cluster_centers']:
        for center_freq in clustering_results['cluster_centers']:
            center_period = frequency_to_period(center_freq)
            ax2.axvline(center_period, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Period (minutes)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Distribution of Individual Peak Frequencies', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(MIN_PERIOD_MINUTES, MAX_PERIOD_MINUTES)
    
    # ==============================================================================
    # 3. Clustering Dendrogram (if available)
    # ==============================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    if HAS_SCIPY and 'linkage_matrix' in clustering_results:
        try:
            dendrogram(clustering_results['linkage_matrix'], ax=ax3, leaf_rotation=90)
            ax3.set_title('Frequency Clustering Dendrogram', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Frequency Index')
            ax3.set_ylabel('Distance')
        except Exception as e:
            ax3.text(0.5, 0.5, f'Dendrogram unavailable:\n{str(e)}', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Clustering Dendrogram (Unavailable)', fontsize=14, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Clustering analysis\nnot available\n(requires scipy)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Clustering Analysis (Unavailable)', fontsize=14, fontweight='bold')
    
    # ==============================================================================
    # 4. Data-Driven Band Analysis
    # ==============================================================================
    ax4 = fig.add_subplot(gs[1, 2])
    
    if data_driven_bands:
        band_names = [band['display_name'] for band in data_driven_bands]
        band_weights = [band['total_weight'] for band in data_driven_bands]
        band_counts = [band['num_source_frequencies'] for band in data_driven_bands]
        
        # Create dual-axis plot
        ax4_twin = ax4.twinx()
        
        x_pos = np.arange(len(band_names))
        bars1 = ax4.bar(x_pos - 0.2, band_weights, 0.4, label='Total Weight', alpha=0.7, color='#2E86AB')
        bars2 = ax4_twin.bar(x_pos + 0.2, band_counts, 0.4, label='Frequency Count', alpha=0.7, color='#F18F01')
        
        ax4.set_xlabel('Data-Driven Frequency Bands', fontsize=12)
        ax4.set_ylabel('Total Weight', fontsize=12, color='#2E86AB')
        ax4_twin.set_ylabel('Frequency Count', fontsize=12, color='#F18F01')
        ax4.set_title('Data-Driven Band Importance', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(band_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, weight in zip(bars1, band_weights):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{weight:.2e}', ha='center', va='bottom', fontsize=8)
        
        for bar, count in zip(bars2, band_counts):
            height = bar.get_height()
            ax4_twin.text(bar.get_x() + bar.get_width()/2., height,
                         f'{count}', ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No data-driven bands\ngenerated', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Data-Driven Bands (None)', fontsize=14, fontweight='bold')
    
    # ==============================================================================
    # 5. Top Dominant Frequencies Table
    # ==============================================================================
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.axis('tight')
    ax5.axis('off')
    
    # Create table of top frequencies
    table_data = []
    
    # Add spectrum peaks
    for i, peak in enumerate(dominant_frequencies['spectrum_peaks'][:15]):
        table_data.append([
            f"Spectrum {i+1}",
            f"{peak['frequency']:.6f}",
            f"{peak['period']:.1f}",
            f"{peak['avg_magnitude']:.2e}",
            f"{peak['energy_weighted_magnitude']:.2e}"
        ])
    
    # Add common individual peaks
    for i, peak in enumerate(dominant_frequencies['common_peaks'][:10]):
        table_data.append([
            f"Individual {i+1}",
            f"{peak['frequency']:.6f}",
            f"{peak['period']:.1f}",
            f"{peak['avg_magnitude']:.2e}",
            f"{peak['count']}"
        ])
    
    if table_data:
        df_table = pd.DataFrame(table_data, 
                              columns=['Type', 'Frequency (cyc/min)', 'Period (min)', 
                                     'Avg Magnitude', 'Weight/Count'])
        
        table = ax5.table(cellText=df_table.values,
                         colLabels=df_table.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
        
        # Color code by type
        for i, row in enumerate(df_table.values):
            if 'Spectrum' in row[0]:
                table[(i+1, 0)].set_facecolor('#E8F4F8')
            else:
                table[(i+1, 0)].set_facecolor('#F8E8E8')
    
    ax5.set_title('Top Dominant Frequencies', fontsize=14, fontweight='bold')
    
    # ==============================================================================
    # 6. Statistical Summary
    # ==============================================================================
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    # Create statistical summary
    individual_periods = [frequency_to_period(p['frequency']) for p in spectrum_data['frequency_peaks']]
    
    stats_text = f"FREQUENCY ANALYSIS STATISTICS\n"
    stats_text += f"{'='*35}\n\n"
    stats_text += f"Dataset Size:\n"
    stats_text += f"  Total Signatures: {spectrum_data['total_signatures']:,}\n"
    stats_text += f"  Significant Peaks: {len(individual_periods):,}\n"
    stats_text += f"  Analysis Range: {MIN_PERIOD_MINUTES}-{MAX_PERIOD_MINUTES} min\n\n"
    
    if individual_periods:
        stats_text += f"Period Distribution:\n"
        stats_text += f"  Mean: {np.mean(individual_periods):.1f} min\n"
        stats_text += f"  Median: {np.median(individual_periods):.1f} min\n"
        stats_text += f"  Std Dev: {np.std(individual_periods):.1f} min\n"
        stats_text += f"  Range: {np.min(individual_periods):.1f}-{np.max(individual_periods):.1f} min\n\n"
    
    stats_text += f"Clustering Results:\n"
    stats_text += f"  Number of Clusters: {len(clustering_results.get('clusters', []))}\n"
    stats_text += f"  Data-Driven Bands: {len(data_driven_bands)}\n"
    stats_text += f"  Method: {clustering_results.get('method', 'none')}\n\n"
    
    if data_driven_bands:
        stats_text += f"Band Centers (periods):\n"
        for band in data_driven_bands:
            stats_text += f"  {band['display_name']}: {band['center_period']:.1f}min\n"
    
    ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # ==============================================================================
    # 7. Comparison with Hard-coded Bands
    # ==============================================================================
    ax7 = fig.add_subplot(gs[3, :])
    
    # Show comparison between hard-coded and data-driven bands
    hardcoded_bands = [
        {'name': 'High (1-5min)', 'range': (1, 5), 'color': '#C73E1D'},
        {'name': 'Medium (15-30min)', 'range': (15, 30), 'color': '#F18F01'},
        {'name': 'Low (60-90min)', 'range': (60, 90), 'color': '#2E86AB'}
    ]
    
    # Plot spectrum again for comparison
    ax7.plot(period_bins, avg_spectrum, color='gray', linewidth=1, alpha=0.6, label='Average Spectrum')
    
    # Show hard-coded bands
    for band in hardcoded_bands:
        min_p, max_p = band['range']
        ax7.axvspan(min_p, max_p, alpha=0.2, color=band['color'], 
                   label=f"Hard-coded: {band['name']}")
    
    # Show data-driven bands
    for i, band in enumerate(data_driven_bands):
        min_period, max_period = band['period_range']
        color = plt.cm.Set3(i / max(1, len(data_driven_bands) - 1))
        ax7.axvspan(min_period, max_period, alpha=0.3, color=color, 
                   label=f"Data-driven: {band['display_name']}")
    
    ax7.set_xlabel('Period (minutes)', fontsize=14)
    ax7.set_ylabel('Average Magnitude', fontsize=14)
    ax7.set_title('Hard-coded vs Data-Driven Frequency Bands', fontsize=16, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=10, ncol=2)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(MIN_PERIOD_MINUTES, MAX_PERIOD_MINUTES)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive analysis plot saved to: {save_path}")
    
    plt.show()

# ==============================================================================
# Export Functions
# ==============================================================================

def export_results(spectrum_data: Dict, dominant_frequencies: Dict, 
                  clustering_results: Dict, data_driven_bands: List[Dict]):
    """Export analysis results for use in prediction models"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export comprehensive results
    results = {
        'analysis_info': {
            'timestamp': timestamp,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_signatures_analyzed': spectrum_data['total_signatures'],
            'frequency_range': (MIN_FREQUENCY, MAX_FREQUENCY),
            'period_range': (MIN_PERIOD_MINUTES, MAX_PERIOD_MINUTES),
            'analysis_parameters': {
                'batch_size': BATCH_SIZE,
                'top_n_frequencies': TOP_N_FREQUENCIES,
                'num_clusters': NUM_CLUSTERS,
                'cluster_method': CLUSTER_METHOD,
                'min_energy_threshold': MIN_ENERGY_THRESHOLD
            }
        },
        'spectrum_analysis': {
            'frequency_bins': spectrum_data['freq_bins'].tolist(),
            'period_bins': [frequency_to_period(f) for f in spectrum_data['freq_bins']],
            'average_spectrum': spectrum_data['avg_spectrum'].tolist(),
            'energy_weighted_spectrum': spectrum_data['energy_weighted_spectrum'].tolist(),
            'total_energy': spectrum_data['total_energy']
        },
        'dominant_frequencies': {
            'spectrum_peaks': dominant_frequencies['spectrum_peaks'],
            'common_peaks': dominant_frequencies['common_peaks'],
            'energy_peaks': dominant_frequencies['energy_peaks'],
            'individual_peaks_summary': {
                'total_count': len(dominant_frequencies['all_individual_peaks']),
                'period_statistics': {
                    'mean': np.mean([frequency_to_period(p['frequency']) for p in dominant_frequencies['all_individual_peaks']]),
                    'median': np.median([frequency_to_period(p['frequency']) for p in dominant_frequencies['all_individual_peaks']]),
                    'std': np.std([frequency_to_period(p['frequency']) for p in dominant_frequencies['all_individual_peaks']]),
                    'min': np.min([frequency_to_period(p['frequency']) for p in dominant_frequencies['all_individual_peaks']]),
                    'max': np.max([frequency_to_period(p['frequency']) for p in dominant_frequencies['all_individual_peaks']])
                }
            }
        },
        'clustering_results': {
            'method': clustering_results.get('method', 'none'),
            'num_clusters': len(clustering_results.get('clusters', [])),
            'clusters': clustering_results.get('clusters', []),
            'cluster_centers_frequency': clustering_results.get('cluster_centers', []),
            'cluster_centers_period': [frequency_to_period(f) for f in clustering_results.get('cluster_centers', [])]
        },
        'data_driven_bands': data_driven_bands,
        'recommended_usage': {
            'description': 'Use these data-driven bands instead of hard-coded frequency ranges',
            'implementation_note': 'Replace FREQ_BANDS in prediction models with these discovered bands',
            'band_format': 'Each band contains frequency_range (min_freq, max_freq) for direct use'
        }
    }
    
    # Save comprehensive results
    results_file = ANALYSIS_DIR / f"frequency_analysis_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Comprehensive results exported to: {results_file}")
    
    # Export simplified band configuration for easy integration
    band_config = {
        'DATA_DRIVEN_FREQ_BANDS': {
            band['band_name']: band['frequency_range'] for band in data_driven_bands
        },
        'BAND_DISPLAY_NAMES': {
            band['band_name']: band['display_name'] for band in data_driven_bands
        },
        'BAND_CENTERS': {
            band['band_name']: band['center_frequency'] for band in data_driven_bands
        },
        'GENERATION_INFO': {
            'timestamp': timestamp,
            'num_signatures': spectrum_data['total_signatures'],
            'method': 'data_driven_clustering'
        }
    }
    
    config_file = ANALYSIS_DIR / f"data_driven_frequency_bands_{timestamp}.json"
    with open(config_file, 'w') as f:
        json.dump(band_config, f, indent=2)
    
    print(f"⚙️  Band configuration exported to: {config_file}")
    
    # Export Python code for direct integration
    python_code = f"""# Data-Driven Frequency Bands
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Based on analysis of {spectrum_data['total_signatures']:,} FFT signatures

DATA_DRIVEN_FREQ_BANDS = {{
"""
    
    for band in data_driven_bands:
        min_freq, max_freq = band['frequency_range']
        python_code += f"    '{band['band_name']}': ({min_freq:.8f}, {max_freq:.8f}),  # {band['display_name']}\n"
    
    python_code += "}\n\n"
    python_code += "# Band display names for plotting and analysis\n"
    python_code += "BAND_DISPLAY_NAMES = {\n"
    
    for band in data_driven_bands:
        python_code += f"    '{band['band_name']}': '{band['display_name']}',\n"
    
    python_code += "}\n"
    
    code_file = ANALYSIS_DIR / f"frequency_bands_config_{timestamp}.py"
    with open(code_file, 'w') as f:
        f.write(python_code)
    
    print(f"🐍 Python configuration exported to: {code_file}")
    
    return results_file, config_file, code_file

# ==============================================================================
# Main Analysis Function
# ==============================================================================

def run_comprehensive_frequency_analysis():
    """Run the complete frequency analysis pipeline on full time series"""
    
    print("🔮 Comprehensive Frequency Analysis - ES Futures Database (Full Time Series)")
    print("=" * 80)
    print(f"🎯 Analyzing frequencies in {MIN_PERIOD_MINUTES}-{MAX_PERIOD_MINUTES} minute range")
    print(f"📊 Processing FULL time series instead of 416-minute windows")
    print(f"🔍 Processing in batches of {BATCH_SIZE} unique CSV files")
    print()
    
    # Load database metadata
    try:
        metadata = load_database_metadata()
        total_signatures = metadata['database_info']['total_signatures']
        print(f"📊 Database: {total_signatures:,} total signatures")
        print(f"📅 Generated: {metadata['generation_info']['processing_date']}")
        print()
    except Exception as e:
        print(f"⚠️ Could not load metadata: {e}")
        return
    
    # Get all signature files, but we'll extract unique source CSV files
    try:
        signature_files = get_all_signature_files()
        print(f"📁 Found {len(signature_files):,} signature files")
        
        # Extract unique source CSV files to avoid duplicate processing
        unique_csv_files = set()
        for sig_file in signature_files:
            signature_name = sig_file.stem  # e.g., "ES_Apr_2024_01_0000"
            source_csv_name = "_".join(signature_name.split("_")[:-1]) + ".csv"  # e.g., "ES_Apr_2024_01.csv"
            unique_csv_files.add(source_csv_name)
        
        unique_csv_files = sorted(list(unique_csv_files))
        print(f"📊 Will analyze {len(unique_csv_files):,} unique CSV files (avoiding duplicates)")
        
        # Create pseudo signature file list based on unique CSVs
        signature_files = [SIGNATURES_DIR / f"{csv_name.replace('.csv', '_0000.csv')}" for csv_name in unique_csv_files]
        
    except Exception as e:
        print(f"❌ Error loading signature files: {e}")
        return
    
    # Process signatures in batches
    print(f"\n🔄 Processing signatures in batches...")
    
    all_spectrum_data = {
        'freq_bins': None,
        'spectral_accumulator': None,
        'energy_weighted_accumulator': None,
        'frequency_peaks': [],
        'total_signatures': 0,
        'total_energy': 0
    }
    
    num_batches = (len(signature_files) + BATCH_SIZE - 1) // BATCH_SIZE
    
    # Overall progress bar for batches
    batch_progress = tqdm(range(num_batches), desc="🔄 Processing batches", unit="batch")
    
    for batch_idx in batch_progress:
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(signature_files))
        
        batch_progress.set_description(f"🔄 Batch {batch_idx + 1}/{num_batches}")
        batch_progress.set_postfix({
            'signatures': f'{start_idx:,}-{end_idx:,}',
            'total_processed': all_spectrum_data['total_signatures'],
            'total_peaks': len(all_spectrum_data['frequency_peaks'])
        })
        
        try:
            # Load batch of full time series (avoid duplicates by using unique source files)
            batch_signatures = load_full_timeseries_batch(signature_files, start_idx, BATCH_SIZE, show_progress=True)
            
            if not batch_signatures:
                print("⚠️ No valid signatures in batch, skipping...")
                continue
            
            # Analyze batch
            batch_spectrum_data = analyze_frequency_spectrum(batch_signatures, show_progress=True)
            
            # Accumulate results
            if all_spectrum_data['freq_bins'] is None:
                # Initialize accumulators
                all_spectrum_data['freq_bins'] = batch_spectrum_data['freq_bins']
                all_spectrum_data['spectral_accumulator'] = batch_spectrum_data['avg_spectrum'] * batch_spectrum_data['total_signatures']
                all_spectrum_data['energy_weighted_accumulator'] = batch_spectrum_data['energy_weighted_spectrum'] * batch_spectrum_data['total_energy']
            else:
                # Accumulate
                all_spectrum_data['spectral_accumulator'] += batch_spectrum_data['avg_spectrum'] * batch_spectrum_data['total_signatures']
                all_spectrum_data['energy_weighted_accumulator'] += batch_spectrum_data['energy_weighted_spectrum'] * batch_spectrum_data['total_energy']
            
            # Accumulate peaks and totals
            all_spectrum_data['frequency_peaks'].extend(batch_spectrum_data['frequency_peaks'])
            all_spectrum_data['total_signatures'] += batch_spectrum_data['total_signatures']
            all_spectrum_data['total_energy'] += batch_spectrum_data['total_energy']
            
            # Update progress bar with latest stats
            batch_progress.set_postfix({
                'signatures': f'{start_idx:,}-{end_idx:,}',
                'total_processed': all_spectrum_data['total_signatures'],
                'total_peaks': len(all_spectrum_data['frequency_peaks']),
                'batch_peaks': len(batch_spectrum_data['frequency_peaks'])
            })
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_idx + 1}: {e}")
            continue
    
    batch_progress.close()
    
    # Finalize accumulated data
    if all_spectrum_data['total_signatures'] > 0:
        all_spectrum_data['avg_spectrum'] = all_spectrum_data['spectral_accumulator'] / all_spectrum_data['total_signatures']
        all_spectrum_data['energy_weighted_spectrum'] = all_spectrum_data['energy_weighted_accumulator'] / all_spectrum_data['total_energy'] if all_spectrum_data['total_energy'] > 0 else all_spectrum_data['avg_spectrum']
    else:
        print("❌ No signatures processed successfully")
        return
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Total signatures analyzed: {all_spectrum_data['total_signatures']:,}")
    print(f"🎯 Total frequency peaks found: {len(all_spectrum_data['frequency_peaks']):,}")
    
    # Analysis pipeline with progress tracking
    analysis_steps = [
        ("🔍 Finding dominant frequencies", lambda: find_dominant_frequencies(all_spectrum_data)),
        ("🔗 Clustering frequencies", lambda: None),  # Will be set after dominant_frequencies
        ("🎯 Generating data-driven bands", lambda: None),  # Will be set after clustering
        ("🎨 Creating visualization", lambda: None),  # Will be set after bands
        ("📤 Exporting results", lambda: None)  # Will be set after visualization
    ]
    
    print(f"\n🔄 Running analysis pipeline...")
    
    # Step 1: Find dominant frequencies
    print(f"\n🔍 Finding dominant frequencies...")
    with tqdm(total=1, desc="Finding dominant frequencies", unit="step") as pbar:
        dominant_frequencies = find_dominant_frequencies(all_spectrum_data)
        pbar.update(1)
    
    # Step 2: Cluster frequencies
    print(f"\n🔗 Clustering frequencies...")
    with tqdm(total=1, desc="Clustering frequencies", unit="step") as pbar:
        clustering_results = cluster_frequencies(dominant_frequencies)
        pbar.update(1)
    
    # Step 3: Generate data-driven bands
    print(f"\n🎯 Generating data-driven frequency bands...")
    with tqdm(total=1, desc="Generating bands", unit="step") as pbar:
        data_driven_bands = generate_data_driven_bands(clustering_results)
        pbar.update(1)
    
    # Step 4: Create comprehensive visualization
    print(f"\n🎨 Creating comprehensive visualization...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = ANALYSIS_DIR / f"comprehensive_frequency_analysis_{timestamp}.png"
    
    with tqdm(total=1, desc="Creating visualization", unit="step") as pbar:
        create_comprehensive_frequency_analysis_plot(
            all_spectrum_data, dominant_frequencies, clustering_results, 
            data_driven_bands, plot_path
        )
        pbar.update(1)
    
    # Step 5: Export results
    print(f"\n📤 Exporting results...")
    with tqdm(total=1, desc="Exporting results", unit="step") as pbar:
        export_results(all_spectrum_data, dominant_frequencies, clustering_results, data_driven_bands)
        pbar.update(1)
    
    print(f"\n🎉 Comprehensive frequency analysis complete!")
    print(f"📁 All results saved to: {ANALYSIS_DIR}")
    
    # Print summary of discovered bands
    if data_driven_bands:
        print(f"\n🎯 DISCOVERED DATA-DRIVEN FREQUENCY BANDS:")
        print("=" * 50)
        for band in data_driven_bands:
            min_period, max_period = band['period_range']
            print(f"  {band['display_name']:<20}: {min_period:6.1f}-{max_period:6.1f} min  ({band['num_source_frequencies']:3d} frequencies)")
        print()
        print("💡 Use these bands instead of hard-coded frequency ranges in your prediction models!")
    else:
        print("\n⚠️ No data-driven bands could be generated")

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover dominant frequencies across FFT signature database')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                      help=f'Batch size for processing signatures (default: {BATCH_SIZE})')
    parser.add_argument('--min-period', type=int, default=MIN_PERIOD_MINUTES,
                      help=f'Minimum period in minutes (default: {MIN_PERIOD_MINUTES})')
    parser.add_argument('--max-period', type=int, default=MAX_PERIOD_MINUTES,
                      help=f'Maximum period in minutes (default: {MAX_PERIOD_MINUTES})')
    parser.add_argument('--clusters', type=int, default=NUM_CLUSTERS,
                      help=f'Number of frequency clusters (default: {NUM_CLUSTERS})')
    parser.add_argument('--top-n', type=int, default=TOP_N_FREQUENCIES,
                      help=f'Number of top frequencies to analyze (default: {TOP_N_FREQUENCIES})')
    
    args = parser.parse_args()
    
    # Update global parameters
    BATCH_SIZE = args.batch_size
    MIN_PERIOD_MINUTES = args.min_period
    MAX_PERIOD_MINUTES = args.max_period
    MIN_FREQUENCY = 1.0 / MAX_PERIOD_MINUTES
    MAX_FREQUENCY = 1.0 / MIN_PERIOD_MINUTES
    NUM_CLUSTERS = args.clusters
    TOP_N_FREQUENCIES = args.top_n
    
    try:
        run_comprehensive_frequency_analysis()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        print("✅ Partial results may have been saved")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
