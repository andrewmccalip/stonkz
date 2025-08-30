#!/usr/bin/env python3
"""
FFT Database Builder - Generate FFT signatures from ES time series data

This script processes all CSV files in datasets/ES/ and generates FFT signatures
for pattern matching. Each signature contains:
- 416-minute context window (sliding backwards from end of day)
- FFT magnitude and phase spectra
- Corresponding 96-minute outcome data
- Metadata about dominant frequencies in target bands

Key Features:
- Prioritizes end-of-day data over overnight periods
- Focuses on 1-5min, 15-30min, and 60-90min frequency patterns
- Generates overlapping sliding windows for comprehensive coverage
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project paths
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

# ==============================================================================
# Configuration
# ==============================================================================

# Data parameters (matching TimesFM configuration)
CONTEXT_LENGTH = 416    # Historical context in minutes (~6.9 hours)
HORIZON_LENGTH = 96     # Prediction horizon in minutes (~1.6 hours)

# Directory paths
DATASETS_DIR = SCRIPT_DIR / "datasets" / "ES"
FFT_DIR = SCRIPT_DIR / "datasets" / "fft"
SIGNATURES_DIR = FFT_DIR / "signatures"
METADATA_DIR = FFT_DIR / "metadata"

# FFT Configuration
SAMPLING_RATE = 1.0  # 1 sample per minute
NYQUIST_FREQ = SAMPLING_RATE / 2  # 0.5 cycles per minute

# Target frequency bands (in cycles per minute)
FREQ_BANDS = {
    'high': (1/5, 1/1),      # 1-5 minute patterns (0.2 to 1.0 cycles/min)
    'medium': (1/30, 1/15),  # 15-30 minute patterns (0.033 to 0.067 cycles/min) 
    'low': (1/90, 1/60)      # 60-90 minute patterns (0.011 to 0.017 cycles/min)
}

# Processing parameters
MIN_SEQUENCE_LENGTH = CONTEXT_LENGTH + HORIZON_LENGTH + 1  # 513 minutes minimum
OVERLAP_MINUTES = 60  # Overlap between sliding windows (1 hour)
END_OF_DAY_PRIORITY_HOURS = [14, 15, 16, 20, 21]  # 2-4 PM and 8-9 PM ET (market close periods)

# ==============================================================================
# Utility Functions
# ==============================================================================

def get_frequency_bins(n_samples: int) -> np.ndarray:
    """
    Get frequency bins for FFT output.
    
    Args:
        n_samples: Number of samples in time series
    
    Returns:
        Array of frequency bins in cycles per minute
    """
    return np.fft.fftfreq(n_samples, d=1.0)[:n_samples//2]  # Only positive frequencies

def find_dominant_frequencies(fft_magnitude: np.ndarray, freq_bins: np.ndarray, 
                            band_range: Tuple[float, float], top_k: int = 3) -> List[float]:
    """
    Find dominant frequencies within a specific band.
    
    Args:
        fft_magnitude: Magnitude spectrum from FFT
        freq_bins: Frequency bins corresponding to FFT
        band_range: (min_freq, max_freq) in cycles per minute
        top_k: Number of top frequencies to return
    
    Returns:
        List of dominant frequencies in the band
    """
    min_freq, max_freq = band_range
    
    # Find indices within the frequency band
    band_mask = (freq_bins >= min_freq) & (freq_bins <= max_freq)
    
    if not np.any(band_mask):
        return []
    
    # Get magnitudes in this band
    band_magnitudes = fft_magnitude[band_mask]
    band_frequencies = freq_bins[band_mask]
    
    # Find top-k peaks
    top_indices = np.argsort(band_magnitudes)[-top_k:][::-1]  # Descending order
    dominant_freqs = band_frequencies[top_indices].tolist()
    
    return dominant_freqs

def is_market_hours(timestamp: pd.Timestamp) -> bool:
    """
    Check if timestamp is during market hours (9:30 AM - 4:00 PM ET).
    
    Args:
        timestamp: Pandas timestamp
    
    Returns:
        True if during market hours
    """
    hour = timestamp.hour
    minute = timestamp.minute
    
    # Market opens at 9:30 AM ET, closes at 4:00 PM ET
    if hour < 9 or hour > 16:
        return False
    if hour == 9 and minute < 30:
        return False
    if hour == 16 and minute > 0:
        return False
    
    return True

def get_priority_score(timestamp: pd.Timestamp) -> float:
    """
    Get priority score for a timestamp (higher = more important).
    Prioritizes end-of-day periods and market hours.
    
    Args:
        timestamp: Pandas timestamp
    
    Returns:
        Priority score (0.0 to 1.0)
    """
    base_score = 0.5
    
    # Boost for market hours
    if is_market_hours(timestamp):
        base_score += 0.3
    
    # Extra boost for end-of-day priority hours
    if timestamp.hour in END_OF_DAY_PRIORITY_HOURS:
        base_score += 0.2
    
    return min(base_score, 1.0)

# ==============================================================================
# FFT Processing Functions
# ==============================================================================

def compute_fft_signature(data: np.ndarray) -> Dict:
    """
    Compute FFT signature for a time series segment.
    
    Args:
        data: Time series data (should be 416 points)
    
    Returns:
        Dictionary containing FFT signature components
    """
    # Ensure we have the right length
    if len(data) != CONTEXT_LENGTH:
        raise ValueError(f"Expected {CONTEXT_LENGTH} data points, got {len(data)}")
    
    # Apply window function to reduce spectral leakage
    windowed_data = data * np.hanning(len(data))
    
    # Compute FFT (full complex spectrum)
    fft_complex = np.fft.fft(windowed_data)
    
    # Extract magnitude and phase
    fft_magnitude = np.abs(fft_complex)
    fft_phase = np.angle(fft_complex)
    
    # Get frequency bins
    freq_bins = get_frequency_bins(len(data))
    
    # Find dominant frequencies in each target band
    dominant_freqs = {}
    for band_name, band_range in FREQ_BANDS.items():
        dominant_freqs[f'dominant_freq_{band_name}'] = find_dominant_frequencies(
            fft_magnitude[:len(freq_bins)], freq_bins, band_range
        )
    
    # Calculate spectral energy in each band
    spectral_energy = {}
    for band_name, band_range in FREQ_BANDS.items():
        min_freq, max_freq = band_range
        band_mask = (freq_bins >= min_freq) & (freq_bins <= max_freq)
        band_energy = np.sum(fft_magnitude[:len(freq_bins)][band_mask] ** 2)
        spectral_energy[f'energy_{band_name}'] = band_energy
    
    return {
        'fft_magnitude': fft_magnitude,
        'fft_phase': fft_phase,
        'freq_bins': freq_bins,
        'dominant_frequencies': dominant_freqs,
        'spectral_energy': spectral_energy,
        'total_energy': np.sum(fft_magnitude ** 2)
    }

def extract_sliding_windows(df: pd.DataFrame, csv_filename: str) -> List[Dict]:
    """
    Extract sliding windows from a CSV file, prioritizing end-of-day data.
    
    Args:
        df: DataFrame with time series data
        csv_filename: Name of the source CSV file
    
    Returns:
        List of window dictionaries with context and outcome data
    """
    windows = []
    
    if len(df) < MIN_SEQUENCE_LENGTH:
        print(f"⚠️ {csv_filename}: Insufficient data ({len(df)} < {MIN_SEQUENCE_LENGTH})")
        return windows
    
    # Convert timestamps if available
    if 'timestamp_pt' in df.columns:
        df = df.copy()
        df['timestamp_pt'] = pd.to_datetime(df['timestamp_pt'])
        df['priority_score'] = df['timestamp_pt'].apply(get_priority_score)
    else:
        # If no timestamp, assign uniform priority
        df = df.copy()
        df['priority_score'] = 0.5
    
    # Generate potential window start positions
    max_start_idx = len(df) - MIN_SEQUENCE_LENGTH
    
    # Create sliding windows working backwards from end
    # This prioritizes more recent data
    window_starts = []
    
    # Start from the end and work backwards
    current_start = max_start_idx
    while current_start >= CONTEXT_LENGTH:
        window_starts.append(current_start)
        current_start -= OVERLAP_MINUTES
    
    # Sort by priority score (if we have timestamps)
    if 'timestamp_pt' in df.columns:
        window_priorities = []
        for start_idx in window_starts:
            context_end_idx = start_idx
            priority = df.iloc[context_end_idx]['priority_score']
            window_priorities.append((priority, start_idx))
        
        # Sort by priority (highest first)
        window_priorities.sort(reverse=True)
        window_starts = [start_idx for _, start_idx in window_priorities]
    
    # Process windows
    for sequence_id, start_idx in enumerate(window_starts):
        try:
            # Define window boundaries
            context_start = start_idx - CONTEXT_LENGTH
            context_end = start_idx
            outcome_start = start_idx
            outcome_end = start_idx + HORIZON_LENGTH
            
            # Extract data
            context_data = df['close'].iloc[context_start:context_end].values
            outcome_data = df['close'].iloc[outcome_start:outcome_end].values
            
            # Validate data
            if len(context_data) != CONTEXT_LENGTH or len(outcome_data) != HORIZON_LENGTH:
                continue
            
            if np.any(np.isnan(context_data)) or np.any(np.isnan(outcome_data)):
                continue
            
            # Get timestamp info
            timestamp_start = df.iloc[context_start].get('timestamp_pt', None)
            timestamp_end = df.iloc[context_end-1].get('timestamp_pt', None)
            
            # Create window record
            window = {
                'sequence_id': f"{Path(csv_filename).stem}_{sequence_id:04d}",
                'source_file': csv_filename,
                'start_idx': context_start,
                'end_idx': context_end,
                'outcome_start_idx': outcome_start,
                'outcome_end_idx': outcome_end,
                'timestamp_start': timestamp_start.isoformat() if timestamp_start else None,
                'timestamp_end': timestamp_end.isoformat() if timestamp_end else None,
                'context_data': context_data,
                'outcome_data': outcome_data,
                'priority_score': df.iloc[context_end-1]['priority_score']
            }
            
            windows.append(window)
            
        except Exception as e:
            print(f"⚠️ Error processing window at index {start_idx} in {csv_filename}: {e}")
            continue
    
    return windows

# ==============================================================================
# Database Generation Functions
# ==============================================================================

def process_csv_file(csv_path: Path) -> Tuple[int, int]:
    """
    Process a single CSV file and generate FFT signatures.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Tuple of (successful_windows, total_windows)
    """
    print(f"📁 Processing {csv_path.name}...")
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        print(f"   📊 Loaded {len(df)} rows")
        
        # Extract sliding windows
        windows = extract_sliding_windows(df, csv_path.name)
        print(f"   🪟 Extracted {len(windows)} windows")
        
        if not windows:
            return 0, 0
        
        successful_windows = 0
        
        # Process each window
        for window in tqdm(windows, desc=f"   Processing {csv_path.stem}", leave=False):
            try:
                # Compute FFT signature
                fft_signature = compute_fft_signature(window['context_data'])
                
                # Create signature record
                signature_record = {
                    # Metadata
                    'sequence_id': window['sequence_id'],
                    'source_file': window['source_file'],
                    'start_idx': window['start_idx'],
                    'end_idx': window['end_idx'],
                    'outcome_start_idx': window['outcome_start_idx'],
                    'outcome_end_idx': window['outcome_end_idx'],
                    'timestamp_start': window['timestamp_start'],
                    'timestamp_end': window['timestamp_end'],
                    'priority_score': window['priority_score'],
                    
                    # Context and outcome data
                    'context_data': window['context_data'].tolist(),
                    'outcome_data': window['outcome_data'].tolist(),
                    
                    # FFT signature data
                    'fft_magnitude': fft_signature['fft_magnitude'].tolist(),
                    'fft_phase': fft_signature['fft_phase'].tolist(),
                    
                    # Dominant frequencies in each band
                    **fft_signature['dominant_frequencies'],
                    
                    # Spectral energy in each band
                    **fft_signature['spectral_energy'],
                    'total_energy': fft_signature['total_energy'],
                    
                    # Processing metadata
                    'processing_timestamp': datetime.now().isoformat(),
                    'context_length': CONTEXT_LENGTH,
                    'horizon_length': HORIZON_LENGTH
                }
                
                # Save signature to CSV
                signature_df = pd.DataFrame([signature_record])
                signature_filename = f"{window['sequence_id']}.csv"
                signature_path = SIGNATURES_DIR / signature_filename
                
                signature_df.to_csv(signature_path, index=False)
                successful_windows += 1
                
            except Exception as e:
                print(f"⚠️ Failed to process window {window['sequence_id']}: {e}")
                continue
        
        print(f"   ✅ Successfully processed {successful_windows}/{len(windows)} windows")
        return successful_windows, len(windows)
        
    except Exception as e:
        print(f"❌ Failed to process {csv_path.name}: {e}")
        return 0, 0

def build_fft_database():
    """
    Build the complete FFT signature database from all CSV files.
    """
    print("🚀 Starting FFT Database Generation")
    print("=" * 60)
    
    # Find all CSV files
    csv_files = list(DATASETS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {DATASETS_DIR}")
        return
    
    print(f"📁 Found {len(csv_files)} CSV files to process")
    
    # Processing statistics
    total_successful = 0
    total_windows = 0
    processed_files = 0
    failed_files = []
    
    start_time = datetime.now()
    
    # Process each CSV file
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            successful, total = process_csv_file(csv_file)
            total_successful += successful
            total_windows += total
            processed_files += 1
            
        except Exception as e:
            print(f"❌ Failed to process {csv_file.name}: {e}")
            failed_files.append(csv_file.name)
            continue
    
    end_time = datetime.now()
    processing_duration = end_time - start_time
    
    # Generate metadata
    metadata = {
        'generation_info': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': processing_duration.total_seconds(),
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'processing_stats': {
            'total_csv_files': len(csv_files),
            'processed_files': processed_files,
            'failed_files': len(failed_files),
            'failed_file_names': failed_files,
            'total_windows_attempted': total_windows,
            'successful_signatures': total_successful,
            'success_rate': (total_successful / total_windows * 100) if total_windows > 0 else 0
        },
        'configuration': {
            'context_length': CONTEXT_LENGTH,
            'horizon_length': HORIZON_LENGTH,
            'overlap_minutes': OVERLAP_MINUTES,
            'frequency_bands': FREQ_BANDS,
            'end_of_day_priority_hours': END_OF_DAY_PRIORITY_HOURS
        },
        'database_info': {
            'signatures_directory': str(SIGNATURES_DIR),
            'total_signatures': total_successful,
            'signature_file_pattern': '{source_csv_stem}_{sequence_id:04d}.csv'
        }
    }
    
    # Save metadata
    metadata_path = METADATA_DIR / "database_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 FFT Database Generation Complete!")
    print("=" * 60)
    print(f"📊 Processing Summary:")
    print(f"   CSV files processed: {processed_files}/{len(csv_files)}")
    print(f"   Total signatures generated: {total_successful:,}")
    print(f"   Success rate: {metadata['processing_stats']['success_rate']:.1f}%")
    print(f"   Processing time: {processing_duration}")
    print(f"📁 Database location: {SIGNATURES_DIR}")
    print(f"📋 Metadata saved to: {metadata_path}")
    
    if failed_files:
        print(f"\n⚠️ Failed files ({len(failed_files)}):")
        for failed_file in failed_files[:10]:  # Show first 10
            print(f"   - {failed_file}")
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more")

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    print("🔮 FFT Database Builder for Time Series Pattern Matching")
    print("=" * 60)
    
    # Verify directories exist
    if not DATASETS_DIR.exists():
        print(f"❌ Datasets directory not found: {DATASETS_DIR}")
        sys.exit(1)
    
    if not FFT_DIR.exists():
        print(f"❌ FFT directory not found: {FFT_DIR}")
        sys.exit(1)
    
    # Start database generation
    try:
        build_fft_database()
        print("\n✅ Database generation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Database generation interrupted by user")
        print("✅ Partial results have been saved")
        
    except Exception as e:
        print(f"\n❌ Database generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
