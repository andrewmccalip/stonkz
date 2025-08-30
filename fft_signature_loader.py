#!/usr/bin/env python3
"""
FFT Signature Loader - Efficient loading and management of FFT signature database

This module provides efficient loading and indexing of the 29,005+ FFT signatures
for fast similarity search. Key features:

1. Memory-efficient loading with optional caching
2. Fast indexing by various criteria (timestamp, priority, source file)
3. Signature validation and filtering
4. Batch loading for memory management
5. Integration with data-driven frequency bands

The loader supports different loading strategies:
- Full database in memory (fast search, high memory ~2-4GB)
- Lazy loading with LRU cache (balanced approach)
- Streaming access (low memory, slower search)
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Iterator
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# Add project paths
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

# Import our data-driven frequency bands
try:
    from datasets.fft.analysis.condensed_frequency_bands_20250830_100049 import FREQ_BANDS, BAND_WEIGHTS
    print("✅ Loaded data-driven frequency bands")
except ImportError:
    # Fallback to discovered bands if import fails
    FREQ_BANDS = {
        'long': (0.008333, 0.012817),        # 78-120 min - MOST IMPORTANT
        'medium': (0.028164, 0.034164),      # 29-35 min   
        'short': (0.040375, 0.060271),       # 17-25 min - VERY IMPORTANT  
        'ultra_short': (0.067000, 0.200000), # 5-15 min - scalping
    }
    BAND_WEIGHTS = {
        'long': 0.50, 'short': 0.30, 'medium': 0.15, 'ultra_short': 0.05
    }
    print("⚠️ Using fallback frequency bands")

# ==============================================================================
# Configuration
# ==============================================================================

# Paths
FFT_DIR = SCRIPT_DIR / "datasets" / "fft"
SIGNATURES_DIR = FFT_DIR / "signatures"
METADATA_DIR = FFT_DIR / "metadata"
CACHE_DIR = FFT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Loading parameters
CONTEXT_LENGTH = 416
HORIZON_LENGTH = 96
DEFAULT_BATCH_SIZE = 1000
CACHE_FILE_PREFIX = "signature_cache"

# Memory management
MAX_MEMORY_SIGNATURES = 50000  # Maximum signatures to keep in memory
ENABLE_DISK_CACHE = True       # Whether to use disk caching

# ==============================================================================
# Data Structures
# ==============================================================================

class FFTSignature:
    """Container for a single FFT signature with efficient access"""
    
    def __init__(self, data_dict: Dict):
        """Initialize from signature data dictionary"""
        
        # Core metadata
        self.sequence_id = data_dict['sequence_id']
        self.source_file = data_dict['source_file']
        self.priority_score = data_dict.get('priority_score', 0.5)
        
        # Temporal information
        self.timestamp_start = data_dict.get('timestamp_start')
        self.timestamp_end = data_dict.get('timestamp_end')
        self.start_idx = data_dict.get('start_idx', 0)
        self.end_idx = data_dict.get('end_idx', 0)
        
        # Time series data
        self.context_data = np.array(data_dict['context_data'])
        self.outcome_data = np.array(data_dict['outcome_data'])
        
        # FFT data
        self.fft_magnitude = np.array(data_dict['fft_magnitude'])
        self.fft_phase = np.array(data_dict['fft_phase'])
        
        # Pre-computed frequency band features using data-driven bands
        self.band_features = self._compute_band_features()
        
        # Energy metrics
        self.total_energy = data_dict.get('total_energy', np.sum(self.fft_magnitude ** 2))
        
        # Processing metadata
        self.processing_timestamp = data_dict.get('processing_timestamp')
    
    def _compute_band_features(self) -> Dict[str, np.ndarray]:
        """Pre-compute frequency band features for fast similarity search"""
        features = {}
        
        # Get frequency bins for this signature
        freq_bins = np.fft.fftfreq(len(self.fft_magnitude), d=1.0)[:len(self.fft_magnitude)//2]
        magnitude_positive = self.fft_magnitude[:len(freq_bins)]
        phase_positive = self.fft_phase[:len(freq_bins)]
        
        # Extract features for each data-driven frequency band
        for band_name, (min_freq, max_freq) in FREQ_BANDS.items():
            # Find frequencies in this band
            band_mask = (freq_bins >= min_freq) & (freq_bins <= max_freq)
            
            if np.any(band_mask):
                # Extract magnitude and phase for this band
                band_magnitude = magnitude_positive[band_mask]
                band_phase = phase_positive[band_mask]
                
                # Compute band features
                features[f'{band_name}_magnitude'] = band_magnitude
                features[f'{band_name}_phase'] = band_phase
                features[f'{band_name}_energy'] = np.sum(band_magnitude ** 2)
                features[f'{band_name}_peak_freq'] = freq_bins[band_mask][np.argmax(band_magnitude)] if len(band_magnitude) > 0 else 0
                features[f'{band_name}_mean_magnitude'] = np.mean(band_magnitude) if len(band_magnitude) > 0 else 0
            else:
                # No frequencies in this band
                features[f'{band_name}_magnitude'] = np.array([])
                features[f'{band_name}_phase'] = np.array([])
                features[f'{band_name}_energy'] = 0.0
                features[f'{band_name}_peak_freq'] = 0.0
                features[f'{band_name}_mean_magnitude'] = 0.0
        
        return features
    
    def get_band_signature(self, band_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get magnitude and phase signature for a specific frequency band"""
        magnitude = self.band_features.get(f'{band_name}_magnitude', np.array([]))
        phase = self.band_features.get(f'{band_name}_phase', np.array([]))
        return magnitude, phase
    
    def get_weighted_signature(self) -> np.ndarray:
        """Get weighted signature combining all bands according to BAND_WEIGHTS"""
        weighted_features = []
        
        for band_name, weight in BAND_WEIGHTS.items():
            magnitude, _ = self.get_band_signature(band_name)
            if len(magnitude) > 0:
                # Weight the magnitude by band importance
                weighted_magnitude = magnitude * weight
                weighted_features.append(weighted_magnitude)
        
        # Concatenate all weighted features
        if weighted_features:
            return np.concatenate(weighted_features)
        else:
            return np.array([])
    
    def __repr__(self):
        return f"FFTSignature({self.sequence_id}, {self.source_file}, priority={self.priority_score:.3f})"

# ==============================================================================
# Database Loader Class
# ==============================================================================

class FFTSignatureDatabase:
    """Efficient loader and manager for FFT signature database"""
    
    def __init__(self, load_strategy: str = 'full', enable_cache: bool = True):
        """
        Initialize the signature database loader.
        
        Args:
            load_strategy: 'full' (all in memory), 'lazy' (on-demand), 'streaming' (minimal memory)
            enable_cache: Whether to use disk caching for faster subsequent loads
        """
        self.load_strategy = load_strategy
        self.enable_cache = enable_cache
        
        # Storage
        self.signatures = {}  # Dict[sequence_id, FFTSignature]
        self.metadata = None
        self.indices = {}     # Various indices for fast lookup
        
        # Statistics
        self.total_signatures = 0
        self.loaded_signatures = 0
        self.memory_usage_mb = 0
        
        # Cache management
        self.cache_file = CACHE_DIR / f"{CACHE_FILE_PREFIX}_{load_strategy}.pkl"
        
        print(f"🔧 Initialized FFT database loader (strategy: {load_strategy}, cache: {enable_cache})")
    
    def load_database_metadata(self) -> Dict:
        """Load database metadata"""
        metadata_path = METADATA_DIR / "database_info.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Database metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.total_signatures = self.metadata['database_info']['total_signatures']
        return self.metadata
    
    def _load_signature_from_file(self, signature_path: Path) -> Optional[FFTSignature]:
        """Load a single signature from CSV file"""
        try:
            df = pd.read_csv(signature_path)
            if len(df) != 1:
                return None
            
            row = df.iloc[0]
            
            # Parse data (handle potential eval issues)
            try:
                data_dict = {
                    'sequence_id': row['sequence_id'],
                    'source_file': row['source_file'],
                    'priority_score': row.get('priority_score', 0.5),
                    'timestamp_start': row.get('timestamp_start'),
                    'timestamp_end': row.get('timestamp_end'),
                    'start_idx': row.get('start_idx', 0),
                    'end_idx': row.get('end_idx', 0),
                    'context_data': eval(row['context_data']),
                    'outcome_data': eval(row['outcome_data']),
                    'fft_magnitude': eval(row['fft_magnitude']),
                    'fft_phase': eval(row['fft_phase']),
                    'total_energy': row.get('total_energy', 0),
                    'processing_timestamp': row.get('processing_timestamp')
                }
                
                return FFTSignature(data_dict)
                
            except Exception as e:
                print(f"⚠️ Failed to parse signature data in {signature_path.name}: {e}")
                return None
                
        except Exception as e:
            print(f"⚠️ Failed to load signature file {signature_path.name}: {e}")
            return None
    
    def load_signatures(self, max_signatures: Optional[int] = None) -> int:
        """
        Load signatures according to the specified strategy.
        
        Args:
            max_signatures: Maximum number of signatures to load (None = all)
        
        Returns:
            Number of signatures successfully loaded
        """
        print(f"📊 Loading FFT signatures (strategy: {self.load_strategy})...")
        
        # Load metadata first
        if self.metadata is None:
            self.load_database_metadata()
        
        # Check for cached data
        if self.enable_cache and self.cache_file.exists():
            print(f"💾 Found cache file: {self.cache_file.name}")
            try:
                return self._load_from_cache()
            except Exception as e:
                print(f"⚠️ Cache loading failed: {e}, proceeding with fresh load")
        
        # Get signature files
        signature_files = list(SIGNATURES_DIR.glob("*.csv"))
        if not signature_files:
            raise FileNotFoundError(f"No signature files found in {SIGNATURES_DIR}")
        
        # Limit if requested
        if max_signatures:
            signature_files = signature_files[:max_signatures]
        
        print(f"📁 Loading {len(signature_files):,} signature files...")
        
        # Load signatures with progress tracking
        loaded_count = 0
        failed_count = 0
        
        # Use different loading strategies
        if self.load_strategy == 'full':
            # Load all signatures into memory
            progress = tqdm(signature_files, desc="Loading signatures", unit="sig")
            
            for sig_file in progress:
                signature = self._load_signature_from_file(sig_file)
                if signature:
                    self.signatures[signature.sequence_id] = signature
                    loaded_count += 1
                else:
                    failed_count += 1
                
                # Update progress
                progress.set_postfix({
                    'loaded': loaded_count,
                    'failed': failed_count,
                    'memory_mb': self._estimate_memory_usage()
                })
        
        elif self.load_strategy == 'lazy':
            # Just index the files, load on demand
            print("📋 Creating lazy loading index...")
            for sig_file in tqdm(signature_files, desc="Indexing"):
                sequence_id = sig_file.stem
                self.signatures[sequence_id] = sig_file  # Store path instead of data
                loaded_count += 1
        
        elif self.load_strategy == 'streaming':
            # Minimal indexing for streaming access
            print("🌊 Creating streaming index...")
            self.signature_files = signature_files
            loaded_count = len(signature_files)
        
        self.loaded_signatures = loaded_count
        
        # Build indices for fast lookup
        if self.load_strategy == 'full':
            self._build_indices()
        
        # Cache if enabled and full loading
        if self.enable_cache and self.load_strategy == 'full' and loaded_count > 0:
            self._save_to_cache()
        
        print(f"\n✅ Signature loading complete!")
        print(f"   Strategy: {self.load_strategy}")
        print(f"   Loaded: {loaded_count:,}")
        print(f"   Failed: {failed_count}")
        print(f"   Memory usage: ~{self._estimate_memory_usage():.1f} MB")
        
        return loaded_count
    
    def _build_indices(self):
        """Build various indices for fast lookup"""
        print("🔍 Building lookup indices...")
        
        self.indices = {
            'by_source_file': defaultdict(list),
            'by_priority': defaultdict(list),
            'by_timestamp': [],
            'high_priority': [],  # Priority > 0.7
            'medium_priority': [],  # Priority 0.4-0.7
            'low_priority': []   # Priority < 0.4
        }
        
        for sequence_id, signature in self.signatures.items():
            if isinstance(signature, FFTSignature):
                # Index by source file
                self.indices['by_source_file'][signature.source_file].append(sequence_id)
                
                # Index by priority
                priority = signature.priority_score
                if priority > 0.7:
                    self.indices['high_priority'].append(sequence_id)
                elif priority > 0.4:
                    self.indices['medium_priority'].append(sequence_id)
                else:
                    self.indices['low_priority'].append(sequence_id)
                
                # Index by timestamp (if available)
                if signature.timestamp_start:
                    self.indices['by_timestamp'].append((signature.timestamp_start, sequence_id))
        
        # Sort timestamp index
        self.indices['by_timestamp'].sort()
        
        print(f"   📋 Built indices for {len(self.signatures):,} signatures")
        print(f"   🎯 High priority: {len(self.indices['high_priority']):,}")
        print(f"   📊 Medium priority: {len(self.indices['medium_priority']):,}")
        print(f"   📉 Low priority: {len(self.indices['low_priority']):,}")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        if self.load_strategy != 'full':
            return 0.1  # Minimal for lazy/streaming
        
        # Rough estimate: each signature ~50KB (context + outcome + FFT data)
        signature_size_kb = 50
        return len(self.signatures) * signature_size_kb / 1024
    
    def _save_to_cache(self):
        """Save loaded signatures to disk cache"""
        try:
            print("💾 Saving to cache...")
            cache_data = {
                'signatures': self.signatures,
                'indices': self.indices,
                'metadata': self.metadata,
                'load_strategy': self.load_strategy,
                'cache_timestamp': datetime.now().isoformat(),
                'total_signatures': self.loaded_signatures
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            cache_size_mb = self.cache_file.stat().st_size / (1024 * 1024)
            print(f"   💾 Cache saved: {cache_size_mb:.1f} MB")
            
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
    
    def _load_from_cache(self) -> int:
        """Load signatures from disk cache"""
        print("📦 Loading from cache...")
        
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.signatures = cache_data['signatures']
            self.indices = cache_data.get('indices', {})
            self.metadata = cache_data.get('metadata')
            self.loaded_signatures = cache_data.get('total_signatures', len(self.signatures))
            
            print(f"   ✅ Loaded {len(self.signatures):,} signatures from cache")
            print(f"   📅 Cache date: {cache_data.get('cache_timestamp', 'unknown')}")
            
            return len(self.signatures)
            
        except Exception as e:
            raise Exception(f"Cache loading failed: {e}")
    
    def get_signature(self, sequence_id: str) -> Optional[FFTSignature]:
        """Get a signature by sequence ID"""
        if self.load_strategy == 'full':
            return self.signatures.get(sequence_id)
        
        elif self.load_strategy == 'lazy':
            # Load on demand
            if sequence_id in self.signatures:
                sig_path = self.signatures[sequence_id]
                if isinstance(sig_path, Path):
                    # Load from file
                    signature = self._load_signature_from_file(sig_path)
                    if signature:
                        self.signatures[sequence_id] = signature  # Cache in memory
                    return signature
                else:
                    return sig_path  # Already loaded
        
        return None
    
    def get_signatures_by_priority(self, priority_level: str = 'high') -> List[FFTSignature]:
        """Get signatures filtered by priority level"""
        if priority_level not in ['high', 'medium', 'low']:
            raise ValueError("Priority level must be 'high', 'medium', or 'low'")
        
        sequence_ids = self.indices.get(f'{priority_level}_priority', [])
        return [self.get_signature(seq_id) for seq_id in sequence_ids if self.get_signature(seq_id)]
    
    def get_random_signatures(self, n: int = 10, priority_filter: Optional[str] = None) -> List[FFTSignature]:
        """Get random signatures for testing/analysis"""
        if priority_filter:
            candidate_ids = self.indices.get(f'{priority_filter}_priority', [])
        else:
            candidate_ids = list(self.signatures.keys())
        
        if len(candidate_ids) < n:
            n = len(candidate_ids)
        
        import random
        selected_ids = random.sample(candidate_ids, n)
        return [self.get_signature(seq_id) for seq_id in selected_ids if self.get_signature(seq_id)]
    
    def iter_signatures(self, batch_size: int = DEFAULT_BATCH_SIZE, 
                       priority_filter: Optional[str] = None) -> Iterator[List[FFTSignature]]:
        """Iterate over signatures in batches"""
        if priority_filter:
            sequence_ids = self.indices.get(f'{priority_filter}_priority', [])
        else:
            sequence_ids = list(self.signatures.keys())
        
        # Yield batches
        for i in range(0, len(sequence_ids), batch_size):
            batch_ids = sequence_ids[i:i + batch_size]
            batch_signatures = []
            
            for seq_id in batch_ids:
                signature = self.get_signature(seq_id)
                if signature:
                    batch_signatures.append(signature)
            
            if batch_signatures:
                yield batch_signatures
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        if self.load_strategy == 'full':
            # Calculate detailed stats
            priorities = [sig.priority_score for sig in self.signatures.values() if isinstance(sig, FFTSignature)]
            energies = [sig.total_energy for sig in self.signatures.values() if isinstance(sig, FFTSignature)]
            
            stats = {
                'total_signatures': len(self.signatures),
                'load_strategy': self.load_strategy,
                'memory_usage_mb': self._estimate_memory_usage(),
                'priority_stats': {
                    'mean': np.mean(priorities) if priorities else 0,
                    'std': np.std(priorities) if priorities else 0,
                    'min': np.min(priorities) if priorities else 0,
                    'max': np.max(priorities) if priorities else 0
                },
                'energy_stats': {
                    'mean': np.mean(energies) if energies else 0,
                    'std': np.std(energies) if energies else 0,
                    'min': np.min(energies) if energies else 0,
                    'max': np.max(energies) if energies else 0
                },
                'indices_available': list(self.indices.keys()) if hasattr(self, 'indices') else []
            }
        else:
            # Basic stats for lazy/streaming
            stats = {
                'total_signatures': self.loaded_signatures,
                'load_strategy': self.load_strategy,
                'memory_usage_mb': self._estimate_memory_usage(),
                'indices_available': []
            }
        
        return stats
    
    def clear_cache(self):
        """Clear disk cache"""
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"🗑️ Cache cleared: {self.cache_file.name}")
        else:
            print("ℹ️ No cache file to clear")

# ==============================================================================
# Convenience Functions
# ==============================================================================

def load_fft_database(strategy: str = 'full', max_signatures: Optional[int] = None, 
                     enable_cache: bool = True) -> FFTSignatureDatabase:
    """
    Convenience function to load the FFT signature database.
    
    Args:
        strategy: Loading strategy ('full', 'lazy', 'streaming')
        max_signatures: Maximum signatures to load (None = all)
        enable_cache: Whether to use disk caching
    
    Returns:
        Loaded FFTSignatureDatabase instance
    """
    print(f"🚀 Loading FFT Signature Database")
    print(f"   Strategy: {strategy}")
    print(f"   Max signatures: {max_signatures or 'all'}")
    print(f"   Caching: {enable_cache}")
    
    db = FFTSignatureDatabase(load_strategy=strategy, enable_cache=enable_cache)
    loaded_count = db.load_signatures(max_signatures)
    
    print(f"✅ Database loaded: {loaded_count:,} signatures")
    return db

def test_database_loading():
    """Test the database loading functionality"""
    print("🧪 Testing FFT Database Loading")
    print("=" * 50)
    
    # Test different loading strategies
    strategies = ['full']  # Start with full loading
    
    for strategy in strategies:
        print(f"\n📊 Testing {strategy} loading strategy...")
        
        try:
            # Load small subset for testing
            db = load_fft_database(strategy=strategy, max_signatures=100, enable_cache=False)
            
            # Get statistics
            stats = db.get_database_stats()
            print(f"   Loaded: {stats['total_signatures']:,} signatures")
            print(f"   Memory: {stats['memory_usage_mb']:.1f} MB")
            
            # Test random access
            random_sigs = db.get_random_signatures(3)
            print(f"   Random signatures: {len(random_sigs)}")
            
            for sig in random_sigs:
                print(f"     {sig.sequence_id}: {sig.source_file} (priority: {sig.priority_score:.3f})")
            
            print(f"   ✅ {strategy} strategy test passed")
            
        except Exception as e:
            print(f"   ❌ {strategy} strategy test failed: {e}")
    
    print(f"\n🎉 Database loading tests complete!")

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FFT Signature Database Loader')
    parser.add_argument('--strategy', type=str, default='full',
                      choices=['full', 'lazy', 'streaming'],
                      help='Loading strategy')
    parser.add_argument('--max-signatures', type=int, default=None,
                      help='Maximum signatures to load')
    parser.add_argument('--no-cache', action='store_true',
                      help='Disable disk caching')
    parser.add_argument('--test', action='store_true',
                      help='Run loading tests')
    parser.add_argument('--clear-cache', action='store_true',
                      help='Clear existing cache')
    
    args = parser.parse_args()
    
    try:
        if args.clear_cache:
            db = FFTSignatureDatabase()
            db.clear_cache()
        
        elif args.test:
            test_database_loading()
        
        else:
            # Load database
            db = load_fft_database(
                strategy=args.strategy,
                max_signatures=args.max_signatures,
                enable_cache=not args.no_cache
            )
            
            # Show stats
            stats = db.get_database_stats()
            print(f"\n📊 Database Statistics:")
            print(f"   Total signatures: {stats['total_signatures']:,}")
            print(f"   Memory usage: {stats['memory_usage_mb']:.1f} MB")
            print(f"   Strategy: {stats['load_strategy']}")
            
            if 'priority_stats' in stats:
                print(f"   Priority range: {stats['priority_stats']['min']:.3f} - {stats['priority_stats']['max']:.3f}")
                print(f"   Energy range: {stats['energy_stats']['min']:.2e} - {stats['energy_stats']['max']:.2e}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Loading interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
