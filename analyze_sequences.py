#!/usr/bin/env python3
"""
Analyze the number of sequences in the fine-tuning dataset.
"""
import pickle
import hashlib
from pathlib import Path
import pandas as pd

# Configuration (copied from pytorch_timesfm_finetune.py)
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv"
CACHE_DIR = SCRIPT_DIR / "dataset_cache"
CONTEXT_LENGTH = 448
HORIZON_LENGTH = 64
INPUT_PATCH_LEN = 32
SEQUENCE_STRIDE = 64
MAX_SEQUENCES = None
CACHE_VERSION = "v2"

def analyze_cached_sequences():
    """Analyze sequences from cached data if available."""
    # Create cache key (same as in main script)
    cache_key_str = f"{DATASET_PATH}_{CONTEXT_LENGTH}_{HORIZON_LENGTH}_{INPUT_PATCH_LEN}_{SEQUENCE_STRIDE}_{MAX_SEQUENCES}_{CACHE_VERSION}"
    cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
    cache_file = CACHE_DIR / f"pytorch_data_{cache_key}.pkl"
    
    if cache_file.exists():
        print("📦 Loading cached data for analysis...")
        print(f"   Cache file: {cache_file.name}")
        
        with open(cache_file, 'rb') as f:
            data_dict = pickle.load(f)
        
        # Analyze each dataset
        datasets = {
            'Train': data_dict['train_dataset'],
            'Validation': data_dict['val_dataset'], 
            'Test': data_dict['test_dataset']
        }
        
        print("\n" + "="*60)
        print("📊 SEQUENCE ANALYSIS")
        print("="*60)
        
        total_sequences = 0
        for name, dataset in datasets.items():
            num_sequences = len(dataset)
            total_sequences += num_sequences
            
            print(f"\n{name} Dataset:")
            print(f"  Total sequences: {num_sequences:,}")
            
            # Analyze unique instruments
            unique_instruments = set()
            unique_symbols = set()
            sequence_info = []
            
            for i in range(min(num_sequences, 1000)):  # Sample first 1000 for analysis
                seq_info = dataset.sequence_to_price_idx[i]
                unique_instruments.add(seq_info['instrument_id'])
                unique_symbols.add(seq_info['symbol'])
                sequence_info.append(seq_info)
            
            print(f"  Unique instruments: {len(unique_instruments)}")
            print(f"  Unique symbols: {len(unique_symbols)}")
            
            # Show sample symbols
            sample_symbols = sorted(list(unique_symbols))[:10]
            print(f"  Sample symbols: {sample_symbols}")
            
            # Analyze sequence distribution per instrument
            if sequence_info:
                instrument_counts = {}
                for seq in sequence_info:
                    inst_id = seq['instrument_id']
                    instrument_counts[inst_id] = instrument_counts.get(inst_id, 0) + 1
                
                print(f"  Sequences per instrument (sample):")
                for inst_id, count in sorted(instrument_counts.items())[:5]:
                    symbol = next(seq['symbol'] for seq in sequence_info if seq['instrument_id'] == inst_id)
                    print(f"    {symbol} (ID: {inst_id}): {count} sequences")
        
        print(f"\n" + "="*60)
        print(f"📈 TOTAL SEQUENCES ACROSS ALL DATASETS: {total_sequences:,}")
        print("="*60)
        
        # Configuration summary
        print(f"\nConfiguration:")
        print(f"  Context length: {CONTEXT_LENGTH} minutes")
        print(f"  Horizon length: {HORIZON_LENGTH} minutes") 
        print(f"  Sequence stride: {SEQUENCE_STRIDE} minutes")
        print(f"  Max sequences limit: {MAX_SEQUENCES if MAX_SEQUENCES else 'None (unlimited)'}")
        
        # Calculate theoretical maximum sequences
        print(f"\nSequence Creation Logic:")
        print(f"  Each sequence needs {CONTEXT_LENGTH + HORIZON_LENGTH} = {CONTEXT_LENGTH + HORIZON_LENGTH} consecutive data points")
        print(f"  Sequences are created every {SEQUENCE_STRIDE} data points within each instrument")
        print(f"  Sequences do NOT cross instrument boundaries (prevents mixing contracts)")
        
        return data_dict
    else:
        print(f"❌ Cache file not found: {cache_file}")
        print("   Run the main fine-tuning script first to generate cached data.")
        return None

def analyze_raw_data():
    """Analyze the raw data to understand potential sequence counts."""
    print("\n📊 Analyzing raw data...")
    
    df = pd.read_csv(DATASET_PATH)
    df['timestamp'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Apply same filtering as main script
    print("🔍 Applying ES futures filtering...")
    original_rows = len(df)
    
    month_codes = ['H', 'M', 'U', 'Z']
    mask = (
        (df['symbol'].str.len() == 4) &
        (df['symbol'].str.startswith('ES')) &
        (~df['symbol'].str.contains('-')) &
        (df['symbol'].str[2].isin(month_codes)) &
        (df['symbol'].str[3].str.isdigit())
    )
    
    df = df[mask].copy()
    print(f"   Filtered from {original_rows:,} to {len(df):,} rows")
    
    # Group by instrument
    df = df.sort_values(['instrument_id', 'timestamp']).reset_index(drop=True)
    df = df.dropna(subset=['close'])
    
    # Analyze by instrument
    print(f"\n📈 Instrument Analysis:")
    total_len_needed = CONTEXT_LENGTH + HORIZON_LENGTH
    potential_sequences = 0
    
    instruments_with_sequences = 0
    for instrument_id, instrument_data in df.groupby('instrument_id'):
        instrument_len = len(instrument_data)
        symbol = instrument_data['symbol'].iloc[0]
        
        if instrument_len >= total_len_needed:
            # Calculate how many sequences this instrument can contribute
            sequences_from_instrument = (instrument_len - total_len_needed) // SEQUENCE_STRIDE + 1
            potential_sequences += sequences_from_instrument
            instruments_with_sequences += 1
            
            if instruments_with_sequences <= 10:  # Show first 10
                print(f"  {symbol} (ID: {instrument_id}): {instrument_len:,} data points → {sequences_from_instrument:,} sequences")
    
    print(f"\nPotential Sequence Summary:")
    print(f"  Instruments with enough data: {instruments_with_sequences}")
    print(f"  Total potential sequences: {potential_sequences:,}")
    
    if MAX_SEQUENCES and potential_sequences > MAX_SEQUENCES:
        print(f"  ⚠️  Will be limited to: {MAX_SEQUENCES:,} sequences")
    
    # Data splits
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    print(f"\nData Split Sizes:")
    print(f"  Train data: {train_end:,} rows ({train_end/n:.1%})")
    print(f"  Val data: {val_end - train_end:,} rows ({(val_end - train_end)/n:.1%})")
    print(f"  Test data: {n - val_end:,} rows ({(n - val_end)/n:.1%})")

def main():
    print("🔍 Sequence Analysis for TimesFM Fine-tuning")
    print("="*60)
    
    # Try to analyze cached data first
    data_dict = analyze_cached_sequences()
    
    if data_dict is None:
        # Fall back to raw data analysis
        analyze_raw_data()
    
    print(f"\n✅ Analysis complete!")
    print(f"\nTo see this information during training, look for these log messages:")
    print(f"  📊 Created X sequences from Y data points.")
    print(f"  📊 Sequences span Z different time windows")
    print(f"  📊 Unique instruments in dataset: N")

if __name__ == "__main__":
    main()
