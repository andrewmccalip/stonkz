#!/usr/bin/env python3
"""
Count sequences that would be created for TimesFM fine-tuning.
"""
import pandas as pd
from pathlib import Path

# Configuration (from pytorch_timesfm_finetune.py)
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv"
CONTEXT_LENGTH = 448    # Context window
HORIZON_LENGTH = 64     # Prediction horizon  
SEQUENCE_STRIDE = 64    # Stride between sequences
MAX_SEQUENCES = None    # No limit

def count_sequences():
    """Count how many sequences would be created from the dataset."""
    print("🔍 Analyzing ES futures data for sequence counting...")
    
    # Load and filter data (same as main script)
    df = pd.read_csv(DATASET_PATH)
    df['timestamp'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"📊 Loaded {len(df):,} total data points")
    
    # Filter for ES futures symbols
    original_rows = len(df)
    month_codes = ['H', 'M', 'U', 'Z']  # Mar, Jun, Sep, Dec
    
    mask = (
        (df['symbol'].str.len() == 4) &
        (df['symbol'].str.startswith('ES')) &
        (~df['symbol'].str.contains('-')) &
        (df['symbol'].str[2].isin(month_codes)) &
        (df['symbol'].str[3].str.isdigit())
    )
    
    df = df[mask].copy()
    print(f"📊 After ES futures filtering: {len(df):,} data points ({len(df)/original_rows:.1%} of original)")
    
    # Group by instrument and sort
    df = df.sort_values(['instrument_id', 'timestamp']).reset_index(drop=True)
    df = df.dropna(subset=['close'])
    print(f"📊 After removing NaN: {len(df):,} data points")
    
    # Calculate data splits (80% train, 10% val, 10% test)
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    train_data = df[:train_end]
    val_data = df[train_end:val_end]
    test_data = df[val_end:]
    
    print(f"\n📈 Data Splits:")
    print(f"   Train: {len(train_data):,} data points ({len(train_data)/n:.1%})")
    print(f"   Val:   {len(val_data):,} data points ({len(val_data)/n:.1%})")
    print(f"   Test:  {len(test_data):,} data points ({len(test_data)/n:.1%})")
    
    # Count sequences for each split
    total_len_needed = CONTEXT_LENGTH + HORIZON_LENGTH  # 448 + 64 = 512
    
    def count_sequences_in_split(data, split_name):
        """Count sequences that can be created from a data split."""
        print(f"\n🔍 Analyzing {split_name} split...")
        
        total_sequences = 0
        instruments_with_data = 0
        instruments_with_sequences = 0
        sequence_details = []
        
        for instrument_id, instrument_data in data.groupby('instrument_id'):
            instruments_with_data += 1
            instrument_len = len(instrument_data)
            symbol = instrument_data['symbol'].iloc[0]
            
            if instrument_len >= total_len_needed:
                # Calculate sequences: (length - needed_length) // stride + 1
                sequences_from_instrument = (instrument_len - total_len_needed) // SEQUENCE_STRIDE + 1
                total_sequences += sequences_from_instrument
                instruments_with_sequences += 1
                
                sequence_details.append({
                    'symbol': symbol,
                    'instrument_id': instrument_id,
                    'data_points': instrument_len,
                    'sequences': sequences_from_instrument
                })
        
        print(f"   Total instruments in split: {instruments_with_data}")
        print(f"   Instruments with enough data (≥{total_len_needed} points): {instruments_with_sequences}")
        print(f"   Total sequences that will be created: {total_sequences:,}")
        
        # Show top 10 instruments by sequence count
        sequence_details.sort(key=lambda x: x['sequences'], reverse=True)
        print(f"   Top 10 instruments by sequence count:")
        for i, detail in enumerate(sequence_details[:10]):
            print(f"     {i+1:2d}. {detail['symbol']} (ID: {detail['instrument_id']}): "
                  f"{detail['data_points']:,} points → {detail['sequences']:,} sequences")
        
        return total_sequences, instruments_with_sequences, sequence_details
    
    # Count sequences for each split
    train_sequences, train_instruments, train_details = count_sequences_in_split(train_data, "Train")
    val_sequences, val_instruments, val_details = count_sequences_in_split(val_data, "Validation")  
    test_sequences, test_instruments, test_details = count_sequences_in_split(test_data, "Test")
    
    # Summary
    total_sequences = train_sequences + val_sequences + test_sequences
    total_instruments = train_instruments + val_instruments + test_instruments
    
    print(f"\n" + "="*70)
    print(f"📊 FINAL SEQUENCE COUNT SUMMARY")
    print(f"="*70)
    print(f"Configuration:")
    print(f"  Context length: {CONTEXT_LENGTH} minutes (~{CONTEXT_LENGTH/60:.1f} hours)")
    print(f"  Horizon length: {HORIZON_LENGTH} minutes (~{HORIZON_LENGTH/60:.1f} hours)")
    print(f"  Total length needed per sequence: {total_len_needed} minutes")
    print(f"  Sequence stride: {SEQUENCE_STRIDE} minutes")
    print(f"  Max sequences limit: {'None (unlimited)' if MAX_SEQUENCES is None else MAX_SEQUENCES}")
    
    print(f"\nSequence Counts:")
    print(f"  Train sequences:      {train_sequences:,}")
    print(f"  Validation sequences: {val_sequences:,}")
    print(f"  Test sequences:       {test_sequences:,}")
    print(f"  TOTAL SEQUENCES:      {total_sequences:,}")
    
    print(f"\nInstrument Counts (with sufficient data):")
    print(f"  Train instruments:    {train_instruments}")
    print(f"  Validation instruments: {val_instruments}")
    print(f"  Test instruments:     {test_instruments}")
    
    # Check if we'd hit the MAX_SEQUENCES limit
    if MAX_SEQUENCES is not None:
        if total_sequences > MAX_SEQUENCES:
            print(f"\n⚠️  WARNING: Total sequences ({total_sequences:,}) exceeds MAX_SEQUENCES limit ({MAX_SEQUENCES:,})")
            print(f"   Only the first {MAX_SEQUENCES:,} sequences will be used for training.")
        else:
            print(f"\n✅ Total sequences ({total_sequences:,}) is within MAX_SEQUENCES limit ({MAX_SEQUENCES:,})")
    
    print(f"\n" + "="*70)
    
    # Show unique symbols across all splits
    all_symbols = set()
    for details in [train_details, val_details, test_details]:
        for detail in details:
            all_symbols.add(detail['symbol'])
    
    print(f"📈 Unique ES futures symbols in dataset: {len(all_symbols)}")
    print(f"   Symbols: {sorted(list(all_symbols))}")
    
    return {
        'train_sequences': train_sequences,
        'val_sequences': val_sequences, 
        'test_sequences': test_sequences,
        'total_sequences': total_sequences,
        'unique_symbols': len(all_symbols)
    }

if __name__ == "__main__":
    print("🚀 TimesFM Sequence Counter")
    print("="*50)
    
    try:
        results = count_sequences()
        print(f"\n✅ Analysis complete!")
        print(f"\nKey takeaway: You will be training on {results['train_sequences']:,} sequences")
        print(f"and validating on {results['val_sequences']:,} sequences.")
        
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {DATASET_PATH}")
        print("   Make sure the dataset file exists at the expected location.")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
