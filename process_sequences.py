#!/usr/bin/env python3
"""
Process Sequences - Create 512-minute overlapping sequences from daily ES futures data.
Each sequence walks forward 60 minutes through each trading day.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR = Path(__file__).parent
DAILY_DATA_DIR = SCRIPT_DIR / "datasets" / "ES"
OUTPUT_DIR = SCRIPT_DIR / "datasets" / "sequences"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sequence parameters
SEQUENCE_LENGTH_MINUTES = 512  # Total sequence length
STRIDE_MINUTES = 60           # Step forward each time
MIN_SEQUENCES_PER_DAY = 1     # Minimum sequences required per day

# Timezone
PT_TZ = pytz.timezone('US/Pacific')

# ==============================================================================
# Sequence Processing Functions
# ==============================================================================

def load_daily_file(daily_file_path):
    """Load a daily ES futures file and prepare for sequence extraction"""
    
    try:
        df = pd.read_csv(daily_file_path)
        
        if len(df) == 0:
            return None
            
        # Convert timestamp columns
        df['timestamp_pt'] = pd.to_datetime(df['timestamp_pt'])
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        
        # Filter for single most active contract (like in other files)
        if 'symbol' in df.columns:
            # Group by symbol and find the most active one (highest volume)
            symbol_volumes = df.groupby('symbol')['volume'].sum().sort_values(ascending=False)
            if len(symbol_volumes) > 0:
                most_active_symbol = symbol_volumes.index[0]
                original_rows = len(df)
                df = df[df['symbol'] == most_active_symbol].copy()
                print(f"   Filtered to most active contract {most_active_symbol}: {original_rows} -> {len(df)} rows")
        
        # Sort by timestamp to ensure proper order
        df = df.sort_values('timestamp_pt').reset_index(drop=True)
        
        # Remove any rows with missing close prices
        df = df.dropna(subset=['close'])
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error loading {daily_file_path.name}: {e}")
        return None

def extract_sequences_from_day(df, trading_date):
    """
    Extract all possible 512-minute sequences from a single trading day.
    
    Args:
        df: DataFrame with daily OHLCV data
        trading_date: The trading date for naming
    
    Returns:
        List of (sequence_data, sequence_name) tuples
    """
    
    if df is None or len(df) < SEQUENCE_LENGTH_MINUTES:
        return []
    
    sequences = []
    
    # Calculate how many sequences we can extract
    max_start_idx = len(df) - SEQUENCE_LENGTH_MINUTES
    
    # Walk forward with stride
    start_idx = 0
    while start_idx <= max_start_idx:
        
        # Extract 512-minute sequence
        end_idx = start_idx + SEQUENCE_LENGTH_MINUTES
        sequence_data = df.iloc[start_idx:end_idx].copy()
        
        # Get the start time for naming
        start_time_pt = sequence_data['timestamp_pt'].iloc[0]
        
        # Create sequence name: sequence_YYYY-MM-DD_HHMMPT.csv
        sequence_name = f"sequence_{trading_date}_{start_time_pt.strftime('%H%M')}PT.csv"
        
        # Select only OHLCV columns
        ohlcv_columns = ['timestamp_pt', 'timestamp_utc', 'open', 'high', 'low', 'close', 'volume']
        sequence_ohlcv = sequence_data[ohlcv_columns].copy()
        
        sequences.append((sequence_ohlcv, sequence_name))
        
        # Move forward by stride
        start_idx += STRIDE_MINUTES
    
    return sequences

def save_sequence(sequence_data, sequence_name):
    """Save a sequence to CSV file"""
    
    try:
        filepath = OUTPUT_DIR / sequence_name
        sequence_data.to_csv(filepath, index=False)
        return filepath
    except Exception as e:
        print(f"   ❌ Error saving {sequence_name}: {e}")
        return None

def process_single_daily_file(daily_file_path):
    """Process a single daily file and extract all sequences"""
    
    # Extract trading date from filename (ES_MMM_YYYY_DD.csv)
    filename = daily_file_path.stem  # Remove .csv
    parts = filename.split('_')
    
    if len(parts) != 4 or parts[0] != 'ES':
        print(f"   ⚠️ Unexpected filename format: {daily_file_path.name}")
        return 0
    
    try:
        month_str = parts[1]
        year = int(parts[2])
        day = int(parts[3])
        
        # Convert month name to number
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        month = month_map.get(month_str)
        
        if month is None:
            print(f"   ⚠️ Unknown month: {month_str}")
            return 0
            
        trading_date = datetime(year, month, day).date()
        
    except (ValueError, IndexError) as e:
        print(f"   ⚠️ Error parsing date from {daily_file_path.name}: {e}")
        return 0
    
    # Load the daily data
    df = load_daily_file(daily_file_path)
    
    if df is None:
        return 0
    
    # Extract sequences
    sequences = extract_sequences_from_day(df, trading_date)
    
    if len(sequences) < MIN_SEQUENCES_PER_DAY:
        print(f"   ⚠️ Skipping {daily_file_path.name}: only {len(sequences)} sequences (need {MIN_SEQUENCES_PER_DAY}+)")
        return 0
    
    # Save all sequences
    saved_count = 0
    for sequence_data, sequence_name in sequences:
        filepath = save_sequence(sequence_data, sequence_name)
        if filepath:
            saved_count += 1
    
    return saved_count

def get_daily_files():
    """Get list of all daily ES files to process"""
    
    if not DAILY_DATA_DIR.exists():
        print(f"❌ Daily data directory not found: {DAILY_DATA_DIR}")
        return []
    
    # Get all ES CSV files
    daily_files = list(DAILY_DATA_DIR.glob("ES_*.csv"))
    daily_files.sort()
    
    print(f"📁 Found {len(daily_files)} daily files to process")
    
    return daily_files

def process_all_sequences(max_days=None):
    """
    Process all daily files and create sequence files.
    
    Args:
        max_days: Maximum number of days to process (for testing)
    """
    
    print("🚀 ES Futures Sequence Processor")
    print("=" * 60)
    print(f"   Sequence length: {SEQUENCE_LENGTH_MINUTES} minutes")
    print(f"   Stride: {STRIDE_MINUTES} minutes")
    print(f"   Input directory: {DAILY_DATA_DIR}")
    print(f"   Output directory: {OUTPUT_DIR}")
    
    # Get daily files
    daily_files = get_daily_files()
    
    if len(daily_files) == 0:
        print("❌ No daily files found")
        return
    
    # Limit for testing
    if max_days:
        daily_files = daily_files[:max_days]
        print(f"\n🧪 Test mode: Processing {len(daily_files)} days only")
    
    # Clear existing sequence files
    existing_sequences = list(OUTPUT_DIR.glob("sequence_*.csv"))
    if existing_sequences:
        print(f"\n🧹 Clearing {len(existing_sequences)} existing sequence files...")
        for seq_file in existing_sequences:
            seq_file.unlink()
    
    # Process each daily file
    total_sequences = 0
    successful_days = 0
    skipped_days = 0
    
    print(f"\n📊 Processing {len(daily_files)} daily files...")
    
    for i, daily_file in enumerate(tqdm(daily_files, desc="Processing days")):
        
        try:
            sequences_created = process_single_daily_file(daily_file)
            
            if sequences_created > 0:
                successful_days += 1
                total_sequences += sequences_created
                
                # Print progress every 500 days
                if (i + 1) % 500 == 0:
                    print(f"\n   📈 Processed {i + 1}/{len(daily_files)} days")
                    print(f"      Latest: {daily_file.name} ({sequences_created} sequences)")
                    print(f"      Total sequences so far: {total_sequences}")
                
                # Print progress every 100 days for more frequent updates
                elif (i + 1) % 100 == 0:
                    print(f"\n   📊 Progress: {i + 1}/{len(daily_files)} days ({total_sequences} sequences)")
            else:
                skipped_days += 1
                
        except Exception as e:
            print(f"\n   ❌ Error processing {daily_file.name}: {e}")
            skipped_days += 1
    
    # Summary
    print(f"\n✅ Sequence processing completed!")
    print(f"   Successful days: {successful_days}")
    print(f"   Skipped days: {skipped_days}")
    print(f"   Total sequences created: {total_sequences}")
    print(f"   Average sequences per day: {total_sequences/successful_days:.1f}" if successful_days > 0 else "   No successful days")
    print(f"   Files saved to: {OUTPUT_DIR}")
    
    # Show some example files
    sequence_files = list(OUTPUT_DIR.glob("sequence_*.csv"))
    if sequence_files:
        print(f"\n📁 Example sequence files created:")
        for i, file in enumerate(sorted(sequence_files)[:5]):
            file_size = file.stat().st_size / 1024  # KB
            print(f"   {i+1}. {file.name} ({file_size:.1f} KB)")
        if len(sequence_files) > 5:
            print(f"   ... and {len(sequence_files) - 5} more files")

def show_sample_sequence():
    """Show sample of a sequence file"""
    
    sequence_files = list(OUTPUT_DIR.glob("sequence_*.csv"))
    if not sequence_files:
        print("❌ No sequence files found. Run process_all_sequences() first.")
        return
    
    # Load a sample file
    sample_file = sorted(sequence_files)[len(sequence_files)//2]  # Middle file
    print(f"\n📊 Sample sequence from: {sample_file.name}")
    
    df_sample = pd.read_csv(sample_file)
    print(f"   Rows: {len(df_sample)} (should be {SEQUENCE_LENGTH_MINUTES})")
    print(f"   Columns: {list(df_sample.columns)}")
    
    if len(df_sample) > 0:
        df_sample['timestamp_pt'] = pd.to_datetime(df_sample['timestamp_pt'])
        start_time = df_sample['timestamp_pt'].iloc[0]
        end_time = df_sample['timestamp_pt'].iloc[-1]
        duration = (end_time - start_time).total_seconds() / 60
        
        print(f"   Time range: {start_time} to {end_time} PT")
        print(f"   Duration: {duration:.0f} minutes")
        
        # Price info
        price_range = df_sample['close'].max() - df_sample['close'].min()
        print(f"   Price range: {df_sample['close'].min():.6f} to {df_sample['close'].max():.6f} (range: {price_range:.6f})")
        
        # Volume info
        vol_stats = df_sample['volume'].describe()
        print(f"   Volume stats: min={vol_stats['min']:.0f}, mean={vol_stats['mean']:.0f}, max={vol_stats['max']:.0f}")
    
    print("\n   First 5 rows:")
    print(df_sample.head())
    
    print("\n   Last 5 rows:")
    print(df_sample.tail())

# ==============================================================================
# Main Function
# ==============================================================================

def main():
    """Main function with options"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Process daily ES data into 512-minute sequences')
    parser.add_argument('--max-days', type=int, help='Maximum number of days to process')
    parser.add_argument('--sample', action='store_true', help='Show sample of existing sequence data')
    parser.add_argument('--test', action='store_true', help='Process only 10 days for testing')
    
    args = parser.parse_args()
    
    if args.sample:
        show_sample_sequence()
    elif args.test:
        print("🧪 Test mode: Processing 10 days only")
        process_all_sequences(max_days=10)
    else:
        process_all_sequences(max_days=args.max_days)

if __name__ == "__main__":
    main()
