#!/usr/bin/env python3
"""
Process Daily Datasets - Create organized daily CSV files for ES futures data.
Each file contains one trading day from 1:01 PM previous day to 1:00 PM current day (Pacific Time).
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
SOURCE_FILE = SCRIPT_DIR / "databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv"
OUTPUT_DIR = SCRIPT_DIR / "datasets"
OUTPUT_DIR.mkdir(exist_ok=True)

# Create subdirectories
ES_DIR = OUTPUT_DIR / "ES"
ES_DIR.mkdir(exist_ok=True)

# Trading session configuration
MARKET_CLOSE_HOUR = 13  # 1:00 PM PT
MARKET_CLOSE_MINUTE = 0
DATA_START_HOUR = 13    # 1:01 PM PT (previous day)
DATA_START_MINUTE = 1

# Timezone
PT_TZ = pytz.timezone('US/Pacific')

# ==============================================================================
# Data Processing Functions
# ==============================================================================

def load_and_filter_es_data():
    """Load the main dataset and filter for ES futures only"""
    
    print("📊 Loading main dataset...")
    df = pd.read_csv(SOURCE_FILE)
    print(f"   Loaded {len(df):,} total rows")
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Filter for ES futures symbols
    print("🔍 Filtering for ES futures symbols...")
    original_rows = len(df)
    
    # Filter criteria:
    # 1. Exactly 4 characters
    # 2. Starts with 'ES' 
    # 3. No dashes (excludes spreads like ESH4-ESM4)
    # 4. Third character is a month code (H, M, U, Z)
    # 5. Fourth character is a digit (year)
    month_codes = ['H', 'M', 'U', 'Z']  # Mar, Jun, Sep, Dec
    
    mask = (
        (df['symbol'].str.len() == 4) &
        (df['symbol'].str.startswith('ES')) &
        (~df['symbol'].str.contains('-')) &
        (df['symbol'].str[2].isin(month_codes)) &
        (df['symbol'].str[3].str.isdigit())
    )
    
    df = df[mask].copy()
    print(f"   Filtered from {original_rows:,} to {len(df):,} rows (ES futures only)")
    
    # Add Pacific Time column
    if df['timestamp'].dt.tz is None:
        df['timestamp_pt'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(PT_TZ)
    else:
        df['timestamp_pt'] = df['timestamp'].dt.tz_convert(PT_TZ)
    
    # Add date column (based on Pacific Time)
    df['date_pt'] = df['timestamp_pt'].dt.date
    
    print(f"   Data range: {df['timestamp_pt'].min()} to {df['timestamp_pt'].max()} PT")
    
    return df

def get_trading_day_data(df, trading_date):
    """
    Get data for a specific trading day.
    Trading day runs from 1:01 PM PT previous day to 1:00 PM PT current day.
    
    Args:
        df: DataFrame with ES futures data
        trading_date: The trading date (date object)
    
    Returns:
        DataFrame with the trading day data, or None if insufficient data
    """
    
    # Calculate start and end times
    prev_date = trading_date - timedelta(days=1)
    
    # Start: 1:01 PM PT on previous day
    start_time_pt = PT_TZ.localize(
        datetime.combine(prev_date, datetime.min.time().replace(
            hour=DATA_START_HOUR, minute=DATA_START_MINUTE
        ))
    )
    
    # End: 1:00 PM PT on trading day
    end_time_pt = PT_TZ.localize(
        datetime.combine(trading_date, datetime.min.time().replace(
            hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE
        ))
    )
    
    # Filter data for this trading session
    mask = (df['timestamp_pt'] >= start_time_pt) & (df['timestamp_pt'] <= end_time_pt)
    day_data = df[mask].copy()
    
    if len(day_data) == 0:
        return None
    
    # Add trading day identifier
    day_data['trading_date'] = trading_date
    
    # Sort by timestamp
    day_data = day_data.sort_values('timestamp_pt').reset_index(drop=True)
    
    return day_data

def create_ohlcv_candles(df, freq='1min'):
    """
    Create OHLCV candlestick data from tick data.
    
    Args:
        df: DataFrame with tick data
        freq: Frequency for candles (default: '1min')
    
    Returns:
        DataFrame with OHLCV candles, filtered and normalized
    """
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Filter for single most active contract (like in sequence processing)
    if 'symbol' in df.columns:
        # Group by symbol and find the most active one (highest volume)
        symbol_volumes = df.groupby('symbol')['volume'].sum().sort_values(ascending=False)
        if len(symbol_volumes) > 0:
            most_active_symbol = symbol_volumes.index[0]
            original_rows = len(df)
            df = df[df['symbol'] == most_active_symbol].copy()
            print(f"   Filtered to most active contract {most_active_symbol}: {original_rows} -> {len(df)} rows")
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Set timestamp as index for resampling
    df_resampled = df.set_index('timestamp_pt')
    
    # Create OHLCV candles
    ohlcv = df_resampled.groupby('symbol').resample(freq).agg({
        'close': ['first', 'max', 'min', 'last'],  # Open, High, Low, Close
        'volume': 'sum'
    }).reset_index()
    
    # Flatten column names
    ohlcv.columns = ['symbol', 'timestamp_pt', 'open', 'high', 'low', 'close', 'volume']
    
    # Remove rows with no data (NaN values)
    ohlcv = ohlcv.dropna()
    
    # Filter out low-volume rows (volume <= 1)
    print(f"   Before volume filter: {len(ohlcv)} candles")
    ohlcv = ohlcv[ohlcv['volume'] > 1].copy()
    print(f"   After volume filter: {len(ohlcv)} candles")
    
    if len(ohlcv) == 0:
        return pd.DataFrame()
    
    # Sort by timestamp to ensure proper order for normalization
    ohlcv = ohlcv.sort_values('timestamp_pt').reset_index(drop=True)
    
    # Normalize prices to start at 1.00 based on first close price
    first_close = ohlcv['close'].iloc[0]
    print(f"   Normalizing prices (first close: {first_close:.2f})")
    
    # Apply normalization to all price columns
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        ohlcv[col] = ohlcv[col] / first_close
    
    # Add UTC timestamp
    ohlcv['timestamp_utc'] = ohlcv['timestamp_pt'].dt.tz_convert('UTC')
    
    # Add trading date
    if len(df) > 0 and 'trading_date' in df.columns:
        ohlcv['trading_date'] = df['trading_date'].iloc[0]
    
    return ohlcv

def get_available_trading_dates(df):
    """Get list of available trading dates with sufficient data"""
    
    print("📅 Finding available trading dates...")
    
    # Get unique dates from the data
    all_dates = sorted(df['date_pt'].unique())
    
    valid_dates = []
    
    for date in tqdm(all_dates, desc="Checking dates"):
        # Skip weekends
        if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            continue
        
        # Get trading day data
        day_data = get_trading_day_data(df, date)
        
        # Check if we have sufficient data (at least 6 hours worth)
        if day_data is not None and len(day_data) >= 360:  # 360 minutes = 6 hours
            valid_dates.append(date)
    
    print(f"   Found {len(valid_dates)} valid trading dates")
    return valid_dates

def process_single_day(df, trading_date):
    """Process a single trading day and return OHLCV data"""
    
    # Get trading day data
    day_data = get_trading_day_data(df, trading_date)
    
    if day_data is None or len(day_data) == 0:
        return None
    
    # Create OHLCV candles
    ohlcv_data = create_ohlcv_candles(day_data)
    
    if len(ohlcv_data) == 0:
        return None
    
    # Add metadata
    ohlcv_data['trading_date'] = trading_date
    
    return ohlcv_data

def save_daily_file(ohlcv_data, trading_date):
    """Save daily OHLCV data to CSV file"""
    
    if ohlcv_data is None or len(ohlcv_data) == 0:
        return None
    
    # Create filename: ES_MMM_YYYY.csv (e.g., ES_Jan_2024.csv)
    month_name = trading_date.strftime('%b')  # Jan, Feb, Mar, etc.
    year = trading_date.year
    day_str = trading_date.strftime('%d')
    
    filename = f"ES_{month_name}_{year}_{day_str}.csv"
    filepath = ES_DIR / filename
    
    # Select and order columns
    columns_to_save = [
        'trading_date', 'timestamp_utc', 'timestamp_pt', 'symbol',
        'open', 'high', 'low', 'close', 'volume'
    ]
    
    # Ensure all columns exist
    for col in columns_to_save:
        if col not in ohlcv_data.columns:
            print(f"   ⚠️ Missing column {col} in data for {trading_date}")
            return None
    
    # Save to CSV
    ohlcv_data[columns_to_save].to_csv(filepath, index=False)
    
    return filepath

# ==============================================================================
# Main Processing Function
# ==============================================================================

def process_all_days(start_date=None, end_date=None, max_days=None):
    """
    Process all available trading days and create daily CSV files.
    
    Args:
        start_date: Start date (YYYY-MM-DD string or date object)
        end_date: End date (YYYY-MM-DD string or date object) 
        max_days: Maximum number of days to process (for testing)
    """
    
    print("🚀 ES Futures Daily Dataset Processor")
    print("=" * 60)
    
    # Load and filter data
    df = load_and_filter_es_data()
    
    # Get available trading dates
    available_dates = get_available_trading_dates(df)
    
    if len(available_dates) == 0:
        print("❌ No valid trading dates found")
        return
    
    # Filter dates if specified
    if start_date:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        available_dates = [d for d in available_dates if d >= start_date]
    
    if end_date:
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        available_dates = [d for d in available_dates if d <= end_date]
    
    if max_days:
        available_dates = available_dates[:max_days]
    
    print(f"\n📊 Processing {len(available_dates)} trading days...")
    print(f"   Date range: {available_dates[0]} to {available_dates[-1]}")
    print(f"   Output directory: {ES_DIR}")
    
    # Process each day
    successful_days = 0
    failed_days = 0
    
    for i, trading_date in enumerate(tqdm(available_dates, desc="Processing days")):
        try:
            # Process the day
            ohlcv_data = process_single_day(df, trading_date)
            
            if ohlcv_data is not None:
                # Save to file
                filepath = save_daily_file(ohlcv_data, trading_date)
                
                if filepath:
                    successful_days += 1
                    
                    # Print progress every 100 days
                    if (i + 1) % 100 == 0:
                        print(f"\n   📈 Processed {i + 1}/{len(available_dates)} days")
                        print(f"      Latest: {filepath.name} ({len(ohlcv_data)} candles, normalized)")
                else:
                    failed_days += 1
            else:
                failed_days += 1
                
        except Exception as e:
            print(f"\n   ❌ Error processing {trading_date}: {e}")
            failed_days += 1
    
    # Summary
    print(f"\n✅ Processing completed!")
    print(f"   Successful days: {successful_days}")
    print(f"   Failed days: {failed_days}")
    print(f"   Success rate: {successful_days/(successful_days+failed_days)*100:.1f}%")
    print(f"   Files saved to: {ES_DIR}")
    
    # Show some example files
    csv_files = list(ES_DIR.glob("*.csv"))
    if csv_files:
        print(f"\n📁 Example files created:")
        for i, file in enumerate(sorted(csv_files)[:5]):
            file_size = file.stat().st_size / 1024  # KB
            print(f"   {i+1}. {file.name} ({file_size:.1f} KB)")
        if len(csv_files) > 5:
            print(f"   ... and {len(csv_files) - 5} more files")

def show_sample_data():
    """Show sample of processed data"""
    
    csv_files = list(ES_DIR.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found. Run process_all_days() first.")
        return
    
    # Load a sample file
    sample_file = sorted(csv_files)[len(csv_files)//2]  # Middle file
    print(f"\n📊 Sample data from: {sample_file.name}")
    
    df_sample = pd.read_csv(sample_file)
    print(f"   Rows: {len(df_sample)}")
    print(f"   Columns: {list(df_sample.columns)}")
    print(f"   Date range: {df_sample['timestamp_pt'].min()} to {df_sample['timestamp_pt'].max()}")
    
    # Show price normalization info
    if 'close' in df_sample.columns and len(df_sample) > 0:
        first_close = df_sample['close'].iloc[0]
        last_close = df_sample['close'].iloc[-1]
        price_range = df_sample['close'].max() - df_sample['close'].min()
        print(f"   Price normalization: starts at {first_close:.6f}, ends at {last_close:.6f}")
        print(f"   Price range: {price_range:.6f} (min: {df_sample['close'].min():.6f}, max: {df_sample['close'].max():.6f})")
    
    # Show volume info
    if 'volume' in df_sample.columns:
        vol_stats = df_sample['volume'].describe()
        print(f"   Volume stats: min={vol_stats['min']:.0f}, mean={vol_stats['mean']:.0f}, max={vol_stats['max']:.0f}")
    
    print("\n   First 5 rows:")
    print(df_sample.head())
    
    print("\n   Data types:")
    print(df_sample.dtypes)

# ==============================================================================
# Main Function
# ==============================================================================

def main():
    """Main function with options"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Process ES futures data into daily CSV files')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--max-days', type=int, help='Maximum number of days to process')
    parser.add_argument('--sample', action='store_true', help='Show sample of existing data')
    parser.add_argument('--test', action='store_true', help='Process only 10 days for testing')
    
    args = parser.parse_args()
    
    if args.sample:
        show_sample_data()
    elif args.test:
        print("🧪 Test mode: Processing 10 days only")
        process_all_days(max_days=10)
    else:
        process_all_days(
            start_date=args.start_date,
            end_date=args.end_date,
            max_days=args.max_days
        )

if __name__ == "__main__":
    main()
