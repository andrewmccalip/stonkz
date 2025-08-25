"""Cache management utilities for the AI Stock Prediction System"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processor import DataProcessor
from config import get_config
import argparse
from datetime import datetime, timedelta
import time

def preprocess_all_dates(filename: str = None):
    """Preprocess and cache data for all available dates"""
    config = get_config()
    processor = DataProcessor(config.DATA_PATH)
    
    filename = filename or config.DEFAULT_DATA_FILE
    
    print(f"Loading data from {filename}...")
    df = processor.load_csv_data(filename)
    dates = processor.get_available_dates(df)
    
    print(f"Found {len(dates)} dates to process")
    
    start_time = time.time()
    for i, date in enumerate(dates):
        print(f"\nProcessing {date} ({i+1}/{len(dates)})...")
        try:
            # This will cache the data if not already cached
            processed = processor.get_processed_trading_day(filename, date, include_indicators=True)
            print(f"  ✓ Successfully processed {len(processed)} rows")
        except Exception as e:
            print(f"  ✗ Error processing {date}: {e}")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Average time per date: {elapsed/len(dates):.2f} seconds")

def clear_cache():
    """Clear all cached data"""
    config = get_config()
    processor = DataProcessor(config.DATA_PATH)
    
    print("Clearing cache...")
    processor.clear_cache()
    print("Cache cleared successfully")

def cache_stats():
    """Show cache statistics"""
    config = get_config()
    cache_dir = Path("cache/")
    
    if not cache_dir.exists():
        print("No cache directory found")
        return
    
    cache_files = list(cache_dir.glob("*.pkl"))
    
    print(f"Cache Statistics:")
    print(f"  Total cached files: {len(cache_files)}")
    
    if cache_files:
        total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)  # MB
        print(f"  Total cache size: {total_size:.2f} MB")
        print(f"  Average file size: {total_size/len(cache_files):.2f} MB")
        
        # Show oldest and newest cache files
        cache_files.sort(key=lambda f: f.stat().st_mtime)
        oldest = cache_files[0]
        newest = cache_files[-1]
        
        print(f"\n  Oldest cache: {oldest.name}")
        print(f"    Created: {datetime.fromtimestamp(oldest.stat().st_mtime)}")
        print(f"\n  Newest cache: {newest.name}")
        print(f"    Created: {datetime.fromtimestamp(newest.stat().st_mtime)}")

def process_specific_date(date: str, filename: str = None):
    """Process and cache a specific date"""
    config = get_config()
    processor = DataProcessor(config.DATA_PATH)
    
    filename = filename or config.DEFAULT_DATA_FILE
    
    print(f"Processing data for {date}...")
    try:
        start_time = time.time()
        processed = processor.get_processed_trading_day(filename, date, include_indicators=True)
        elapsed = time.time() - start_time
        
        print(f"  ✓ Successfully processed {len(processed)} rows in {elapsed:.2f} seconds")
        
        # Show sample of indicators
        if not processed.empty:
            last_row = processed.iloc[-1]
            print(f"\n  Sample indicators at {processed.index[-1]}:")
            print(f"    Close (normalized): {last_row['close_norm']:.4f}")
            if 'rsi' in processed.columns:
                print(f"    RSI: {last_row['rsi']:.2f}")
            if 'sma_20' in processed.columns:
                print(f"    SMA 20: {last_row['sma_20']:.4f}")
            if 'macd' in processed.columns:
                print(f"    MACD: {last_row['macd']:.4f}")
                
    except Exception as e:
        print(f"  ✗ Error processing {date}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Cache management for AI Stock Predictor')
    parser.add_argument('command', choices=['preprocess', 'clear', 'stats', 'process'],
                        help='Command to execute')
    parser.add_argument('--date', type=str, help='Specific date to process (YYYY-MM-DD)')
    parser.add_argument('--file', type=str, help='Data file to use')
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        preprocess_all_dates(args.file)
    elif args.command == 'clear':
        clear_cache()
    elif args.command == 'stats':
        cache_stats()
    elif args.command == 'process':
        if not args.date:
            print("Error: --date required for process command")
            return
        process_specific_date(args.date, args.file)

if __name__ == '__main__':
    main()
