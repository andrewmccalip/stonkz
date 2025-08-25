#!/usr/bin/env python3
"""
Standalone Dataset Analysis Script for ES Futures Data
Analyzes the large Databento dataset and creates comprehensive visualizations
"""

import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))

# Import the analysis functions from finetuning.py
from finetuning import analyze_large_dataset, plot_dataset_analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import random
import pickle
import hashlib

def get_cache_key(dataset_path: str, max_rows: int = 2000000) -> str:
    """Generate a cache key based on dataset path and file modification time"""
    try:
        file_stat = Path(dataset_path).stat()
        cache_data = f"{dataset_path}_{file_stat.st_mtime}_{file_stat.st_size}_{max_rows}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    except:
        return hashlib.md5(f"{dataset_path}_{max_rows}".encode()).hexdigest()

def load_cached_data(cache_key: str) -> pd.DataFrame:
    """Load cached processed data"""
    cache_dir = Path("dataset_cache")
    cache_file = cache_dir / f"{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            print(f"📦 Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}")
            return None
    return None

def save_cached_data(cache_key: str, data: pd.DataFrame):
    """Save processed data to cache"""
    cache_dir = Path("dataset_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.pkl"
    
    try:
        print(f"💾 Saving data to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Cache saved successfully ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"⚠️  Failed to save cache: {e}")

def show_cache_status():
    """Show current cache status"""
    cache_dir = Path("dataset_cache")
    if not cache_dir.exists():
        print("📦 No cache directory found")
        return
    
    cache_files = list(cache_dir.glob("*.pkl"))
    if not cache_files:
        print("📦 No cache files found")
        return
    
    print("📦 Cache Status:")
    total_size = 0
    for cache_file in cache_files:
        size_mb = cache_file.stat().st_size / 1024 / 1024
        total_size += size_mb
        modified = datetime.fromtimestamp(cache_file.stat().st_mtime)
        print(f"   {cache_file.name}: {size_mb:.1f} MB (modified: {modified.strftime('%Y-%m-%d %H:%M')})")
    
    print(f"   Total cache size: {total_size:.1f} MB")

def clear_cache():
    """Clear all cache files"""
    cache_dir = Path("dataset_cache")
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.pkl"))
        for cache_file in cache_files:
            cache_file.unlink()
        print(f"🗑️  Cleared {len(cache_files)} cache files")
    else:
        print("📦 No cache directory found")

def quick_spot_check(dataset_path: str, sample_days: int = 100):
    """
    Quick spot check analysis with normalized price plotting
    
    Args:
        dataset_path: Path to the CSV file
        sample_days: Number of random days to sample
    """
    print("\n🎯 QUICK SPOT CHECK - 100 Random Days")
    print("="*50)
    print(f"📁 Dataset: {dataset_path}")
    print(f"🎯 Sampling {sample_days} random days for normalized price analysis")
    
    # Check for cached data first
    max_rows_for_cache = 2000000
    cache_key = get_cache_key(dataset_path, max_rows_for_cache)
    df = load_cached_data(cache_key)
    
    if df is not None:
        print(f"✅ Using cached data: {len(df):,} rows")
        total_rows = len(df)
    else:
        print("📊 No cache found, reading dataset in chunks...")
        
        # Read dataset in chunks to handle large file
        chunk_size = 100000
        chunks = []
        total_rows = 0
        
        try:
            for i, chunk in enumerate(pd.read_csv(dataset_path, chunksize=chunk_size)):
                # Convert timestamp
                chunk['ts_event'] = pd.to_datetime(chunk['ts_event'])
                chunk['date'] = chunk['ts_event'].dt.date
                
                chunks.append(chunk)
                total_rows += len(chunk)
                
                if i % 20 == 0:
                    print(f"   Processed {(i+1) * chunk_size:,} rows...")
                    
                # Stop after reasonable amount for spot check
                if total_rows > max_rows_for_cache:
                    print(f"   Stopping at {total_rows:,} rows for quick analysis...")
                    break
                    
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return
        
        print(f"✅ Loaded {total_rows:,} total rows for analysis")
        
        # Combine chunks
        print("🔄 Combining data chunks...")
        df = pd.concat(chunks, ignore_index=True)
        
        # Save raw data to cache before filtering
        save_cached_data(cache_key, df)
    
    # Check for filtered data cache
    filtered_cache_key = f"{cache_key}_filtered"
    filtered_df = load_cached_data(filtered_cache_key)
    
    if filtered_df is not None:
        print("✅ Using cached filtered data")
        df = filtered_df
        original_rows = len(df) + 1000000  # Approximate for display
        valid_symbols = df['symbol'].unique().tolist()
    else:
        # Filter for ES futures symbols only
        print("🔍 Filtering for ES futures symbols...")
        original_rows = len(df)
        
        # Filter criteria for proper ES futures
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
        
        if len(df) == 0:
            print("❌ No valid ES futures data found")
            return
        
        # Get symbol statistics
        symbol_counts = df['symbol'].value_counts()
        valid_symbols = symbol_counts[symbol_counts > 1000].index.tolist()  # Lower threshold for spot check
        print(f"   Valid symbols with >1000 rows: {valid_symbols}")
        
        if not valid_symbols:
            print("❌ No symbols found with sufficient data")
            return
        
        # Filter to valid symbols
        df = df[df['symbol'].isin(valid_symbols)].copy()
        print(f"   Final dataset: {len(df):,} rows from {len(valid_symbols)} symbols")
        
        # Save filtered data to cache
        save_cached_data(filtered_cache_key, df)
    
    print(f"📊 Using {len(df):,} rows from symbols: {valid_symbols}")
    
    # Sample random days
    print(f"🎯 Sampling {sample_days} random days...")
    unique_dates = df['date'].unique()
    if len(unique_dates) > sample_days:
        sampled_dates = random.sample(list(unique_dates), sample_days)
    else:
        sampled_dates = unique_dates
        sample_days = len(sampled_dates)
    
    sampled_data = df[df['date'].isin(sampled_dates)].copy()
    print(f"   Sampled {len(sampled_dates)} dates with {len(sampled_data):,} total rows")
    print(f"   Date range: {min(sampled_dates)} to {max(sampled_dates)}")
    
    # Create Monte Carlo style daily overlay plot
    print("📊 Creating Monte Carlo daily overlay plot...")
    
    # Create plots directory
    plots_dir = Path("spot_check_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Group data by date and symbol for daily normalization
    daily_data = []
    total_days_plotted = 0
    
    print("🔄 Processing daily data for Monte Carlo overlay...")
    
    for symbol in valid_symbols:
        symbol_data = sampled_data[sampled_data['symbol'] == symbol].copy()
        if len(symbol_data) == 0:
            continue
            
        # Group by date
        for date, day_data in symbol_data.groupby('date'):
            day_data = day_data.sort_values('ts_event').copy()
            
            # Only process days with sufficient data points (at least 100 minutes of data)
            if len(day_data) < 100:
                continue
                
            # Normalize each day to start at 1.00
            first_price = day_data['close'].iloc[0]
            day_data['normalized_price'] = day_data['close'] / first_price
            
            # Create time index (minutes from start of day)
            day_data['minutes_from_start'] = range(len(day_data))
            
            # Store the daily data
            daily_data.append({
                'symbol': symbol,
                'date': date,
                'minutes': day_data['minutes_from_start'].values,
                'normalized_prices': day_data['normalized_price'].values,
                'actual_prices': day_data['close'].values
            })
            
            total_days_plotted += 1
    
    print(f"   Processed {total_days_plotted} trading days for Monte Carlo overlay")
    
    if not daily_data:
        print("❌ No daily data found for Monte Carlo plot")
        return
    
    # Create the Monte Carlo overlay plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: All days overlaid (Monte Carlo style)
    print("   Creating Monte Carlo overlay...")
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_symbols)))
    symbol_colors = {symbol: colors[i] for i, symbol in enumerate(valid_symbols)}
    
    for i, day in enumerate(daily_data):
        # Limit to first 480 minutes (8 hours) for consistency
        max_minutes = min(480, len(day['minutes']))
        minutes = day['minutes'][:max_minutes]
        prices = day['normalized_prices'][:max_minutes]
        
        color = symbol_colors[day['symbol']]
        alpha = 0.3 if total_days_plotted > 50 else 0.5  # More transparent if many days
        
        ax1.plot(minutes, prices, color=color, alpha=alpha, linewidth=0.8)
    
    # Calculate and plot percentiles
    print("   Calculating percentile bands...")
    max_length = max(len(day['minutes']) for day in daily_data)
    max_length = min(max_length, 480)  # Limit to 8 hours
    
    # Create matrix of all normalized prices
    price_matrix = []
    for minute in range(max_length):
        minute_prices = []
        for day in daily_data:
            if minute < len(day['normalized_prices']):
                minute_prices.append(day['normalized_prices'][minute])
        if minute_prices:
            price_matrix.append(minute_prices)
    
    if price_matrix:
        minutes_axis = range(len(price_matrix))
        percentiles = [5, 25, 50, 75, 95]
        percentile_data = {}
        
        for p in percentiles:
            percentile_data[p] = [np.percentile(prices, p) for prices in price_matrix]
        
        # Plot percentile bands
        ax1.fill_between(minutes_axis, percentile_data[5], percentile_data[95], 
                        alpha=0.2, color='gray', label='5th-95th percentile')
        ax1.fill_between(minutes_axis, percentile_data[25], percentile_data[75], 
                        alpha=0.3, color='blue', label='25th-75th percentile')
        ax1.plot(minutes_axis, percentile_data[50], color='red', linewidth=2, 
                label='Median (50th percentile)')
    
    ax1.set_title(f'Monte Carlo Daily Overlay - {total_days_plotted} Trading Days\n'
                  f'Each day normalized to start at 1.00', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Minutes from Market Open')
    ax1.set_ylabel('Normalized Price (Start = 1.00)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Starting Price')
    
    # Plot 2: Distribution of end-of-day returns
    print("   Creating return distribution...")
    end_of_day_returns = []
    for day in daily_data:
        if len(day['normalized_prices']) > 0:
            eod_return = (day['normalized_prices'][-1] - 1.0) * 100  # Convert to percentage
            end_of_day_returns.append(eod_return)
    
    if end_of_day_returns:
        ax2.hist(end_of_day_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Change')
        ax2.axvline(x=np.mean(end_of_day_returns), color='green', linestyle='-', 
                   linewidth=2, label=f'Mean: {np.mean(end_of_day_returns):.2f}%')
        
        ax2.set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Daily Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = plots_dir / f"monte_carlo_daily_overlay_{sample_days}days.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Monte Carlo overlay plot saved: {plot_path}")
    
    # Print statistics
    if end_of_day_returns:
        print(f"\n📊 Daily Return Statistics:")
        print(f"   Mean daily return: {np.mean(end_of_day_returns):.3f}%")
        print(f"   Std deviation: {np.std(end_of_day_returns):.3f}%")
        print(f"   Min daily return: {np.min(end_of_day_returns):.3f}%")
        print(f"   Max daily return: {np.max(end_of_day_returns):.3f}%")
        print(f"   95th percentile: {np.percentile(end_of_day_returns, 95):.3f}%")
        print(f"   5th percentile: {np.percentile(end_of_day_returns, 5):.3f}%")
    
    # Print summary statistics
    print(f"\n📊 Spot Check Summary:")
    print(f"   Sampled days: {sample_days}")
    print(f"   Total data points: {len(sampled_data):,}")
    print(f"   Valid symbols: {len(valid_symbols)}")
    print(f"   Symbol breakdown:")
    for symbol in valid_symbols:
        symbol_data = sampled_data[sampled_data['symbol'] == symbol]
        if len(symbol_data) > 0:
            print(f"     {symbol}: {len(symbol_data):,} points, "
                  f"${symbol_data['close'].min():.2f}-${symbol_data['close'].max():.2f}")
    
    print(f"🎯 Spot check complete! Check {plots_dir}/ for visualizations")

def main():
    """Run comprehensive dataset analysis"""
    
    print("🔍 ES Futures Dataset Analysis Tool")
    print("="*50)
    print("📋 Analysis Configuration:")
    print("   ✅ Filter: ES futures only (ESH5, ESM5, ESU5, ESZ5, etc.)")
    print("   ✅ Filter: No spread symbols (excludes ESH4-ESM4, etc.)")
    print("   ✅ Filter: Standard month codes only (H=Mar, M=Jun, U=Sep, Z=Dec)")
    print("   ✅ Filter: Symbols with >10,000 rows only")
    print("   ✅ Data Quality: Remove invalid OHLC bars")
    print("   ✅ Normalization: Overlay normalized prices for comparison")
    print("   ✅ Caching: Pickle cache for faster subsequent runs")
    print()
    
    # Show cache status
    show_cache_status()
    print()
    
    # Configuration
    dataset_path = r"C:\repo_stonkz\stonkz\databento\ES\glbx-mdp3-20100606-20250822.ohlcv-1m.csv"
    
    # Run quick spot check first
    quick_spot_check(dataset_path, sample_days=100)
    
    # Ask user if they want to continue with full analysis
    print("\n" + "="*70)
    print("Continue with full comprehensive analysis? (y/n): ", end="")
    try:
        response = input().lower().strip()
        if response not in ['y', 'yes']:
            print("✅ Spot check complete! Exiting...")
            return
    except (EOFError, KeyboardInterrupt):
        print("\n✅ Spot check complete! Exiting...")
        return
    
    print("\n🔍 COMPREHENSIVE ANALYSIS")
    print("="*50)
    
    # Analysis options
    analysis_options = {
        'full_dataset': {
            'start_date': '2010-06-06',
            'end_date': '2025-08-22',
            'sample_days': 500,
            'description': 'Full 15-year dataset analysis'
        },
        'recent_5_years': {
            'start_date': '2020-01-01',
            'end_date': '2025-08-22',
            'sample_days': 200,
            'description': 'Recent 5 years (2020-2025)'
        },
        'covid_period': {
            'start_date': '2020-01-01',
            'end_date': '2021-12-31',
            'sample_days': 100,
            'description': 'COVID period (2020-2021)'
        },
        'recent_year': {
            'start_date': '2024-01-01',
            'end_date': '2025-08-22',
            'sample_days': 100,
            'description': 'Most recent year (2024-2025)'
        }
    }
    
    print("Available analysis options:")
    for key, config in analysis_options.items():
        print(f"  {key}: {config['description']}")
    
    # Run analysis for each option
    for analysis_name, config in analysis_options.items():
        print(f"\n🎯 Running analysis: {config['description']}")
        print("-" * 60)
        
        try:
            # Analyze dataset
            stats, sampled_data = analyze_large_dataset(
                dataset_path=dataset_path,
                start_date=config['start_date'],
                end_date=config['end_date'],
                sample_days=config['sample_days']
            )
            
            if stats:
                # Create plots
                plots_dir = plot_dataset_analysis(
                    stats=stats,
                    sampled_data=sampled_data,
                    date_range=(config['start_date'], config['end_date'])
                )
                
                # Rename plots directory to be specific
                specific_plots_dir = Path(f"dataset_analysis_{analysis_name}")
                if plots_dir.exists():
                    if specific_plots_dir.exists():
                        import shutil
                        shutil.rmtree(specific_plots_dir)
                    plots_dir.rename(specific_plots_dir)
                    print(f"✅ Analysis complete! Plots saved to: {specific_plots_dir}")
                
                # Print summary
                print(f"\n📊 Summary for {config['description']}:")
                print(f"   Original rows: {stats['original_rows']:,}")
                print(f"   Filtered rows: {stats['total_rows']:,} (4-letter symbols with >10K rows)")
                print(f"   Filtering efficiency: {(stats['total_rows']/stats['original_rows']*100):.1f}% data retained")
                print(f"   Trading days: {stats['unique_dates']:,}")
                print(f"   Valid symbols: {', '.join(stats['symbols'])}")
                print(f"   Symbol breakdown:")
                for symbol, count in stats['symbol_counts'].items():
                    print(f"     {symbol}: {count:,} rows")
                print(f"   Estimated training sequences: {stats['sequence_analysis']['training_sequences_estimate']:,.0f}")
                print(f"   Estimated validation sequences: {stats['sequence_analysis']['validation_sequences_estimate']:,.0f}")
                print(f"   Estimated test sequences: {stats['sequence_analysis']['test_sequences_estimate']:,.0f}")
                
        except Exception as e:
            print(f"❌ Error in analysis {analysis_name}: {e}")
            continue
    
    print(f"\n✅ All dataset analyses complete!")
    print(f"📁 Check the dataset_analysis_* directories for plots and visualizations")

if __name__ == "__main__":
    main()
