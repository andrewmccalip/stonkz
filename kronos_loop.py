#!/usr/bin/env python3
"""
Kronos Loop - Run predictions every hour and overlay them over ground truth for the whole day.
Uses the same approach as kronos_single_test.py but loops through multiple prediction times.
"""

import os
import sys
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import pytz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

# Add Kronos to path
sys.path.append("Kronos")
sys.path.append("Kronos/examples")

# Import Kronos components
from model import Kronos, KronosTokenizer, KronosPredictor
import torch

# ==============================================================================
# Configuration
# ==============================================================================

# Script directory
SCRIPT_DIR = Path(__file__).parent

# Test Configuration  
TEST_DATE = '2010-09-07'  # Fixed date for testing (same as single test)
START_HOUR_PT = 10   # Start predictions at 10 AM PT (same as single test)
END_HOUR_PT = 13    # End predictions at 1 PM PT (market close)
PREDICTION_INTERVAL_HOURS = 1  # Make prediction every hour

# Data Configuration  
DATASET_PATH = SCRIPT_DIR / "databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv"
LOOKBACK_MINUTES = 416  # Historical context (416 minutes = ~6.9 hours)
PREDICTION_HORIZON = 96  # Predict 96 minutes ahead (~1.6 hours)

# Kronos Model Configuration
KRONOS_MODEL = "NeoQuasar/Kronos-base"
KRONOS_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
DEVICE = "cpu"
MAX_CONTEXT = 512

# Caching Configuration
CACHE_DIR = SCRIPT_DIR / "kronos_cache"
CACHE_DIR.mkdir(exist_ok=True)
USE_CACHE = True

# Colors for plotting - different color for each hour
HOUR_COLORS = [
    '#1f77b4',  # Blue - 10:00
    '#ff7f0e',  # Orange - 11:00  
    '#2ca02c',  # Green - 12:00
    '#d62728',  # Red - 13:00
    '#9467bd',  # Purple - 14:00 (if needed)
    '#8c564b',  # Brown - 15:00 (if needed)
    '#e377c2',  # Pink - 16:00 (if needed)
]

COLORS = {
    'ground_truth': 'black',       # Black  
    'context': '#808080',          # Gray for context
    'reference': 'lightgray',      # Light gray
    'current_time': '#ff7f0e'      # Orange
}

# ==============================================================================
# Data Loading Functions (same as single test)
# ==============================================================================

def load_and_filter_data(test_date):
    """Load and filter ES futures data for the test date and previous day"""
    
    # Create cache key
    cache_key_str = f"{DATASET_PATH}_{test_date}_{LOOKBACK_MINUTES}_{PREDICTION_HORIZON}"
    cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
    cache_file = CACHE_DIR / f"loop_data_{cache_key}.pkl"
    
    if cache_file.exists() and USE_CACHE:
        print(f"📦 Loading cached data from {cache_file.name}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"📊 Loading data for {test_date}")
    df = pd.read_csv(DATASET_PATH)
    df['timestamp'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Filter for ES futures symbols
    print("🔍 Filtering for ES futures symbols...")
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
    print(f"   Filtered from {original_rows:,} to {len(df):,} rows (ES futures only)")
    
    # Parse test date
    test_date_obj = datetime.strptime(test_date, '%Y-%m-%d').date()
    prev_date_obj = test_date_obj - timedelta(days=1)
    
    # Load data for test date and previous day (for lookback)
    start_date_str = prev_date_obj.strftime('%Y-%m-%d')
    end_date_str = test_date_obj.strftime('%Y-%m-%d')
    
    # Filter date range
    df['date'] = df['timestamp'].dt.date
    date_mask = (df['date'] >= prev_date_obj) & (df['date'] <= test_date_obj)
    df = df[date_mask].copy()
    
    print(f"   Loaded {len(df):,} rows for {start_date_str} and {end_date_str}")
    
    # Add Pacific Time column for filtering
    pt_tz = pytz.timezone('US/Pacific')
    # Check if timestamp is already timezone-aware
    if df['timestamp'].dt.tz is None:
        df['timestamp_pt'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(pt_tz)
    else:
        df['timestamp_pt'] = df['timestamp'].dt.tz_convert(pt_tz)
    
    # Filter to trading hours (7 AM to 1 PM PT) but include previous day's close
    # Previous day: only from 1:01 PM PT onwards (after market close)
    # Test day: from 7:00 AM to 1:00 PM PT
    prev_day_mask = (df['date'] == prev_date_obj) & (df['timestamp_pt'].dt.hour >= 13) & (df['timestamp_pt'].dt.minute >= 1)
    test_day_mask = (df['date'] == test_date_obj) & (df['timestamp_pt'].dt.hour >= 7) & (df['timestamp_pt'].dt.hour <= 13)
    
    df = df[prev_day_mask | test_day_mask].copy()
    
    print(f"   Filtered to {len(df):,} ES futures rows")
    print(f"   Time range: {df['timestamp_pt'].min()} to {df['timestamp_pt'].max()} PT")
    
    # Cache the processed data
    if USE_CACHE:
        print(f"💾 Saving processed data to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        print(f"   ✅ Cache saved: {cache_file.name}")
    
    return df

def prepare_sequence_data(df, test_time_utc):
    """Prepare sequence data for a specific test time (same as single test)"""
    
    # Get historical data up to AND INCLUDING test time
    historical_mask = df['timestamp'] <= test_time_utc
    historical_data = df[historical_mask].copy()
    
    if len(historical_data) < LOOKBACK_MINUTES:
        # Return empty data to trigger the error handling above
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.DataFrame()
    
    # Find the exact row that matches test_time_utc
    exact_match = df[df['timestamp'] == test_time_utc]
    if len(exact_match) == 0:
        # Use the last available data point before or at test time
        context_data = historical_data.tail(LOOKBACK_MINUTES).copy()
    else:
        # Find the position of the exact match in the filtered DataFrame
        exact_position = df[df['timestamp'] == test_time_utc].index[0]
        # Get the position relative to the filtered DataFrame
        exact_pos_in_df = df.index.get_loc(exact_position)
        # Take LOOKBACK_MINUTES ending exactly at this point
        start_pos = max(0, exact_pos_in_df - LOOKBACK_MINUTES + 1)
        context_data = df.iloc[start_pos:exact_pos_in_df + 1].copy()
    
    # Get future data for comparison
    future_start = test_time_utc + timedelta(minutes=1)
    future_end = test_time_utc + timedelta(minutes=PREDICTION_HORIZON)
    future_mask = (df['timestamp'] > test_time_utc) & (df['timestamp'] <= future_end)
    future_data = df[future_mask].copy()
    
    # Prepare OHLCV data for Kronos
    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
    ohlcv_data = context_data[ohlcv_columns].copy()
    ohlcv_data['amount'] = ohlcv_data['close'] * ohlcv_data['volume']
    
    # Create timestamps for Kronos
    timestamps = pd.Series(context_data['timestamp'].values)
    
    return ohlcv_data, timestamps, future_data, context_data

# ==============================================================================
# Plotting Functions
# ==============================================================================

def create_loop_plot(all_predictions, df, test_date):
    """Create a comprehensive plot showing all hourly predictions with context history"""
    
    fig, ax = plt.subplots(1, 1, figsize=(24, 14))
    
    # Get the full day's data for ground truth
    test_date_obj = datetime.strptime(test_date, '%Y-%m-%d').date()
    day_mask = df['timestamp_pt'].dt.date == test_date_obj
    day_data = df[day_mask].copy()
    
    if len(day_data) == 0:
        print("⚠️ No data available for the test date")
        return
    
    # Use the first prediction's base price for consistency
    base_price = all_predictions[0]['base_price'] if all_predictions else day_data['close'].iloc[0]
    
    # Plot ground truth for the full day (black line, thinner and more subtle)
    day_data_normalized = day_data['close'].values / base_price
    start_time = day_data['timestamp_pt'].iloc[0]
    day_times_minutes = [(t - start_time).total_seconds() / 60 for t in day_data['timestamp_pt']]
    
    ax.plot(day_times_minutes, day_data_normalized, color=COLORS['ground_truth'], 
           linewidth=1.5, label='Ground Truth (Full Day)', alpha=0.7, zorder=1)
    
    # Plot each prediction with its context
    for i, pred_info in enumerate(all_predictions):
        color = HOUR_COLORS[i % len(HOUR_COLORS)]
        test_time_pt = pred_info['test_time_pt']
        context_data = pred_info['context_data']
        prediction_normalized = pred_info['prediction_normalized']
        
        # Calculate time offsets
        pred_start_minutes = (test_time_pt - start_time).total_seconds() / 60
        
        # Plot context history for this prediction (lighter version of hour color)
        if len(context_data) > 0:
            context_normalized = context_data['close'].values / base_price
            # Create context timeline ending at prediction start
            context_x = np.arange(pred_start_minutes - len(context_normalized), pred_start_minutes)
            
            ax.plot(context_x, context_normalized, color=color, linewidth=1.5, 
                   alpha=0.4, linestyle='-', 
                   label=f'Context {test_time_pt.strftime("%H:%M")} PT' if i == 0 else "")
        
        # Plot prediction (solid line with hour-specific color)
        pred_x = np.arange(pred_start_minutes, pred_start_minutes + len(prediction_normalized))
        ax.plot(pred_x, prediction_normalized, color=color, linewidth=2.5, 
               alpha=0.8, linestyle='-', zorder=3,
               label=f'Prediction {test_time_pt.strftime("%H:%M")} PT')
        
        # Connect context to prediction with a thin line
        if len(context_data) > 0 and len(prediction_normalized) > 0:
            context_end_price = context_normalized[-1]
            pred_start_price = prediction_normalized[0]
            ax.plot([pred_start_minutes-1, pred_start_minutes], 
                   [context_end_price, pred_start_price], 
                   color=color, linewidth=1, alpha=0.6, zorder=2)
        
        # Mark prediction start point
        if len(context_data) > 0:
            pred_start_price = context_data['close'].iloc[-1] / base_price
            ax.scatter([pred_start_minutes], [pred_start_price], 
                      color=color, s=80, zorder=5, alpha=0.9, 
                      edgecolors='white', linewidth=1)
        
        # Add vertical line at prediction time
        ax.axvline(x=pred_start_minutes, color=color, linestyle=':', 
                  alpha=0.6, linewidth=1.5, zorder=0)
    
    # Add horizontal reference line at 1.0
    ax.axhline(y=1.0, color=COLORS['reference'], linestyle=':', alpha=0.5, 
              label='Normalized Start (1.00)')
    
    # Calculate comprehensive statistics for all predictions
    if len(all_predictions) > 0:
        day_change = (day_data_normalized[-1] - day_data_normalized[0]) * 100
        
        # Calculate average metrics across all predictions
        all_mse = []
        all_mae = []
        all_dir_acc = []
        all_corr = []
        
        for pred_info in all_predictions:
            if len(pred_info['future_data']) > 0:
                future_prices = pred_info['future_data']['close'].values / base_price
                pred_prices = pred_info['prediction_normalized']
                
                min_len = min(len(future_prices), len(pred_prices))
                if min_len > 0:
                    gt = future_prices[:min_len]
                    pred = pred_prices[:min_len]
                    
                    all_mse.append(np.mean((pred - gt) ** 2))
                    all_mae.append(np.mean(np.abs(pred - gt)))
                    
                    if len(gt) > 1:
                        gt_dir = np.sign(np.diff(gt))
                        pred_dir = np.sign(np.diff(pred))
                        all_dir_acc.append(np.mean(gt_dir == pred_dir) * 100)
                    
                    if np.std(pred) > 1e-6 and np.std(gt) > 1e-6:
                        all_corr.append(np.corrcoef(pred, gt)[0, 1])
        
        # Create cleaner summary text
        prediction_times = [p['test_time_pt'].strftime('%H:%M') for p in all_predictions]
        
        summary_text = (
            f'Kronos Hourly Predictions - {test_date}\n'
            f'{"="*40}\n'
            f'Predictions: {len(all_predictions)} ({", ".join(prediction_times)} PT)\n'
            f'Lookback: {LOOKBACK_MINUTES}min | Horizon: {PREDICTION_HORIZON}min\n\n'
            f'Average Performance:\n'
            f'MSE: {np.mean(all_mse):.6f}\n' if all_mse else 'MSE: N/A\n'
            f'MAE: {np.mean(all_mae):.6f}\n' if all_mae else 'MAE: N/A\n'
            f'Dir. Accuracy: {np.mean(all_dir_acc):.1f}%\n' if all_dir_acc else 'Dir. Accuracy: N/A\n'
            f'Correlation: {np.mean(all_corr):.3f}\n' if all_corr else 'Correlation: N/A\n'
            f'\nDay Summary:\n'
            f'Total Change: {day_change:.2f}%\n'
            f'Base Price: ${base_price:.2f}\n'
            f'Time Range: {start_time.strftime("%H:%M")} - {day_data["timestamp_pt"].iloc[-1].strftime("%H:%M")} PT'
        )
    else:
        summary_text = f'No predictions generated for {test_date}'
    
    # Formatting and labels
    ax.set_xlabel('Time (Minutes from Day Start)', fontsize=14)
    ax.set_ylabel('Normalized Price', fontsize=14)
    ax.set_title(f'Kronos Hourly Predictions with Context History - {test_date}', 
                fontsize=16, fontweight='bold')
    
    # Improved legend - split into two columns
    handles, labels = ax.get_legend_handles_labels()
    # Separate ground truth and reference from predictions
    ground_truth_items = [(h, l) for h, l in zip(handles, labels) if 'Ground Truth' in l or 'Normalized Start' in l]
    prediction_items = [(h, l) for h, l in zip(handles, labels) if 'Prediction' in l]
    context_items = [(h, l) for h, l in zip(handles, labels) if 'Context' in l]
    
    # Create legend in two parts
    if ground_truth_items:
        legend1 = ax.legend([h for h, l in ground_truth_items], [l for h, l in ground_truth_items], 
                           loc='upper left', fontsize=10)
        ax.add_artist(legend1)
    
    if prediction_items:
        ax.legend([h for h, l in prediction_items], [l for h, l in prediction_items], 
                 loc='upper right', fontsize=10, ncol=2)
    
    ax.grid(True, alpha=0.2)
    
    # Add summary text box (smaller and positioned better)
    ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, 
           verticalalignment='bottom', fontsize=9, fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Set reasonable y-axis limits with some padding
    if len(day_data_normalized) > 0:
        all_prices = [day_data_normalized]
        for pred_info in all_predictions:
            if len(pred_info['context_data']) > 0:
                all_prices.append(pred_info['context_data']['close'].values / base_price)
            all_prices.append(pred_info['prediction_normalized'])
        
        all_values = np.concatenate([p for p in all_prices if len(p) > 0])
        y_min, y_max = np.min(all_values), np.max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    plt.tight_layout()
    
    # Save plot to kronos_plots folder
    plots_dir = SCRIPT_DIR / 'kronos_plots'
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / f'kronos_loop_{test_date}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Loop plot saved to: {plot_path}")
    
    plt.close()

# ==============================================================================
# Main Loop Function
# ==============================================================================

def run_kronos_loop():
    """Run Kronos predictions every hour and create overlay plot"""
    
    print("🚀 Kronos Hourly Loop")
    print("=" * 50)
    print(f"Test Date: {TEST_DATE}")
    print(f"Prediction Times: {START_HOUR_PT}:00 to {END_HOUR_PT}:00 PT (every {PREDICTION_INTERVAL_HOURS} hour)")
    print(f"Lookback: {LOOKBACK_MINUTES} minutes")
    print(f"Prediction Horizon: {PREDICTION_HORIZON} minutes")
    print("=" * 50)
    
    # Load data
    df = load_and_filter_data(TEST_DATE)
    
    # Initialize Kronos model (same as single test)
    print("\n🤖 Loading Kronos model...")
    model = Kronos.from_pretrained(KRONOS_MODEL)
    tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)
    print("✅ Kronos model loaded!")
    
    # Generate prediction times
    pt_tz = pytz.timezone('US/Pacific')
    test_date_obj = datetime.strptime(TEST_DATE, '%Y-%m-%d').date()
    
    prediction_times = []
    for hour in range(START_HOUR_PT, END_HOUR_PT + 1, PREDICTION_INTERVAL_HOURS):
        if hour <= END_HOUR_PT:  # Don't exceed end hour
            test_time_pt = pt_tz.localize(datetime.combine(test_date_obj, datetime.min.time().replace(hour=hour)))
            test_time_utc = test_time_pt.astimezone(pytz.UTC)
            prediction_times.append((test_time_pt, test_time_utc))
    
    print(f"\n🔮 Will generate {len(prediction_times)} predictions:")
    for i, (time_pt, time_utc) in enumerate(prediction_times):
        print(f"   {i+1}. {time_pt.strftime('%H:%M')} PT")
    
    # Run predictions
    all_predictions = []
    
    for i, (test_time_pt, test_time_utc) in enumerate(prediction_times):
        print(f"\n📍 Prediction {i+1}/{len(prediction_times)}: {test_time_pt.strftime('%H:%M')} PT")
        
        try:
            # Prepare data
            ohlcv_data, timestamps, future_data, context_data = prepare_sequence_data(df, test_time_utc)
            
            print(f"   Context: {len(context_data)} points ending at {test_time_pt.strftime('%H:%M')} PT")
            print(f"   Future data: {len(future_data)} points available")
            
            # Debug: Check if we have enough context data
            if len(context_data) < LOOKBACK_MINUTES:
                print(f"   ⚠️ Insufficient context data: {len(context_data)} < {LOOKBACK_MINUTES}")
                print(f"   Available data range: {df['timestamp_pt'].min()} to {df['timestamp_pt'].max()}")
                continue
            
            # Generate prediction (same signature as single test)
            pred_timestamps = []
            pred_start_time = test_time_utc + timedelta(minutes=1)
            for i in range(PREDICTION_HORIZON):
                pred_timestamps.append(pred_start_time + timedelta(minutes=i))
            
            prediction_df = predictor.predict(
                df=ohlcv_data,
                x_timestamp=timestamps,
                y_timestamp=pd.Series(pred_timestamps),
                pred_len=PREDICTION_HORIZON,
                T=1.0,
                top_p=0.9,
                sample_count=1,
            )
            
            # Normalize prediction (same as single test)
            historical_prices = context_data['close'].values
            base_price = historical_prices[0]
            normalized_historical = historical_prices / base_price
            last_historical_normalized = normalized_historical[-1]
            
            prediction_values = prediction_df.iloc[:, 0].values
            if len(prediction_values) > 0:
                prediction_base = prediction_values[0] if prediction_values[0] != 0 else 1.0
                prediction_relative = prediction_values / prediction_base
                prediction_normalized = prediction_relative * last_historical_normalized
            else:
                prediction_normalized = np.array([])
            
            # Store prediction info
            pred_info = {
                'test_time_pt': test_time_pt,
                'test_time_utc': test_time_utc,
                'context_data': context_data,
                'future_data': future_data,
                'prediction_df': prediction_df,
                'prediction_normalized': prediction_normalized,
                'base_price': base_price
            }
            all_predictions.append(pred_info)
            
            print(f"   ✅ Prediction generated: {len(prediction_normalized)} points")
            
        except Exception as e:
            print(f"   ❌ Failed to generate prediction: {e}")
            continue
    
    print(f"\n📊 Successfully generated {len(all_predictions)} predictions")
    
    # Create comprehensive plot
    if len(all_predictions) > 0:
        print("\n📈 Creating overlay plot...")
        create_loop_plot(all_predictions, df, TEST_DATE)
        print("✅ Loop completed successfully!")
    else:
        print("❌ No predictions were generated")
    
    return all_predictions

# ==============================================================================
# Main Function
# ==============================================================================

def main():
    """Main function"""
    
    # Clear previous plots
    plots_dir = SCRIPT_DIR / 'kronos_plots'
    if plots_dir.exists():
        loop_plots = list(plots_dir.glob(f'kronos_loop_{TEST_DATE}.png'))
        for plot in loop_plots:
            plot.unlink()
            print(f"🧹 Cleared previous plot: {plot.name}")
    
    # Run the loop
    all_predictions = run_kronos_loop()
    
    # Print summary
    if len(all_predictions) > 0:
        print(f"\n📋 Final Summary:")
        print(f"   Date: {TEST_DATE}")
        print(f"   Predictions: {len(all_predictions)}")
        print(f"   Time Range: {all_predictions[0]['test_time_pt'].strftime('%H:%M')} - {all_predictions[-1]['test_time_pt'].strftime('%H:%M')} PT")
        print(f"   Plot: kronos_plots/kronos_loop_{TEST_DATE}.png")

if __name__ == "__main__":
    main()
