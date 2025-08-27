#!/usr/bin/env python3
"""
Kronos Custom Loop Predictor - Uses Kronos-base model with ES futures data
Loops through time to simulate ground truth mapping like flamingo_loop.py
Uses data processing approach from pytorch_timesfm_finetune.py

Key Features:
- Random day selection with caching
- Pacific Time (7am PT start, matching market data flow)
- 416-minute lookback, 96-minute prediction horizon
- Ground truth evaluation after 1 hour
- Real-time visualization with consistent colors
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
import time
from pathlib import Path
import pytz
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import hashlib
import shutil

# Add Kronos to path
sys.path.append("Kronos")
sys.path.append("Kronos/examples")

# Import Kronos components
from model import Kronos, KronosTokenizer, KronosPredictor

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths and Directories
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv"
CACHE_DIR = SCRIPT_DIR / "kronos_cache"
PLOT_DIR = SCRIPT_DIR / "kronos_plots"

# Create directories
CACHE_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# Time Configuration (Pacific Time - matching market data flow)
START_HOUR_PT = 7    # Start at 7am PT (covers pre-market + regular hours)
END_HOUR_PT = 13     # End at 1pm PT (market close)
TIME_STEP_MINUTES = 30  # Move forward every 30 minutes

# Model Configuration
KRONOS_MODEL = "NeoQuasar/Kronos-base"  # 102.3M parameters
KRONOS_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
DEVICE = "cpu"  # Use CPU for compatibility
MAX_CONTEXT = 512  # Kronos-base context length

# Prediction Configuration (matching pytorch_timesfm_finetune.py approach)
LOOKBACK_MINUTES = 416   # Historical context (~6.9 hours)
PREDICTION_HORIZON = 96  # Predict 96 minutes ahead (~1.6 hours)
USE_NORMALIZED_PRICES = True  # Normalize relative to sequence start

# Visualization Configuration
ANIMATION_DELAY = 1  # Seconds between predictions
SAVE_PLOTS = True

# Analysis Configuration
FLAT_THRESHOLD_PCT = 0.05  # ±0.05% threshold for direction classification

# Caching Configuration
USE_CACHE = True  # Enable caching for faster runs

# Color Scheme (consistent with other scripts)
COLORS = {
    'historical': '#1f77b4',    # Blue - historical data
    'ground_truth': '#2ca02c',  # Green - actual future data
    'prediction': '#ff7f0e',    # Orange - model predictions
    'current_time': '#d62728',  # Red - current time marker
    'reference': '#7f7f7f'      # Gray - reference lines
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def clear_plots_directory():
    """Clear the plots directory before starting"""
    print("🧹 Clearing plots directory...")
    
    if PLOT_DIR.exists():
        # Count existing files
        plot_files = list(PLOT_DIR.glob("*.png")) + list(PLOT_DIR.glob("*.jpg"))
        
        if plot_files:
            print(f"   Removing {len(plot_files)} existing plot files...")
            for file in plot_files:
                try:
                    file.unlink()
                except Exception as e:
                    print(f"   ⚠️ Failed to delete {file.name}: {e}")
        else:
            print(f"   ✓ Plots directory is already empty")
    
    print("   ✅ Plots directory cleared")

def create_cache_key(dataset_path, lookback, horizon):
    """Create a cache key for the processed data"""
    key_string = f"{dataset_path}_{lookback}_{horizon}_v2"
    return hashlib.md5(key_string.encode()).hexdigest()

# ==============================================================================
# DATA PROCESSING
# ==============================================================================

def load_and_filter_es_data():
    """
    Load and filter ES futures data for one random day with caching.
    
    Note: Market data flow starts from previous day's close (1:01pm PT)
    and continues through the next trading day.
    """
    global START_DATE, END_DATE
    
    # Create cache key
    cache_key = create_cache_key(DATASET_PATH, LOOKBACK_MINUTES, PREDICTION_HORIZON)
    cache_file = CACHE_DIR / f"es_data_{cache_key}.pkl"
    
    # Try to load from cache first
    if USE_CACHE and cache_file.exists():
        print(f"📦 Loading cached data from {cache_file.name}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            START_DATE = cached_data['selected_date']
            END_DATE = cached_data['end_date']
            
            print(f"   ✅ Loaded cached data for date: {START_DATE}")
            print(f"   📊 Data shape: {cached_data['df'].shape}")
            return cached_data['df']
        except Exception as e:
            print(f"   ⚠️ Failed to load cache: {e}")
            print(f"   Proceeding with fresh data loading...")
    
    print(f"📊 Loading fresh data from {DATASET_PATH}")
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    
    # Sample data to find available dates
    print("🔍 Sampling data to find available dates...")
    sample_df = pd.read_csv(DATASET_PATH, nrows=200000)
    sample_df['timestamp'] = pd.to_datetime(sample_df['ts_event'], utc=True)
    
    # Filter for ES futures symbols
    month_codes = ['H', 'M', 'U', 'Z']  # Mar, Jun, Sep, Dec
    es_mask = (
        (sample_df['symbol'].str.len() == 4) &
        (sample_df['symbol'].str.startswith('ES')) &
        (~sample_df['symbol'].str.contains('-')) &
        (sample_df['symbol'].str[2].isin(month_codes)) &
        (sample_df['symbol'].str[3].str.isdigit())
    )
    sample_df = sample_df[es_mask].copy()
    
    # Get available dates
    sample_df['date'] = sample_df['timestamp'].dt.date
    available_dates = sorted(sample_df['date'].unique())
    print(f"   Found {len(available_dates)} unique dates in sample")
    print(f"   Date range: {available_dates[0]} to {available_dates[-1]}")
    
    # Select random date from middle portion (avoid edge cases)
    middle_start = len(available_dates) // 4
    middle_end = 3 * len(available_dates) // 4
    middle_dates = available_dates[middle_start:middle_end]
    
    import random
    selected_date = random.choice(middle_dates)
    print(f"🎯 Randomly selected date: {selected_date}")
    
    # Load full dataset for selected date + surrounding data
    # Note: Include previous day to capture market close -> next day flow
    print("📊 Loading full dataset for selected date range...")
    df = pd.read_csv(DATASET_PATH)
    df['timestamp'] = pd.to_datetime(df['ts_event'], utc=True)
    df['date'] = df['timestamp'].dt.date
    
    # Include previous day and selected date to capture full market flow
    prev_date = selected_date - timedelta(days=1)
    date_mask = (df['date'] == prev_date) | (df['date'] == selected_date)
    df = df[date_mask].copy()
    print(f"   Loaded data for {prev_date} and {selected_date}: {len(df):,} rows")
    
    if len(df) == 0:
        raise ValueError(f"No data found for selected date range")
    
    # Filter for ES futures symbols
    print("🔍 Filtering for ES futures symbols...")
    original_rows = len(df)
    df = df[es_mask].copy()
    print(f"   Filtered from {original_rows:,} to {len(df):,} rows (ES futures only)")
    
    if len(df) == 0:
        raise ValueError(f"No ES futures data found for selected dates")
    
    # Sort by instrument and time for continuity
    print("🔍 Sorting by instrument and timestamp...")
    df = df.sort_values(['instrument_id', 'timestamp']).reset_index(drop=True)
    
    # Show instruments for this date range
    unique_instruments = df[['instrument_id', 'symbol']].drop_duplicates()
    print(f"   Found {len(unique_instruments)} unique instruments")
    print(f"   Instruments: {unique_instruments['symbol'].tolist()}")
    
    # Clean data
    df = df.dropna(subset=['close'])
    
    # Convert to Pacific Time for analysis
    print("🕐 Converting timestamps to Pacific Time...")
    pt_tz = pytz.timezone('US/Pacific')
    df['timestamp_pt'] = df['timestamp'].dt.tz_convert(pt_tz)
    df['hour_pt'] = df['timestamp_pt'].dt.hour
    df['minute_pt'] = df['timestamp_pt'].dt.minute
    df['date_pt'] = df['timestamp_pt'].dt.date
    
    # Filter to extended trading hours in Pacific Time
    # From previous day 1:01pm PT through selected day 1:00pm PT
    print(f"🕐 Filtering to extended Pacific Time hours...")
    
    # Create time-based filter
    prev_close_mask = (
        (df['date_pt'] == prev_date) & 
        (df['hour_pt'] >= 13) &  # From 1pm PT onwards on previous day
        ~((df['hour_pt'] == 13) & (df['minute_pt'] == 0))  # Exclude exactly 1:00pm
    )
    
    selected_day_mask = (
        (df['date_pt'] == selected_date) & 
        (df['hour_pt'] <= END_HOUR_PT)  # Up to 1pm PT on selected day
    )
    
    trading_hours_mask = prev_close_mask | selected_day_mask
    df = df[trading_hours_mask].copy()
    
    print(f"   Filtered to extended trading session: {len(df):,} rows")
    print(f"   Time range: {df['timestamp_pt'].min()} to {df['timestamp_pt'].max()} PT")
    
    # Update global date configuration
    START_DATE = selected_date.strftime('%Y-%m-%d')
    END_DATE = (selected_date + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"   Analysis date: {START_DATE}")
    
    # Cache the processed data
    if USE_CACHE:
        print(f"💾 Caching processed data...")
        try:
            cache_data = {
                'df': df,
                'selected_date': START_DATE,
                'end_date': END_DATE
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"   ✅ Data cached successfully!")
        except Exception as e:
            print(f"   ⚠️ Failed to cache data: {e}")
    
    return df

def prepare_sequence_data(df, now_time_utc, lookback_minutes, prediction_minutes):
    """
    Prepare data sequence for Kronos prediction.
    
    Takes trailing 416 timesteps from NOW time (matching pytorch_timesfm_finetune.py approach)
    """
    
    # Filter data up to NOW time
    historical_mask = df['timestamp'] <= now_time_utc
    historical_data = df[historical_mask].copy()
    
    if len(historical_data) < lookback_minutes:
        return None, None, None
    
    # Take trailing N minutes (like pytorch_timesfm_finetune.py)
    context_data = historical_data.tail(lookback_minutes).copy()
    
    # Get future data for ground truth evaluation
    future_start = now_time_utc + timedelta(minutes=1)
    future_end = now_time_utc + timedelta(minutes=prediction_minutes)
    future_mask = (df['timestamp'] > now_time_utc) & (df['timestamp'] <= future_end)
    future_data = df[future_mask].copy()
    
    # Prepare OHLCV data for Kronos
    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
    ohlcv_data = context_data[ohlcv_columns].copy()
    
    # Add 'amount' column (Kronos expects it)
    ohlcv_data['amount'] = ohlcv_data['close'] * ohlcv_data['volume']
    
    # Create timestamps
    timestamps = pd.Series(context_data['timestamp'].values)
    
    return ohlcv_data, timestamps, future_data

def analyze_prediction_direction(predictions, current_price):
    """Analyze directional prediction from Kronos"""
    
    if len(predictions) == 0:
        return "UNKNOWN", 0.0, 0.5
    
    # Get prediction at 1-hour mark (60 minutes ahead)
    hour_ahead_idx = min(59, len(predictions) - 1)
    
    if hasattr(predictions, 'iloc'):
        hour_ahead_price = predictions.iloc[hour_ahead_idx]['close']
        price_series = predictions['close'].values
    else:
        hour_ahead_price = predictions[hour_ahead_idx]
        price_series = predictions
    
    # Calculate expected move
    price_change = hour_ahead_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # Determine direction
    if price_change_pct > FLAT_THRESHOLD_PCT:
        direction = "UP"
    elif price_change_pct < -FLAT_THRESHOLD_PCT:
        direction = "DOWN"
    else:
        direction = "FLAT"
    
    # Calculate confidence based on prediction consistency
    if len(price_series) > 1:
        changes = np.diff(price_series)
        positive_changes = np.sum(changes > 0)
        negative_changes = np.sum(changes < 0)
        total_changes = len(changes)
        
        if total_changes > 0:
            directional_consistency = max(positive_changes, negative_changes) / total_changes
            confidence = directional_consistency
        else:
            confidence = 0.5
    else:
        confidence = 0.5
    
    return direction, price_change_pct, confidence

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def create_initial_plot():
    """Create the initial plot setup with consistent colors"""
    
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    ax1 = plt.subplot(gs[0])  # Main price plot
    ax2 = plt.subplot(gs[1])  # Accuracy matrix
    
    # Set up main plot
    ax1.set_ylabel('Normalized Price' if USE_NORMALIZED_PRICES else 'Price ($)', fontsize=12)
    ax1.set_title('Kronos-base Real-Time Predictions - ES Futures', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', labelbottom=False)
    
    # Set up matrix subplot
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Actual', 'Predicted', 'Correct'], fontsize=10)
    ax2.set_xlabel('Time (PT)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add legend for matrix
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='UP / Correct'),
        Patch(facecolor='red', edgecolor='black', label='DOWN / Wrong'),
        Patch(facecolor='gray', edgecolor='black', label='FLAT')
    ]
    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    
    return fig, (ax1, ax2)

def update_plot(fig, ax1, ax2, prediction_entry, all_predictions, loop_count):
    """Update the plot with new prediction data using consistent colors"""
    
    # Clear and redraw main plot
    ax1.clear()
    
    # Plot historical price data
    historical_data = prediction_entry['historical_data']
    historical_timestamps = prediction_entry['historical_timestamps']
    
    if len(historical_data) > 0:
        prices = historical_data['close'].values
        
        # Normalize if requested
        if USE_NORMALIZED_PRICES:
            base_price = prices[0]
            prices = prices / base_price
            ax1.axhline(y=1.0, color=COLORS['reference'], linestyle=':', alpha=0.5, 
                       label='Sequence Start (1.00)')
        
        # Plot historical data in blue
        ax1.plot(historical_timestamps, prices, color=COLORS['historical'], 
                linewidth=1.5, label='Historical Price', alpha=0.9)
    
    # Plot all previous predictions (faded orange)
    for i, prev_pred in enumerate(all_predictions[:-1]):
        pred_timestamps = prev_pred['prediction_timestamps']
        pred_prices = prev_pred['prediction_prices']
        
        ax1.plot(pred_timestamps, pred_prices, 
                color=COLORS['prediction'], alpha=0.3, linewidth=1.0)
        
        # Mark prediction start
        start_time = prev_pred['start_time']
        start_price = prev_pred['start_price']
        ax1.scatter(start_time, start_price, 
                   color=COLORS['prediction'], s=30, alpha=0.5, zorder=4)
    
    # Plot current prediction (highlighted orange)
    pred_timestamps = prediction_entry['prediction_timestamps']
    pred_prices = prediction_entry['prediction_prices']
    
    ax1.plot(pred_timestamps, pred_prices, 
            color=COLORS['prediction'], alpha=0.8, linewidth=2.5,
            label=f"Prediction: {prediction_entry['direction']}")
    
    # Mark prediction start point
    start_time = prediction_entry['start_time']
    start_price = prediction_entry['start_price']
    ax1.scatter(start_time, start_price, 
               color=COLORS['prediction'], s=80, zorder=5, 
               edgecolor='black', linewidth=1)
    
    # Add current time marker (red)
    ax1.axvline(x=start_time, color=COLORS['current_time'], 
               linestyle='--', alpha=0.6, linewidth=2, label='Current Time')
    
    # Restore plot formatting
    price_label = 'Normalized Price' if USE_NORMALIZED_PRICES else 'Price ($)'
    ax1.set_ylabel(price_label, fontsize=12)
    ax1.set_title('Kronos-base Real-Time Predictions - ES Futures', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', labelbottom=False)
    
    # Format x-axis (limit ticks to prevent memory issues)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    # Update legend
    ax1.legend(loc='upper left', fontsize=10, ncol=2)
    
    # Update accuracy matrix
    update_accuracy_matrix(ax2, all_predictions, prediction_entry)
    
    # Update the plot
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)

def update_accuracy_matrix(ax2, all_predictions, current_prediction):
    """Update the accuracy matrix subplot"""
    
    # Clear matrix
    ax2.clear()
    
    # Set up matrix subplot again
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Actual', 'Predicted', 'Correct'], fontsize=10)
    ax2.set_xlabel('Time (PT)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Color maps for directions
    dir_colors = {'UP': 'green', 'DOWN': 'red', 'FLAT': 'gray'}
    bar_height = 0.35
    bar_width = timedelta(minutes=TIME_STEP_MINUTES * 0.9)
    
    # Current time for evaluation
    current_time = current_prediction['start_time']
    
    # Draw bars for all predictions
    for pred in all_predictions:
        pred_time = pred['start_time']
        
        # Can we evaluate this prediction? (1 hour after prediction start)
        one_hour_after = pred_time + timedelta(hours=1)
        can_evaluate = current_time >= one_hour_after
        
        # Always draw prediction bar
        ax2.barh(1, bar_width.total_seconds()/86400, 
                left=pred_time, height=bar_height, 
                color=dir_colors[pred['direction']], 
                alpha=0.7, edgecolor='black', linewidth=0.5)
        
        if can_evaluate and 'actual_direction' in pred:
            # Draw actual bar (green for ground truth)
            ax2.barh(0, bar_width.total_seconds()/86400, 
                    left=pred_time, height=bar_height,
                    color=dir_colors[pred['actual_direction']], 
                    alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Draw correctness bar
            is_correct = pred['is_correct']
            correct_color = 'green' if is_correct else 'red'
            ax2.barh(2, bar_width.total_seconds()/86400,
                    left=pred_time, height=bar_height,
                    color=correct_color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add correctness indicator
            symbol = '✓' if is_correct else '✗'
            ax2.text(pred_time + timedelta(minutes=TIME_STEP_MINUTES // 2), 2, symbol,
                    ha='center', va='center', color='white', fontsize=8, weight='bold')
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='UP / Correct'),
        Patch(facecolor='red', edgecolor='black', label='DOWN / Wrong'),
        Patch(facecolor='gray', edgecolor='black', label='FLAT')
    ]
    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
               ncol=1, fontsize=8)

# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_prediction(prediction, df, current_time):
    """Evaluate a prediction against actual data (ground truth)"""
    
    pred_start = prediction['start_time']
    one_hour_later = pred_start + timedelta(hours=1)
    
    # Get actual data at 1 hour mark
    mask = (df['timestamp'] >= pred_start) & (df['timestamp'] <= one_hour_later)
    actual_data = df[mask]
    
    if len(actual_data) > 0:
        # Get actual price movement
        start_price = prediction['start_price']
        
        if USE_NORMALIZED_PRICES:
            # For normalized prices, get the actual close price at prediction start
            start_actual_price = prediction['historical_data']['close'].iloc[-1]
            end_actual_price = actual_data['close'].iloc[-1]
            actual_move_pct = ((end_actual_price - start_actual_price) / start_actual_price) * 100
        else:
            end_price = actual_data['close'].iloc[-1]
            actual_move_pct = ((end_price - start_price) / start_price) * 100
        
        # Determine actual direction
        if actual_move_pct > FLAT_THRESHOLD_PCT:
            actual_direction = 'UP'
        elif actual_move_pct < -FLAT_THRESHOLD_PCT:
            actual_direction = 'DOWN'
        else:
            actual_direction = 'FLAT'
        
        # Check if prediction was correct
        is_correct = prediction['direction'] == actual_direction
        
        # Store evaluation results
        prediction['actual_direction'] = actual_direction
        prediction['actual_move_pct'] = actual_move_pct
        prediction['is_correct'] = is_correct
        
        # Convert times to Pacific for display
        pt_tz = pytz.timezone('US/Pacific')
        pred_start_pt = pred_start.astimezone(pt_tz)
        
        print(f"\n  📊 Evaluation for {pred_start_pt.strftime('%H:%M')} PT prediction:")
        print(f"     Predicted: {prediction['direction']}, Actual: {actual_direction}")
        print(f"     Actual move: {actual_move_pct:.2f}%")
        print(f"     Result: {'✅ CORRECT' if is_correct else '❌ WRONG'}")

def analyze_final_results(predictions):
    """Analyze final prediction accuracy"""
    
    # Filter evaluated predictions
    evaluated_predictions = [p for p in predictions if 'actual_direction' in p]
    
    if not evaluated_predictions:
        print("\n⚠️ No predictions have been evaluated yet")
        return
    
    # Calculate accuracy metrics
    total_predictions = len(evaluated_predictions)
    correct_predictions = sum(1 for p in evaluated_predictions if p['is_correct'])
    accuracy = (correct_predictions / total_predictions) * 100
    
    print("\n" + "="*60)
    print("KRONOS PREDICTION ACCURACY ANALYSIS")
    print("="*60)
    print(f"Total evaluated predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall accuracy: {accuracy:.1f}%")
    
    # Accuracy by direction
    print(f"\nAccuracy by predicted direction:")
    for direction in ['UP', 'DOWN', 'FLAT']:
        dir_preds = [p for p in evaluated_predictions if p['direction'] == direction]
        if dir_preds:
            dir_correct = sum(1 for p in dir_preds if p['is_correct'])
            dir_accuracy = (dir_correct / len(dir_preds)) * 100
            print(f"  {direction}: {dir_accuracy:.1f}% ({dir_correct}/{len(dir_preds)})")

# ==============================================================================
# MAIN PREDICTION LOOP
# ==============================================================================

def run_kronos_prediction_loop():
    """Main prediction loop with proper Pacific Time handling"""
    
    # Clear plots directory
    clear_plots_directory()
    
    print("🚀 Kronos Custom Loop Predictor")
    print("=" * 60)
    print(f"Model: {KRONOS_MODEL}")
    print(f"Tokenizer: {KRONOS_TOKENIZER}")
    print(f"Device: {DEVICE}")
    print(f"Time range: {START_HOUR_PT}:00 AM - {END_HOUR_PT}:00 PM PT")
    print(f"Time step: {TIME_STEP_MINUTES} minutes")
    print(f"Lookback: {LOOKBACK_MINUTES} minutes (~{LOOKBACK_MINUTES/60:.1f} hours)")
    print(f"Prediction horizon: {PREDICTION_HORIZON} minutes (~{PREDICTION_HORIZON/60:.1f} hours)")
    print("=" * 60)
    
    # Load and prepare data
    print("\n📊 Loading ES futures data...")
    df = load_and_filter_es_data()
    
    # Initialize Kronos model
    print(f"\n🤖 Loading Kronos model: {KRONOS_MODEL}")
    try:
        tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER)
        model = Kronos.from_pretrained(KRONOS_MODEL)
        predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)
        print("✅ Kronos model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load Kronos model: {e}")
        return []
    
    # Enable interactive plotting
    plt.ion()
    
    # Create initial figure
    fig, (ax1, ax2) = create_initial_plot()
    
    # Initialize results storage
    all_predictions = []
    
    # Parse date range (will be updated by data loading)
    start_date = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_date = datetime.strptime(END_DATE, '%Y-%m-%d')
    
    loop_count = 0
    
    # Loop through the selected day
    current_date = start_date
    while current_date < end_date:
        
        # Loop through Pacific Time hours
        current_hour_pt = START_HOUR_PT
        current_minute_pt = 0
        
        while current_hour_pt < END_HOUR_PT or (current_hour_pt == END_HOUR_PT and current_minute_pt == 0):
            loop_count += 1
            
            # Create current timestamp in Pacific Time, convert to UTC for data filtering
            pt_tz = pytz.timezone('US/Pacific')
            current_time_pt = pt_tz.localize(
                current_date.replace(hour=current_hour_pt, minute=current_minute_pt, second=0)
            )
            current_time_utc = current_time_pt.astimezone(pytz.UTC)
            
            print(f"\n--- Loop {loop_count}: NOW = {current_time_pt.strftime('%Y-%m-%d %H:%M')} PT ---")
            
            # Prepare data for this timestamp (use UTC for data filtering)
            ohlcv_data, timestamps, future_data = prepare_sequence_data(
                df, current_time_utc, LOOKBACK_MINUTES, PREDICTION_HORIZON
            )
            
            if ohlcv_data is None:
                print("   Skipping - insufficient data")
                # Increment time and continue
                current_minute_pt += TIME_STEP_MINUTES
                if current_minute_pt >= 60:
                    current_hour_pt += current_minute_pt // 60
                    current_minute_pt = current_minute_pt % 60
                continue
            
            try:
                # Get current price
                current_price = ohlcv_data['close'].iloc[-1]
                
                # Create prediction timestamps
                pred_start_time = timestamps.iloc[-1] + timedelta(minutes=1)
                pred_timestamps = []
                for i in range(PREDICTION_HORIZON):
                    pred_timestamps.append(pred_start_time + timedelta(minutes=i))
                
                # Generate Kronos prediction
                print(f"   Generating {PREDICTION_HORIZON}-minute prediction...")
                pred_df = predictor.predict(
                    df=ohlcv_data,
                    x_timestamp=timestamps,
                    y_timestamp=pd.Series(pred_timestamps),
                    pred_len=PREDICTION_HORIZON,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False
                )
                
                # Analyze prediction direction
                direction, expected_move_pct, confidence = analyze_prediction_direction(
                    pred_df, current_price
                )
                
                # Prepare prediction prices for plotting
                pred_prices = pred_df['close'].values
                if USE_NORMALIZED_PRICES:
                    # Normalize predictions relative to current price
                    pred_prices = pred_prices / current_price
                    current_price_norm = 1.0
                else:
                    current_price_norm = current_price
                
                # Store prediction results
                prediction_entry = {
                    'loop': loop_count,
                    'start_time': current_time_utc,  # Store UTC for consistency
                    'start_price': current_price_norm,
                    'historical_data': ohlcv_data,
                    'historical_timestamps': timestamps,
                    'prediction_timestamps': pred_timestamps,
                    'prediction_prices': pred_prices,
                    'direction': direction,
                    'confidence': confidence,
                    'expected_move_pct': expected_move_pct,
                    'raw_prediction': pred_df
                }
                
                # Evaluate previous predictions that are now 1 hour old
                for prev_pred in all_predictions:
                    if 'actual_direction' not in prev_pred:
                        # Check if 1 hour has passed since this prediction
                        time_diff = current_time_utc - prev_pred['start_time']
                        if time_diff >= timedelta(hours=1):
                            # Evaluate this prediction
                            evaluate_prediction(prev_pred, df, current_time_utc)
                
                all_predictions.append(prediction_entry)
                
                print(f"   Current price: {current_price:.2f}")
                print(f"   Prediction: {direction} ({expected_move_pct:.2f}%)")
                print(f"   Confidence: {confidence:.1%}")
                
                # Update plot
                update_plot(fig, ax1, ax2, prediction_entry, all_predictions, loop_count-1)
                
                # Add animation delay
                time.sleep(ANIMATION_DELAY)
                
            except Exception as e:
                print(f"   ❌ Error generating prediction: {e}")
                import traceback
                traceback.print_exc()
            
            # Increment time for next loop (Pacific Time)
            current_minute_pt += TIME_STEP_MINUTES
            if current_minute_pt >= 60:
                current_hour_pt += current_minute_pt // 60
                current_minute_pt = current_minute_pt % 60
        
        # Move to next day
        current_date += timedelta(days=1)
    
    print(f"\n✅ Completed {loop_count} prediction loops")
    
    # Save final plot
    if SAVE_PLOTS:
        plt.savefig(PLOT_DIR / 'kronos_realtime_predictions.png', dpi=150, bbox_inches='tight')
        print(f"Real-time plot saved to: {PLOT_DIR / 'kronos_realtime_predictions.png'}")
    
    # Turn off interactive mode and show final plot
    plt.ioff()
    plt.show()
    
    # Analyze results
    analyze_final_results(all_predictions)
    
    return all_predictions

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("🚀 Starting Kronos Custom Loop Predictor...")
    
    # Check if required files exist
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        print("Please ensure the ES futures data file exists at the specified path.")
        sys.exit(1)
    
    # Run the prediction loop
    try:
        predictions = run_kronos_prediction_loop()
        print(f"\n✅ Prediction loop completed successfully!")
        print(f"📊 Generated {len(predictions)} predictions")
        
    except KeyboardInterrupt:
        print("\n⚠️ Prediction loop interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in prediction loop: {e}")
        import traceback
        traceback.print_exc()