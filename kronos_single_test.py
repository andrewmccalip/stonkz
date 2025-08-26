#!/usr/bin/env python3
"""
Kronos Single Test - One prediction to verify x-axis and data alignment
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pytz
import pickle
import hashlib

# Add Kronos to path
sys.path.append("Kronos")
sys.path.append("Kronos/examples")

# Import Kronos components
from model import Kronos, KronosTokenizer, KronosPredictor

# Configuration
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv"
CACHE_DIR = SCRIPT_DIR / "kronos_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Model Configuration
KRONOS_MODEL = "NeoQuasar/Kronos-base"
KRONOS_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
DEVICE = "cpu"
MAX_CONTEXT = 512

# Prediction Configuration
LOOKBACK_MINUTES = 416
PREDICTION_HORIZON = 96
USE_NORMALIZED_PRICES = True

# Test Configuration
TEST_DATE = '2011-09-09'  # Fixed date for testing
TEST_TIME_PT = '10:00'    # 10:00 AM Pacific Time

# Colors
COLORS = {
    'historical': '#1f77b4',    # Blue
    'prediction': '#ff7f0e',    # Orange  
    'current_time': '#d62728',  # Red
    'reference': '#7f7f7f'      # Gray
}

def load_test_data():
    """Load a specific day's data for testing"""
    
    print(f"📊 Loading test data for {TEST_DATE}")
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    
    # Load data for test date and previous day
    df = pd.read_csv(DATASET_PATH)
    df['timestamp'] = pd.to_datetime(df['ts_event'], utc=True)
    df['date'] = df['timestamp'].dt.date
    
    # Get test date and previous day
    test_date_obj = datetime.strptime(TEST_DATE, '%Y-%m-%d').date()
    prev_date = test_date_obj - timedelta(days=1)
    
    # Filter for date range
    date_mask = (df['date'] == prev_date) | (df['date'] == test_date_obj)
    df = df[date_mask].copy()
    print(f"   Loaded {len(df):,} rows for {prev_date} and {test_date_obj}")
    
    # Filter for ES futures
    month_codes = ['H', 'M', 'U', 'Z']
    es_mask = (
        (df['symbol'].str.len() == 4) &
        (df['symbol'].str.startswith('ES')) &
        (~df['symbol'].str.contains('-')) &
        (df['symbol'].str[2].isin(month_codes)) &
        (df['symbol'].str[3].str.isdigit())
    )
    df = df[es_mask].copy()
    print(f"   Filtered to {len(df):,} ES futures rows")
    
    # Sort by time
    df = df.sort_values(['instrument_id', 'timestamp']).reset_index(drop=True)
    df = df.dropna(subset=['close'])
    
    # Convert to Pacific Time
    pt_tz = pytz.timezone('US/Pacific')
    df['timestamp_pt'] = df['timestamp'].dt.tz_convert(pt_tz)
    df['hour_pt'] = df['timestamp_pt'].dt.hour
    df['minute_pt'] = df['timestamp_pt'].dt.minute
    df['date_pt'] = df['timestamp_pt'].dt.date
    
    print(f"   Time range: {df['timestamp_pt'].min()} to {df['timestamp_pt'].max()} PT")
    
    return df

def prepare_test_sequence(df, test_time_str):
    """Prepare sequence for the specific test time"""
    
    # Parse test time
    test_hour, test_minute = map(int, test_time_str.split(':'))
    test_date_obj = datetime.strptime(TEST_DATE, '%Y-%m-%d').date()
    
    # Create test timestamp in Pacific Time
    pt_tz = pytz.timezone('US/Pacific')
    test_time_pt = pt_tz.localize(
        datetime.combine(test_date_obj, datetime.min.time()).replace(
            hour=test_hour, minute=test_minute
        )
    )
    test_time_utc = test_time_pt.astimezone(pytz.UTC)
    
    print(f"📍 Test time: {test_time_pt.strftime('%Y-%m-%d %H:%M')} PT")
    print(f"📍 Test time UTC: {test_time_utc.strftime('%Y-%m-%d %H:%M')} UTC")
    
    # Get historical data up to AND INCLUDING test time
    historical_mask = df['timestamp'] <= test_time_utc
    historical_data = df[historical_mask].copy()
    
    print(f"   Historical data available: {len(historical_data):,} rows")
    
    if len(historical_data) < LOOKBACK_MINUTES:
        raise ValueError(f"Insufficient historical data: {len(historical_data)} < {LOOKBACK_MINUTES}")
    
    # Find the exact row that matches test_time_utc
    exact_match = df[df['timestamp'] == test_time_utc]
    if len(exact_match) == 0:
        print(f"   ⚠️  No exact match for test time, using closest available")
        # Use the last available data point before or at test time
        context_data = historical_data.tail(LOOKBACK_MINUTES).copy()
    else:
        print(f"   ✅ Found exact match for test time")
        # Get the index of the exact match
        exact_idx = exact_match.index[0]
        # Take LOOKBACK_MINUTES ending exactly at this point
        start_idx = max(0, exact_idx - LOOKBACK_MINUTES + 1)
        context_data = df.iloc[start_idx:exact_idx + 1].copy()
    
    # Verify the context data ends exactly at test time
    last_context_time = context_data['timestamp'].iloc[-1]
    print(f"   Using last {LOOKBACK_MINUTES} minutes of data")
    print(f"   Context data range: {context_data['timestamp_pt'].min()} to {context_data['timestamp_pt'].max()} PT")
    print(f"   Last context timestamp (UTC): {last_context_time}")
    print(f"   Test time (UTC):              {test_time_utc}")
    print(f"   Context ends at test time? {last_context_time == test_time_utc}")
    
    # Get future data for comparison
    future_start = test_time_utc + timedelta(minutes=1)
    future_end = test_time_utc + timedelta(minutes=PREDICTION_HORIZON)
    future_mask = (df['timestamp'] > test_time_utc) & (df['timestamp'] <= future_end)
    future_data = df[future_mask].copy()
    
    print(f"   Future data available: {len(future_data)} rows")
    if len(future_data) > 0:
        print(f"   Future data range: {future_data['timestamp_pt'].min()} to {future_data['timestamp_pt'].max()} PT")
    
    # Prepare OHLCV data for Kronos
    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
    ohlcv_data = context_data[ohlcv_columns].copy()
    ohlcv_data['amount'] = ohlcv_data['close'] * ohlcv_data['volume']
    
    # Create timestamps for Kronos
    timestamps = pd.Series(context_data['timestamp'].values)
    
    return ohlcv_data, timestamps, future_data, test_time_utc, context_data

def create_single_plot(ohlcv_data, timestamps, prediction_df, future_data, test_time_utc, context_data):
    """Create a single plot using the approach from pytorch_timesfm_finetune.py"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Convert test time to Pacific Time for plotting
    pt_tz = pytz.timezone('US/Pacific')
    test_time_pt = test_time_utc.astimezone(pt_tz)
    
    # Prepare data similar to pytorch_timesfm_finetune.py validation plots
    # Historical context (normalized to start at 1.0 like in the finetune script)
    historical_prices = context_data['close'].values
    base_price = historical_prices[0]  # Normalize to first price
    normalized_historical = historical_prices / base_price
    
    # Ground truth (future data)
    if len(future_data) > 0:
        ground_truth_prices = future_data['close'].values
        normalized_ground_truth = ground_truth_prices / base_price
    else:
        normalized_ground_truth = np.array([])
    
    # Prediction data (normalize to continue from end of historical data)
    prediction_values = prediction_df.iloc[:, 0].values  # First column
    # Normalize Kronos prediction to start at the end of historical data
    last_historical_normalized = normalized_historical[-1]
    # Kronos outputs relative changes, so we need to apply them starting from the last historical point
    if len(prediction_values) > 0:
        # Assume Kronos prediction is a continuation - normalize to start from last historical point
        prediction_base = prediction_values[0] if prediction_values[0] != 0 else 1.0
        prediction_relative = prediction_values / prediction_base  # Get relative changes
        prediction_normalized = prediction_relative * last_historical_normalized  # Apply to last historical point
    else:
        prediction_normalized = np.array([])
    
    # Create x-axis arrays like in pytorch_timesfm_finetune.py
    context_len = len(normalized_historical)
    prediction_len = len(prediction_normalized)
    ground_truth_len = len(normalized_ground_truth)
    
    # X-axis for context (historical data)
    context_x = np.arange(context_len)
    
    # X-axis for predictions (starts right after context)
    prediction_x = np.arange(context_len, context_len + prediction_len)
    
    # X-axis for ground truth (same as prediction)
    ground_truth_x = np.arange(context_len, context_len + min(ground_truth_len, prediction_len))
    
    # Plot historical context in blue (like pytorch_timesfm_finetune.py)
    ax.plot(context_x, normalized_historical, 'b-', linewidth=2, label='History (Context)', alpha=0.8)
    
    # Plot ground truth in black (like pytorch_timesfm_finetune.py)
    if len(normalized_ground_truth) > 0:
        ground_truth_to_plot = normalized_ground_truth[:prediction_len]  # Match prediction length
        ax.plot(ground_truth_x, ground_truth_to_plot, 'k-', linewidth=3, label='Ground Truth', alpha=0.9)
    
    # Plot Kronos prediction in red (like pytorch_timesfm_finetune.py)
    ax.plot(prediction_x, prediction_normalized, 'r-', linewidth=2, label='Kronos Prediction', alpha=0.8)
    
    # Connect context to predictions with thin lines (like pytorch_timesfm_finetune.py)
    if len(normalized_historical) > 0 and len(prediction_normalized) > 0:
        ax.plot([context_x[-1], prediction_x[0]], [normalized_historical[-1], prediction_normalized[0]], 
                'r-', linewidth=1, alpha=0.3)
    
    if len(normalized_historical) > 0 and len(normalized_ground_truth) > 0:
        ax.plot([context_x[-1], ground_truth_x[0]], [normalized_historical[-1], normalized_ground_truth[0]], 
                'k-', linewidth=1, alpha=0.3)
    
    # Add vertical line to separate context from predictions (like pytorch_timesfm_finetune.py)
    ax.axvline(x=context_len, color='gray', linestyle='--', alpha=0.5, label='Prediction Start')
    
    # Add horizontal reference line at 1.0 (normalized start)
    ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.5, label='Sequence Start (1.00)')
    
    # Calculate metrics (like pytorch_timesfm_finetune.py)
    if len(normalized_ground_truth) > 0 and len(prediction_normalized) > 0:
        # Ensure same length for comparison
        min_len = min(len(normalized_ground_truth), len(prediction_normalized))
        gt_for_metrics = normalized_ground_truth[:min_len]
        pred_for_metrics = prediction_normalized[:min_len]
        
        # MSE and MAE
        mse = np.mean((pred_for_metrics - gt_for_metrics) ** 2)
        mae = np.mean(np.abs(pred_for_metrics - gt_for_metrics))
        
        # Directional accuracy
        if len(gt_for_metrics) > 1:
            gt_direction = np.sign(np.diff(gt_for_metrics))
            pred_direction = np.sign(np.diff(pred_for_metrics))
            dir_accuracy = np.mean(gt_direction == pred_direction) * 100
        else:
            dir_accuracy = 50.0
        
        # Correlation
        if np.std(pred_for_metrics) > 1e-6 and np.std(gt_for_metrics) > 1e-6:
            correlation = np.corrcoef(pred_for_metrics, gt_for_metrics)[0, 1]
        else:
            correlation = 0.0
        
        # Price changes
        context_change = (normalized_historical[-1] - normalized_historical[0]) / normalized_historical[0] * 100 if len(normalized_historical) > 0 else 0
        gt_change = (gt_for_metrics[-1] - gt_for_metrics[0]) / gt_for_metrics[0] * 100 if len(gt_for_metrics) > 0 and gt_for_metrics[0] != 0 else 0
        pred_change = (pred_for_metrics[-1] - pred_for_metrics[0]) / pred_for_metrics[0] * 100 if len(pred_for_metrics) > 0 and pred_for_metrics[0] != 0 else 0
        
        # Create metrics text (like pytorch_timesfm_finetune.py)
        metrics_text = (
            f'Kronos Prediction Metrics:\n'
            f'MSE: {mse:.6f}\n'
            f'MAE: {mae:.6f}\n'
            f'Dir. Accuracy: {dir_accuracy:.1f}%\n'
            f'Correlation: {correlation:.3f}\n\n'
            f'Price Changes:\n'
            f'Context: {context_change:.2f}%\n'
            f'Ground Truth: {gt_change:.2f}%\n'
            f'Predicted: {pred_change:.2f}%\n\n'
            f'Base Price: ${base_price:.2f}\n'
            f'Test Time: {test_time_pt.strftime("%Y-%m-%d %H:%M")} PT'
        )
    else:
        metrics_text = (
            f'Kronos Prediction:\n'
            f'No ground truth available\n\n'
            f'Base Price: ${base_price:.2f}\n'
            f'Test Time: {test_time_pt.strftime("%Y-%m-%d %H:%M")} PT'
        )
    
    # Formatting and labels (like pytorch_timesfm_finetune.py)
    ax.set_xlabel('Time Steps (Minutes)', fontsize=12)
    ax.set_ylabel('Normalized Price', fontsize=12)
    ax.set_title(f'Kronos Single Test - {TEST_DATE} at {TEST_TIME_PT} PT', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add metrics text box (like pytorch_timesfm_finetune.py)
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10, fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    # Set reasonable y-axis limits for normalized prices
    all_values = np.concatenate([normalized_historical, 
                                normalized_ground_truth[:prediction_len] if len(normalized_ground_truth) > 0 else [], 
                                prediction_normalized])
    if len(all_values) > 0:
        y_min, y_max = np.min(all_values), np.max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    plt.tight_layout()
    
    # Save plot to kronos_plots folder
    plots_dir = SCRIPT_DIR / 'kronos_plots'
    plots_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
    plot_path = plots_dir / 'kronos_single_test.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {plot_path}")
    
    # Close the plot instead of showing it
    plt.close()
    
    # Print detailed debug info (adapted from pytorch_timesfm_finetune.py style)
    print(f"\n🔍 Data Alignment Debug Info:")
    print(f"Test time: {test_time_pt.strftime('%Y-%m-%d %H:%M')} PT")
    print(f"Context length: {context_len} points")
    print(f"Prediction length: {prediction_len} points")
    print(f"Ground truth length: {ground_truth_len} points")
    print(f"Historical data range: {normalized_historical[0]:.4f} to {normalized_historical[-1]:.4f} (normalized)")
    if len(normalized_ground_truth) > 0:
        print(f"Ground truth range: {normalized_ground_truth[0]:.4f} to {normalized_ground_truth[-1]:.4f} (normalized)")
    print(f"Prediction range: {prediction_normalized[0]:.4f} to {prediction_normalized[-1]:.4f} (normalized)")
    print(f"Base price: ${base_price:.2f}")
    
    # Check data alignment
    print(f"\n⚠️  X-axis Alignment Check:")
    print(f"Context x-axis: {context_x[0]} to {context_x[-1]}")
    print(f"Prediction x-axis: {prediction_x[0]} to {prediction_x[-1]}")
    print(f"Prediction starts right after context? {prediction_x[0] == context_x[-1] + 1}")
    
    if len(normalized_ground_truth) > 0:
        print(f"Ground truth x-axis: {ground_truth_x[0]} to {ground_truth_x[-1]}")
        print(f"Ground truth aligns with prediction? {ground_truth_x[0] == prediction_x[0]}")

def main():
    """Main test function"""
    
    print("🚀 Kronos Single Test")
    print("=" * 50)
    print(f"Test Date: {TEST_DATE}")
    print(f"Test Time: {TEST_TIME_PT} PT")
    print(f"Lookback: {LOOKBACK_MINUTES} minutes")
    print(f"Prediction: {PREDICTION_HORIZON} minutes")
    print("=" * 50)
    
    # Load test data
    df = load_test_data()
    
    # Prepare sequence
    ohlcv_data, timestamps, future_data, test_time_utc, context_data = prepare_test_sequence(df, TEST_TIME_PT)
    
    # Load Kronos model
    print(f"\n🤖 Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER)
    model = Kronos.from_pretrained(KRONOS_MODEL)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)
    print("✅ Kronos model loaded!")
    
    # Generate prediction
    print(f"\n🔮 Generating prediction...")
    
    # Create prediction timestamps
    pred_start_time = timestamps.iloc[-1] + timedelta(minutes=1)
    pred_timestamps = []
    for i in range(PREDICTION_HORIZON):
        pred_timestamps.append(pred_start_time + timedelta(minutes=i))
    
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
    
    print("✅ Prediction generated!")
    
    # Create plot
    print(f"\n📊 Creating plot...")
    create_single_plot(ohlcv_data, timestamps, pred_df, future_data, test_time_utc, context_data)
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()
