#!/usr/bin/env python3
"""
Kronos Backtest - Run predictions on multiple days and evaluate performance.
Evaluates directional accuracy (up/down) and shape similarity across many days.
"""

import os
import sys
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import seaborn as sns

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

# Backtest Configuration
NUM_DAYS = 15  # Number of random days to test
TEST_HOURS = [7, 8, 9, 10, 11]  # Hours to test each day (PT)
RANDOM_SEED = 42  # For reproducible results

# Data Configuration  
DATASET_PATH = SCRIPT_DIR / "databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv"
LOOKBACK_MINUTES = 416  # Historical context (416 minutes = ~6.9 hours)
PREDICTION_HORIZON = 96  # Predict 96 minutes ahead (~1.6 hours)

# Kronos Model Configuration
KRONOS_MODEL = "NeoQuasar/Kronos-mini"
KRONOS_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
DEVICE = "cpu"
MAX_CONTEXT = 512

# Caching Configuration
CACHE_DIR = SCRIPT_DIR / "kronos_cache"
CACHE_DIR.mkdir(exist_ok=True)
USE_CACHE = True

# Evaluation Thresholds
DIRECTION_THRESHOLD_PCT = 0.1  # Minimum % change to consider as up/down (vs flat)
SHAPE_CORRELATION_THRESHOLD = 0.5  # Threshold for "good" shape correlation

# ==============================================================================
# Data Loading Functions
# ==============================================================================

def get_available_dates(df, min_data_points=1000):
    """Get list of dates that have sufficient data for backtesting"""
    
    # Filter for ES futures symbols
    month_codes = ['H', 'M', 'U', 'Z']  # Mar, Jun, Sep, Dec
    mask = (
        (df['symbol'].str.len() == 4) &
        (df['symbol'].str.startswith('ES')) &
        (~df['symbol'].str.contains('-')) &
        (df['symbol'].str[2].isin(month_codes)) &
        (df['symbol'].str[3].str.isdigit())
    )
    df_filtered = df[mask].copy()
    
    # Add Pacific Time column
    pt_tz = pytz.timezone('US/Pacific')
    if df_filtered['timestamp'].dt.tz is None:
        df_filtered['timestamp_pt'] = df_filtered['timestamp'].dt.tz_localize('UTC').dt.tz_convert(pt_tz)
    else:
        df_filtered['timestamp_pt'] = df_filtered['timestamp'].dt.tz_convert(pt_tz)
    
    # Group by date and count data points
    df_filtered['date'] = df_filtered['timestamp_pt'].dt.date
    date_counts = df_filtered.groupby('date').size()
    
    # Filter dates with sufficient data
    valid_dates = date_counts[date_counts >= min_data_points].index.tolist()
    
    # Filter to weekdays only (exclude weekends)
    weekday_dates = [d for d in valid_dates if d.weekday() < 5]
    
    return sorted(weekday_dates)

def load_day_data(df, test_date):
    """Load and prepare data for a specific test date"""
    
    # Create cache key
    cache_key_str = f"backtest_{test_date}_{LOOKBACK_MINUTES}_{PREDICTION_HORIZON}"
    cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
    cache_file = CACHE_DIR / f"backtest_day_{cache_key}.pkl"
    
    if cache_file.exists() and USE_CACHE:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Parse test date
    test_date_obj = test_date
    prev_date_obj = test_date_obj - timedelta(days=1)
    
    # Filter for ES futures symbols
    month_codes = ['H', 'M', 'U', 'Z']
    mask = (
        (df['symbol'].str.len() == 4) &
        (df['symbol'].str.startswith('ES')) &
        (~df['symbol'].str.contains('-')) &
        (df['symbol'].str[2].isin(month_codes)) &
        (df['symbol'].str[3].str.isdigit())
    )
    df_filtered = df[mask].copy()
    
    # Add Pacific Time column
    pt_tz = pytz.timezone('US/Pacific')
    if df_filtered['timestamp'].dt.tz is None:
        df_filtered['timestamp_pt'] = df_filtered['timestamp'].dt.tz_localize('UTC').dt.tz_convert(pt_tz)
    else:
        df_filtered['timestamp_pt'] = df_filtered['timestamp'].dt.tz_convert(pt_tz)
    
    # Filter date range (include previous day for lookback)
    df_filtered['date'] = df_filtered['timestamp_pt'].dt.date
    date_mask = (df_filtered['date'] >= prev_date_obj) & (df_filtered['date'] <= test_date_obj)
    day_data = df_filtered[date_mask].copy()
    
    # Filter to trading hours but include previous day's close
    prev_day_mask = (day_data['date'] == prev_date_obj) & (day_data['timestamp_pt'].dt.hour >= 13) & (day_data['timestamp_pt'].dt.minute >= 1)
    test_day_mask = (day_data['date'] == test_date_obj) & (day_data['timestamp_pt'].dt.hour >= 6) & (day_data['timestamp_pt'].dt.hour <= 14)
    
    day_data = day_data[prev_day_mask | test_day_mask].copy()
    
    # Cache the processed data
    if USE_CACHE:
        with open(cache_file, 'wb') as f:
            pickle.dump(day_data, f)
    
    return day_data

def prepare_prediction_data(df, test_time_utc):
    """Prepare data for a single prediction (same logic as loop)"""
    
    # Get historical data up to test time
    historical_mask = df['timestamp'] <= test_time_utc
    historical_data = df[historical_mask].copy()
    
    if len(historical_data) < LOOKBACK_MINUTES:
        return None, None, None, None
    
    # Find exact match or use closest
    exact_match = df[df['timestamp'] == test_time_utc]
    if len(exact_match) == 0:
        context_data = historical_data.tail(LOOKBACK_MINUTES).copy()
    else:
        exact_position = df[df['timestamp'] == test_time_utc].index[0]
        exact_pos_in_df = df.index.get_loc(exact_position)
        start_pos = max(0, exact_pos_in_df - LOOKBACK_MINUTES + 1)
        context_data = df.iloc[start_pos:exact_pos_in_df + 1].copy()
    
    # Get future data for evaluation
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
# Evaluation Functions
# ==============================================================================

def evaluate_direction(prediction, ground_truth, threshold_pct=DIRECTION_THRESHOLD_PCT):
    """Evaluate if prediction correctly predicted direction (up/down/flat)"""
    
    if len(prediction) == 0 or len(ground_truth) == 0:
        return False, "no_data", "no_data"
    
    # Calculate percentage changes from start to end
    pred_change_pct = (prediction[-1] - prediction[0]) / prediction[0] * 100
    gt_change_pct = (ground_truth[-1] - ground_truth[0]) / ground_truth[0] * 100
    
    # Classify directions
    def classify_direction(change_pct):
        if change_pct > threshold_pct:
            return "up"
        elif change_pct < -threshold_pct:
            return "down"
        else:
            return "flat"
    
    pred_direction = classify_direction(pred_change_pct)
    gt_direction = classify_direction(gt_change_pct)
    
    # Check if prediction is correct
    is_correct = pred_direction == gt_direction
    
    return is_correct, pred_direction, gt_direction

def evaluate_shape_similarity(prediction, ground_truth):
    """Evaluate how similar the prediction shape is to ground truth"""
    
    if len(prediction) == 0 or len(ground_truth) == 0:
        return 0.0, 0.0, 0.0
    
    # Ensure same length for comparison
    min_len = min(len(prediction), len(ground_truth))
    pred = prediction[:min_len]
    gt = ground_truth[:min_len]
    
    # Calculate metrics
    mse = mean_squared_error(gt, pred)
    mae = mean_absolute_error(gt, pred)
    
    # Calculate correlation (shape similarity)
    if np.std(pred) > 1e-6 and np.std(gt) > 1e-6:
        correlation, _ = pearsonr(pred, gt)
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    return correlation, mse, mae

# ==============================================================================
# Backtesting Engine
# ==============================================================================

def run_single_prediction(predictor, df, test_date, test_hour):
    """Run a single prediction and evaluate it"""
    
    # Create test time
    pt_tz = pytz.timezone('US/Pacific')
    test_time_pt = pt_tz.localize(datetime.combine(test_date, datetime.min.time().replace(hour=test_hour)))
    test_time_utc = test_time_pt.astimezone(pytz.UTC)
    
    try:
        # Prepare data
        ohlcv_data, timestamps, future_data, context_data = prepare_prediction_data(df, test_time_utc)
        
        if ohlcv_data is None or len(context_data) < LOOKBACK_MINUTES:
            return None
        
        # Generate prediction
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
        
        # Normalize prediction
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
            return None
        
        # Prepare ground truth
        if len(future_data) == 0:
            return None
        
        ground_truth_prices = future_data['close'].values
        ground_truth_normalized = ground_truth_prices / base_price
        
        # Evaluate predictions
        direction_correct, pred_dir, gt_dir = evaluate_direction(
            prediction_normalized, ground_truth_normalized)
        
        correlation, mse, mae = evaluate_shape_similarity(
            prediction_normalized, ground_truth_normalized)
        
        return {
            'date': test_date,
            'hour': test_hour,
            'test_time_pt': test_time_pt,
            'prediction_normalized': prediction_normalized,
            'ground_truth_normalized': ground_truth_normalized,
            'base_price': base_price,
            'direction_correct': direction_correct,
            'pred_direction': pred_dir,
            'gt_direction': gt_dir,
            'correlation': correlation,
            'mse': mse,
            'mae': mae,
            'context_length': len(context_data),
            'future_length': len(future_data)
        }
        
    except Exception as e:
        print(f"   ❌ Error in prediction for {test_date} {test_hour}:00: {e}")
        return None

def run_backtest():
    """Run the complete backtest across multiple days"""
    
    global NUM_DAYS  # Make NUM_DAYS accessible
    
    print("🚀 Kronos Backtest")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Days to test: {NUM_DAYS}")
    print(f"  Hours per day: {TEST_HOURS}")
    print(f"  Total predictions: {NUM_DAYS * len(TEST_HOURS)}")
    print(f"  Lookback: {LOOKBACK_MINUTES} min | Horizon: {PREDICTION_HORIZON} min")
    print("=" * 60)
    
    # Load dataset and get available dates
    print("📊 Loading dataset and finding available dates...")
    df = pd.read_csv(DATASET_PATH)
    df['timestamp'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    available_dates = get_available_dates(df)
    print(f"   Found {len(available_dates)} valid trading days")
    
    if len(available_dates) < NUM_DAYS:
        print(f"   ⚠️ Only {len(available_dates)} days available, reducing test size")
        NUM_DAYS = len(available_dates)
    
    # Randomly select test dates
    random.seed(RANDOM_SEED)
    test_dates = random.sample(available_dates, NUM_DAYS)
    test_dates.sort()
    
    print(f"   Selected {len(test_dates)} test dates:")
    for i, date in enumerate(test_dates[:5]):  # Show first 5
        print(f"     {i+1}. {date}")
    if len(test_dates) > 5:
        print(f"     ... and {len(test_dates)-5} more")
    
    # Initialize Kronos model
    print("\n🤖 Loading Kronos model...")
    model = Kronos.from_pretrained(KRONOS_MODEL)
    tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)
    print("✅ Kronos model loaded!")
    
    # Run backtest
    print(f"\n🔮 Running backtest on {len(test_dates)} days...")
    all_results = []
    
    for date_idx, test_date in enumerate(test_dates):
        print(f"\n📅 Day {date_idx+1}/{len(test_dates)}: {test_date}")
        
        # Load day data
        try:
            day_data = load_day_data(df, test_date)
            print(f"   Loaded {len(day_data)} data points")
        except Exception as e:
            print(f"   ❌ Failed to load data: {e}")
            continue
        
        # Run predictions for each hour
        day_results = []
        for hour in TEST_HOURS:
            print(f"   🕐 {hour}:00 PT", end=" ")
            
            result = run_single_prediction(predictor, day_data, test_date, hour)
            if result is not None:
                day_results.append(result)
                all_results.append(result)
                print(f"✅ Dir: {result['direction_correct']} | Corr: {result['correlation']:.3f}")
            else:
                print("❌ Failed")
        
        print(f"   📊 Day summary: {len(day_results)}/{len(TEST_HOURS)} successful predictions")
    
    print(f"\n📈 Backtest completed!")
    print(f"   Total successful predictions: {len(all_results)}")
    print(f"   Success rate: {len(all_results)/(len(test_dates)*len(TEST_HOURS))*100:.1f}%")
    
    return all_results, test_dates

# ==============================================================================
# Results Analysis and Plotting
# ==============================================================================

def create_backtest_summary_plot(results, test_dates):
    """Create comprehensive summary plots of backtest results"""
    
    if len(results) == 0:
        print("❌ No results to plot")
        return
    
    # Convert results to DataFrame for easier analysis
    df_results = pd.DataFrame(results)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Directional Accuracy by Hour
    ax1 = plt.subplot(3, 3, 1)
    hour_accuracy = df_results.groupby('hour')['direction_correct'].mean() * 100
    bars1 = ax1.bar(hour_accuracy.index, hour_accuracy.values, 
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(hour_accuracy)])
    ax1.set_title('Directional Accuracy by Hour', fontweight='bold')
    ax1.set_xlabel('Hour (PT)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax1.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 2. Shape Correlation by Hour
    ax2 = plt.subplot(3, 3, 2)
    hour_correlation = df_results.groupby('hour')['correlation'].mean()
    bars2 = ax2.bar(hour_correlation.index, hour_correlation.values,
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(hour_correlation)])
    ax2.set_title('Average Shape Correlation by Hour', fontweight='bold')
    ax2.set_xlabel('Hour (PT)')
    ax2.set_ylabel('Correlation')
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axhline(y=SHAPE_CORRELATION_THRESHOLD, color='green', linestyle='--', alpha=0.5, 
               label=f'Good threshold ({SHAPE_CORRELATION_THRESHOLD})')
    ax2.legend()
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 3. Overall Performance Distribution
    ax3 = plt.subplot(3, 3, 3)
    overall_accuracy = df_results['direction_correct'].mean() * 100
    overall_correlation = df_results['correlation'].mean()
    
    metrics = ['Directional\nAccuracy (%)', 'Shape\nCorrelation']
    values = [overall_accuracy, overall_correlation * 100]  # Scale correlation for visualization
    colors = ['lightblue', 'lightgreen']
    
    bars3 = ax3.bar(metrics, values, color=colors)
    ax3.set_title('Overall Performance', fontweight='bold')
    ax3.set_ylabel('Value')
    
    # Add value labels
    ax3.text(0, overall_accuracy + 2, f'{overall_accuracy:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax3.text(1, overall_correlation * 100 + 2, f'{overall_correlation:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Correlation Distribution
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(df_results['correlation'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(x=df_results['correlation'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df_results["correlation"].mean():.3f}')
    ax4.axvline(x=SHAPE_CORRELATION_THRESHOLD, color='green', linestyle='--', 
               label=f'Good threshold: {SHAPE_CORRELATION_THRESHOLD}')
    ax4.set_title('Shape Correlation Distribution', fontweight='bold')
    ax4.set_xlabel('Correlation')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    
    # 5. Direction Prediction Confusion Matrix
    ax5 = plt.subplot(3, 3, 5)
    direction_counts = df_results.groupby(['gt_direction', 'pred_direction']).size().unstack(fill_value=0)
    
    if len(direction_counts) > 0:
        sns.heatmap(direction_counts, annot=True, fmt='d', cmap='Blues', ax=ax5)
        ax5.set_title('Direction Prediction Matrix', fontweight='bold')
        ax5.set_xlabel('Predicted Direction')
        ax5.set_ylabel('Actual Direction')
    
    # 6. Performance Over Time (by date)
    ax6 = plt.subplot(3, 3, 6)
    daily_accuracy = df_results.groupby('date')['direction_correct'].mean() * 100
    ax6.plot(daily_accuracy.index, daily_accuracy.values, 'o-', alpha=0.7)
    ax6.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax6.axhline(y=daily_accuracy.mean(), color='green', linestyle='--', alpha=0.7, 
               label=f'Average ({daily_accuracy.mean():.1f}%)')
    ax6.set_title('Daily Directional Accuracy', fontweight='bold')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Accuracy (%)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.legend()
    
    # 7. MSE and MAE by Hour
    ax7 = plt.subplot(3, 3, 7)
    hour_mse = df_results.groupby('hour')['mse'].mean()
    hour_mae = df_results.groupby('hour')['mae'].mean()
    
    x = np.arange(len(hour_mse))
    width = 0.35
    
    ax7.bar(x - width/2, hour_mse.values, width, label='MSE', alpha=0.7)
    ax7.bar(x + width/2, hour_mae.values, width, label='MAE', alpha=0.7)
    ax7.set_title('Prediction Error by Hour', fontweight='bold')
    ax7.set_xlabel('Hour (PT)')
    ax7.set_ylabel('Error')
    ax7.set_xticks(x)
    ax7.set_xticklabels(hour_mse.index)
    ax7.legend()
    
    # 8. Success Rate by Day of Week
    ax8 = plt.subplot(3, 3, 8)
    df_results['day_of_week'] = pd.to_datetime(df_results['date']).dt.day_name()
    dow_accuracy = df_results.groupby('day_of_week')['direction_correct'].mean() * 100
    
    # Reorder to start with Monday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    dow_accuracy = dow_accuracy.reindex([d for d in day_order if d in dow_accuracy.index])
    
    bars8 = ax8.bar(range(len(dow_accuracy)), dow_accuracy.values, 
                    color=['lightcoral', 'lightsalmon', 'lightgreen', 'lightblue', 'plum'][:len(dow_accuracy)])
    ax8.set_title('Accuracy by Day of Week', fontweight='bold')
    ax8.set_xlabel('Day of Week')
    ax8.set_ylabel('Accuracy (%)')
    ax8.set_xticks(range(len(dow_accuracy)))
    ax8.set_xticklabels(dow_accuracy.index, rotation=45)
    ax8.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    # Add value labels
    for i, bar in enumerate(bars8):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate comprehensive statistics
    total_predictions = len(df_results)
    successful_predictions = len(df_results)
    directional_accuracy = df_results['direction_correct'].mean() * 100
    avg_correlation = df_results['correlation'].mean()
    avg_mse = df_results['mse'].mean()
    avg_mae = df_results['mae'].mean()
    
    # Direction breakdown
    up_correct = df_results[(df_results['gt_direction'] == 'up') & (df_results['direction_correct'])].shape[0]
    down_correct = df_results[(df_results['gt_direction'] == 'down') & (df_results['direction_correct'])].shape[0]
    flat_correct = df_results[(df_results['gt_direction'] == 'flat') & (df_results['direction_correct'])].shape[0]
    
    up_total = df_results[df_results['gt_direction'] == 'up'].shape[0]
    down_total = df_results[df_results['gt_direction'] == 'down'].shape[0]
    flat_total = df_results[df_results['gt_direction'] == 'flat'].shape[0]
    
    stats_text = f"""Kronos Backtest Summary
{'='*40}
Test Period: {min(test_dates)} to {max(test_dates)}
Total Days Tested: {len(test_dates)}
Hours per Day: {len(TEST_HOURS)} ({', '.join(map(str, TEST_HOURS))})
Total Predictions: {total_predictions}

Overall Performance:
• Directional Accuracy: {directional_accuracy:.1f}%
• Average Shape Correlation: {avg_correlation:.3f}
• Average MSE: {avg_mse:.6f}
• Average MAE: {avg_mae:.6f}

Direction Breakdown:
• UP predictions: {up_correct}/{up_total} ({up_correct/up_total*100 if up_total > 0 else 0:.1f}%)
• DOWN predictions: {down_correct}/{down_total} ({down_correct/down_total*100 if down_total > 0 else 0:.1f}%)
• FLAT predictions: {flat_correct}/{flat_total} ({flat_correct/flat_total*100 if flat_total > 0 else 0:.1f}%)

Best Performing Hour: {hour_accuracy.idxmax()}:00 PT ({hour_accuracy.max():.1f}%)
Best Correlation Hour: {hour_correlation.idxmax()}:00 PT ({hour_correlation.max():.3f})

Model Configuration:
• Lookback: {LOOKBACK_MINUTES} minutes
• Horizon: {PREDICTION_HORIZON} minutes
• Direction Threshold: {DIRECTION_THRESHOLD_PCT}%"""
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = SCRIPT_DIR / 'kronos_plots'
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / f'kronos_backtest_{NUM_DAYS}days.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Backtest summary plot saved to: {plot_path}")
    
    plt.close()
    
    return df_results

# ==============================================================================
# Main Function
# ==============================================================================

def main():
    """Main backtest function"""
    
    # Clear previous plots
    plots_dir = SCRIPT_DIR / 'kronos_plots'
    if plots_dir.exists():
        backtest_plots = list(plots_dir.glob('kronos_backtest_*.png'))
        for plot in backtest_plots:
            plot.unlink()
            print(f"🧹 Cleared previous plot: {plot.name}")
    
    # Run backtest
    results, test_dates = run_backtest()
    
    if len(results) > 0:
        # Create summary plots
        print(f"\n📈 Creating summary plots...")
        df_results = create_backtest_summary_plot(results, test_dates)
        
        # Print final summary
        print(f"\n📋 Final Backtest Summary:")
        print(f"   Tested {len(test_dates)} days with {len(TEST_HOURS)} hours each")
        print(f"   Successful predictions: {len(results)}")
        print(f"   Overall directional accuracy: {df_results['direction_correct'].mean()*100:.1f}%")
        print(f"   Average shape correlation: {df_results['correlation'].mean():.3f}")
        print(f"   Best hour: {df_results.groupby('hour')['direction_correct'].mean().idxmax()}:00 PT")
        print(f"   Plot saved: kronos_plots/kronos_backtest_{NUM_DAYS}days.png")
    else:
        print("❌ No successful predictions to analyze")

if __name__ == "__main__":
    main()
