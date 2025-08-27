#!/usr/bin/env python3
"""
TimesFM Stock Prediction System
==============================

This script uses Google's TimesFM model to predict stock prices based on historical data.
It loads normalized ES futures data and makes predictions at regular intervals throughout
the trading day, comparing predictions with actual ground truth values.

Features:
- GPU acceleration support (automatic detection)
- Normalized price predictions with 448 minutes of historical context
- 64-minute prediction horizon
- Visual comparison of predictions vs ground truth
- MSE/MAE metrics for prediction accuracy

Based on the official TimesFM PyTorch implementation.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from datetime import datetime, timedelta
import pytz
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

# Import the official TimesFM package
import timesfm

# Import data processor from the project
from src.data_processor import DataProcessor

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')

# ==============================================================================
# CONFIGURATION SETTINGS
# ==============================================================================

# Hardware Configuration
print("\n🔧 Checking hardware configuration...")
if torch.cuda.is_available():
    TIMESFM_BACKEND = "gpu"
    print(f"   ✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    TIMESFM_BACKEND = "cpu"
    print("   ⚠️  No GPU detected, using CPU (predictions will be slower)")

# Model Configuration
TIMESFM_MODEL_REPO = "google/timesfm-1.0-200m-pytorch"  # 200M parameter PyTorch model
print(f"   📦 Model: {TIMESFM_MODEL_REPO}")

# Data Configuration
TARGET_DATE = '2025-05-19'  # Trading day to analyze
START_TIME = '06:30'        # Market open in PT
END_TIME = '11:54'          # End time in PT (ensures predictions stay before 1 PM)
USER_TIMEZONE = 'US/Pacific'

# Prediction Settings
CONTEXT_LENGTH = 448        # ~7.5 hours of minute-level historical data
HORIZON_LENGTH = 64         # Predict 64 minutes ahead (~1 hour)
USE_NORMALIZED_PRICES = True  # Normalized prices improve model performance
HISTORICAL_SAMPLES = 448    # Must match CONTEXT_LENGTH

print(f"\n📈 Prediction Configuration:")
print(f"   Context window: {CONTEXT_LENGTH} minutes (~{CONTEXT_LENGTH/60:.1f} hours)")
print(f"   Prediction horizon: {HORIZON_LENGTH} minutes (~{HORIZON_LENGTH/60:.1f} hours)")
print(f"   Using normalized prices: {USE_NORMALIZED_PRICES}")

# Output Configuration
script_dir = Path(__file__).parent.absolute()
PLOT_DIR = script_dir / "stock_plots"
try:
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"   ✅ Output directory: {PLOT_DIR}")
    # Test write permissions
    test_file = PLOT_DIR / ".test_write"
    test_file.touch()
    test_file.unlink()
    print(f"   ✅ Write permissions verified")
except Exception as e:
    print(f"   ❌ Error creating output directory: {e}")
    PLOT_DIR = Path.cwd() / "stock_plots"
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"   📁 Using fallback directory: {PLOT_DIR}")

# ==============================================================================

def load_stock_data(target_date, start_time_str, end_time_str):
    """
    Load and prepare ES futures data for the specified date and time range.
    
    Args:
        target_date: Date string in YYYY-MM-DD format
        start_time_str: Start time in HH:MM format (Pacific Time)
        end_time_str: End time in HH:MM format (Pacific Time)
        
    Returns:
        tuple: (full_day_data, time_range_data, est_start, est_end)
    """
    print(f"\n{'='*60}")
    print(f"📊 LOADING STOCK DATA")
    print(f"{'='*60}")
    
    # Configure data paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(script_dir, 'databento/ES/')
    DEFAULT_DATA_FILE = 'glbx-mdp3-20100606-20250822.ohlcv-1m.csv'
    
    print(f"📁 Data source: {DATA_PATH}")
    print(f"📄 Data file: {DEFAULT_DATA_FILE}")
    print(f"📅 Target date: {target_date}")
    print(f"⏰ Time range: {start_time_str} to {end_time_str} PT")
    
    # Initialize data processor
    print("\n🔄 Initializing data processor...")
    data_processor = DataProcessor(DATA_PATH)
    
    # Load full trading day data
    print("📥 Loading full day data (without indicators for speed)...")
    full_day_data = data_processor.get_processed_trading_day(
        DEFAULT_DATA_FILE, 
        target_date, 
        include_indicators=False
    )
    
    if full_day_data is None or len(full_day_data) == 0:
        raise ValueError(f"❌ No data found for {target_date}")
    
    print(f"   ✅ Loaded {len(full_day_data):,} minute bars for full trading day")
    
    # Set up timezone converters
    pst_tz = pytz.timezone(USER_TIMEZONE)
    est_tz = pytz.timezone('US/Eastern')
    
    # Parse start and end times
    start_hour, start_minute = map(int, start_time_str.split(':'))
    end_hour, end_minute = map(int, end_time_str.split(':'))
    
    target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
    
    # Convert PT to EST for data slicing (market data is in EST)
    print("\n🕐 Converting time zones (PT → EST)...")
    pst_start = pst_tz.localize(
        datetime.combine(target_date_obj.date(), datetime.min.time()).replace(hour=start_hour, minute=start_minute)
    )
    pst_end = pst_tz.localize(
        datetime.combine(target_date_obj.date(), datetime.min.time()).replace(hour=end_hour, minute=end_minute)
    )
    
    est_start = pst_start.astimezone(est_tz)
    est_end = pst_end.astimezone(est_tz)
    
    print(f"   PT: {pst_start.strftime('%I:%M %p')} → {pst_end.strftime('%I:%M %p')}")
    print(f"   EST: {est_start.strftime('%I:%M %p')} → {est_end.strftime('%I:%M %p')}")
    
    # Filter data to time range
    print("\n🔍 Filtering data to specified time range...")
    mask = (full_day_data.index >= est_start) & (full_day_data.index <= est_end)
    time_range_data = full_day_data[mask]
    
    print(f"   ✅ Extracted {len(time_range_data):,} data points")
    print(f"   📊 Data range: {time_range_data.index[0].strftime('%I:%M %p')} to {time_range_data.index[-1].strftime('%I:%M %p')} EST")
    
    # Display data statistics
    if USE_NORMALIZED_PRICES:
        price_col = 'close_norm'
        print(f"\n📈 Normalized price statistics:")
    else:
        price_col = 'close'
        print(f"\n📈 Price statistics:")
        
    print(f"   Min: {time_range_data[price_col].min():.4f}")
    print(f"   Max: {time_range_data[price_col].max():.4f}")
    print(f"   Mean: {time_range_data[price_col].mean():.4f}")
    print(f"   Std: {time_range_data[price_col].std():.4f}")
    
    return full_day_data, time_range_data, est_start, est_end

def prepare_context_data(data, context_length, use_normalized=True):
    """
    Prepare historical context data for TimesFM prediction.
    
    Args:
        data: DataFrame with price data
        context_length: Number of historical points needed
        use_normalized: Whether to use normalized prices
        
    Returns:
        numpy array of context data
    """
    # Select price column
    price_col = 'close_norm' if use_normalized else 'close'
    
    if len(data) < context_length:
        print(f"   ⚠️  Warning: Only {len(data)} points available, need {context_length}")
        print(f"   📌 Padding with edge values to reach required length")
        
        prices = data[price_col].values
        padding_needed = context_length - len(prices)
        padded_prices = np.concatenate([np.full(padding_needed, prices[0]), prices])
        
        print(f"   ✅ Padded {padding_needed} points at the beginning")
        return padded_prices
    
    # Take the last context_length points
    recent_data = data.tail(context_length)
    prices = recent_data[price_col].values
    
    # Display context statistics
    print(f"   📊 Context stats - Min: {prices.min():.4f}, Max: {prices.max():.4f}, Mean: {prices.mean():.4f}")
    
    return prices

def make_predictions(model, full_day_data, prediction_times):
    """
    Generate predictions at multiple time points throughout the trading day.
    
    Args:
        model: Loaded TimesFM model
        full_day_data: Complete day's trading data
        prediction_times: List of (timestamp, time_string) tuples
        
    Returns:
        List of prediction dictionaries
    """
    predictions = []
    print(f"\n{'='*60}")
    print(f"🔮 GENERATING PREDICTIONS")
    print(f"{'='*60}")
    
    for i, (pred_time, pred_time_str) in enumerate(prediction_times, 1):
        print(f"\n{'─'*50}")
        print(f"📍 Prediction {i}/{len(prediction_times)} at {pred_time_str}")
        print(f"{'─'*50}")
        
        # Get historical data up to prediction time
        mask = full_day_data.index <= pred_time
        historical_data = full_day_data[mask]
        
        print(f"   📊 Available historical data: {len(historical_data)} points")
        
        if len(historical_data) < 100:
            print(f"   ⚠️  SKIPPING - Need at least 100 points, only have {len(historical_data)}")
            continue
        
        # Prepare context window
        print(f"\n   🔧 Preparing context window ({CONTEXT_LENGTH} points)...")
        context = prepare_context_data(historical_data, CONTEXT_LENGTH, USE_NORMALIZED_PRICES)
        last_price = context[-1]
        
        print(f"   📈 Last observed price: {last_price:.4f}")
        print(f"   🎯 Predicting next {HORIZON_LENGTH} minutes...")
        
        # Make prediction using TimesFM
        inputs = [context.tolist()]  # Batch of 1
        freq = [0]  # Frequency indicator (0 for minute-level data)
        
        try:
            # Generate forecast
            forecast, _ = model.forecast(inputs, freq)
            forecast = forecast[0][:HORIZON_LENGTH]  # Extract first series, limit to horizon
            
            print(f"   ✅ Forecast generated successfully")
            
            # Analyze the prediction
            final_price = forecast[-1]
            price_change = final_price - last_price
            price_change_pct = (price_change / last_price) * 100
            
            # Determine direction
            if price_change_pct > 0.01:
                direction = "UP"
            elif price_change_pct < -0.01:
                direction = "DOWN"
            else:
                direction = "FLAT"
            
            print(f"   ✅ Prediction: {direction} ({price_change_pct:.2f}%)")
            print(f"   Final predicted price: {final_price:.4f}")
            
            # Get ground truth data (actual future values)
            future_mask = (full_day_data.index > pred_time) & (full_day_data.index <= pred_time + timedelta(minutes=HORIZON_LENGTH))
            future_data = full_day_data[future_mask]
            
            if len(future_data) > 0:
                if USE_NORMALIZED_PRICES:
                    ground_truth = future_data['close_norm'].values[:HORIZON_LENGTH]
                else:
                    ground_truth = future_data['close'].values[:HORIZON_LENGTH]
            else:
                ground_truth = None
            
            predictions.append({
                'time': pred_time,
                'time_str': pred_time_str,
                'context': context,
                'last_price': last_price,
                'forecast': forecast,
                'ground_truth': ground_truth,
                'direction': direction,
                'expected_move_pct': price_change_pct
            })
            
        except Exception as e:
            print(f"   ❌ Error making prediction: {e}")
            continue
    
    return predictions

def plot_predictions(predictions, full_day_data):
    """
    Create comprehensive visualizations of TimesFM predictions.
    
    Generates:
    - Main plot showing all predictions on price chart
    - Individual plots for each prediction with ground truth comparison
    
    Args:
        predictions: List of prediction dictionaries
        full_day_data: Complete day's trading data
    """
    print(f"\n{'='*60}")
    print(f"📊 CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Import matplotlib date formatting
    import matplotlib.dates as mdates
    
    print(f"\n📈 Creating main prediction overview plot...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
    
    # Main plot
    ax1 = axes[0]
    
    # Plot actual price data
    if USE_NORMALIZED_PRICES:
        prices = full_day_data['close_norm'].values
        ylabel = 'Normalized Price'
    else:
        prices = full_day_data['close'].values
        ylabel = 'Price ($)'
    
    timestamps = full_day_data.index
    
    # Convert timestamps to PT for display
    pst_tz = pytz.timezone(USER_TIMEZONE)
    timestamps_pt = [ts.astimezone(pst_tz) for ts in timestamps]
    
    # Plot historical prices
    ax1.plot(timestamps, prices, 'b-', linewidth=2, label='Actual Price', alpha=0.8)
    
    # Add reference line
    if USE_NORMALIZED_PRICES:
        ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Previous Close (1.00)')
    
    # Color map for predictions
    colors = plt.cm.rainbow(np.linspace(0, 1, len(predictions)))
    
    # Plot each prediction
    for i, pred in enumerate(predictions):
        # Create timestamps for prediction
        pred_timestamps = []
        start_time = pred['time']
        for j in range(len(pred['forecast'])):
            pred_timestamps.append(start_time + timedelta(minutes=j+1))
        
        # Plot prediction
        ax1.plot(pred_timestamps, pred['forecast'], 
                color=colors[i], alpha=0.8, linewidth=2,
                label=f"{pred['time_str']} - {pred['direction']} ({pred['expected_move_pct']:.1f}%)")
        
        # Mark prediction start point
        ax1.scatter(start_time, pred['last_price'], 
                   color=colors[i], s=100, zorder=5, edgecolor='black', linewidth=2)
        
        # Add vertical line at prediction time
        ax1.axvline(x=start_time, color=colors[i], linestyle='--', alpha=0.3)
    
    # Format main plot
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(f'TimesFM Stock Predictions - {TARGET_DATE}', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # Set x-axis to show PT times
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p', tz=pst_tz))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    
    # Direction matrix plot
    ax2 = axes[1]
    
    # Create direction bars
    bar_height = 0.8
    dir_colors = {'UP': 'green', 'DOWN': 'red', 'FLAT': 'gray'}
    
    for i, pred in enumerate(predictions):
        color = dir_colors[pred['direction']]
        bar_width = timedelta(minutes=30)  # 30-minute bars
        
        ax2.barh(0, bar_width.total_seconds()/86400, 
                left=pred['time'], height=bar_height,
                color=color, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add text label
        ax2.text(pred['time'] + timedelta(minutes=15), 0, pred['direction'],
                ha='center', va='center', color='white', fontsize=10, weight='bold')
    
    # Format direction plot
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([0])
    ax2.set_yticklabels(['Prediction'], fontsize=12)
    ax2.set_xlabel('Time (PT)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p', tz=pst_tz))
    
    # Share x-axis
    ax2.sharex(ax1)
    
    # Add legend for direction colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='UP'),
        Patch(facecolor='red', edgecolor='black', label='DOWN'),
        Patch(facecolor='gray', edgecolor='black', label='FLAT')
    ]
    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    
    plt.tight_layout()
    
    # Save main plot with error handling
    output_path = PLOT_DIR / 'timefm_stock_predictions.png'
    try:
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        # Verify file was saved
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"\n✅ Main plot saved successfully:")
            print(f"   Path: {output_path}")
            print(f"   Size: {file_size:,} bytes")
        else:
            print(f"\n❌ Failed to save main plot to {output_path}")
    except Exception as e:
        print(f"\n❌ Error saving main plot: {e}")
    
    # Save individual prediction plots
    print(f"\n📊 Creating individual prediction plots...")
    for i, pred in enumerate(predictions):
        print(f"\n   Plot {i+1}/{len(predictions)}: {pred['time_str']}")
        fig_ind, ax_ind = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot context
        context_timestamps = []
        context_start_idx = len(pred['context'])
        for j in range(context_start_idx):
            context_timestamps.append(pred['time'] - timedelta(minutes=context_start_idx-j))
        
        ax_ind.plot(context_timestamps, pred['context'], 'gray', 
                   linewidth=2, alpha=0.7, label='Historical Context')
        
        # Plot prediction
        pred_timestamps = []
        for j in range(len(pred['forecast'])):
            pred_timestamps.append(pred['time'] + timedelta(minutes=j+1))
        
        ax_ind.plot(pred_timestamps, pred['forecast'], 'red', 
                   linewidth=2.5, label=f'Prediction - {pred["direction"]}')
        
        # Plot ground truth if available
        if pred.get('ground_truth') is not None and len(pred['ground_truth']) > 0:
            # Only plot as many ground truth points as we have
            gt_timestamps = pred_timestamps[:len(pred['ground_truth'])]
            ax_ind.plot(gt_timestamps, pred['ground_truth'], 'green', 
                       linewidth=2.5, label='Ground Truth', alpha=0.8)
        
        # Connect context to prediction
        ax_ind.plot([context_timestamps[-1], pred_timestamps[0]], 
                   [pred['context'][-1], pred['forecast'][0]], 
                   'black', linewidth=1, alpha=0.3)
        
        # Add vertical line
        ax_ind.axvline(x=pred['time'], color='blue', linestyle='--', alpha=0.5, 
                      label='Prediction Start')
        
        # Formatting
        ax_ind.set_xlabel('Time', fontsize=12)
        ax_ind.set_ylabel(ylabel, fontsize=12)
        ax_ind.set_title(f'TimesFM Prediction at {pred["time_str"]} - {pred["direction"]} ({pred["expected_move_pct"]:.2f}%)', 
                        fontsize=14, fontweight='bold')
        ax_ind.grid(True, alpha=0.3)
        ax_ind.legend()
        ax_ind.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p', tz=pst_tz))
        
        # Add MSE if ground truth is available
        if pred.get('ground_truth') is not None and len(pred['ground_truth']) > 0:
            # Calculate MSE for overlapping points
            min_len = min(len(pred['forecast']), len(pred['ground_truth']))
            mse = np.mean((pred['forecast'][:min_len] - pred['ground_truth'][:min_len]) ** 2)
            mae = np.mean(np.abs(pred['forecast'][:min_len] - pred['ground_truth'][:min_len]))
            
            # Add metrics to plot
            textstr = f'MSE: {mse:.6f}\nMAE: {mae:.6f}'
            ax_ind.text(0.02, 0.98, textstr, transform=ax_ind.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save individual plot with error handling
        ind_filename = f'timefm_prediction_{i+1}_{pred["time_str"].replace(":", "")}.png'
        ind_path = PLOT_DIR / ind_filename
        
        try:
            plt.savefig(str(ind_path), dpi=150, bbox_inches='tight')
            plt.close(fig_ind)
            
            # Verify file was saved
            if ind_path.exists():
                file_size = ind_path.stat().st_size
                print(f"   ✅ Saved: {ind_filename} ({file_size:,} bytes)")
            else:
                print(f"   ❌ Failed to save: {ind_filename}")
        except Exception as e:
            print(f"   ❌ Error saving {ind_filename}: {e}")
    
    print(f"\n📊 All plots saved to: {PLOT_DIR}")

def main():
    """Main function to run TimesFM stock predictions."""
    from datetime import datetime
    
    print("\n🚀 TimesFM Stock Prediction")
    print("=" * 60)
    
    # Load stock data first
    full_day_data, time_range_data, est_start, est_end = load_stock_data(
        TARGET_DATE, START_TIME, END_TIME
    )
    
    # Initialize TimesFM model after data is loaded successfully
    print(f"\n📊 Initializing TimesFM model on {TIMESFM_BACKEND}...")
    
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=TIMESFM_BACKEND,
            per_core_batch_size=32,
            horizon_len=128,  # Model's default
            num_layers=20,
            use_positional_embedding=True,
            context_len=512,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id=TIMESFM_MODEL_REPO
        ),
    )
    
    print(f"✅ Model loaded successfully on {TIMESFM_BACKEND}")
    
    # Define prediction times
    pst_tz = pytz.timezone(USER_TIMEZONE)
    prediction_times = []
    
    # Create predictions every 30 minutes, ensuring we don't predict past 1:00 PM
    current_time = est_start + timedelta(minutes=30)  # Start 30 minutes after market open (7:00 AM)
    # Convert 11:56 AM PT to ET for the max prediction time
    est_tz = pytz.timezone('US/Eastern')
    max_time_pt = pst_tz.localize(datetime.strptime(f"{TARGET_DATE} 11:56", "%Y-%m-%d %H:%M"))
    max_prediction_time = max_time_pt.astimezone(est_tz)
    
    while current_time <= min(est_end, max_prediction_time):
        time_str = current_time.astimezone(pst_tz).strftime('%I:%M %p')
        prediction_times.append((current_time, time_str))
        current_time += timedelta(minutes=30)
    
    print(f"\n📅 Will make {len(prediction_times)} predictions")
    
    # Make predictions
    predictions = make_predictions(model, full_day_data, prediction_times)
    
    if len(predictions) == 0:
        print("\n❌ No predictions were generated")
        return
    
    print(f"\n✅ Generated {len(predictions)} predictions")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    plot_predictions(predictions, time_range_data)
    
    # Summary
    print("\n📈 Prediction Summary:")
    print("=" * 40)
    for pred in predictions:
        print(f"{pred['time_str']}: {pred['direction']} ({pred['expected_move_pct']:.2f}%)")
    
    # Calculate statistics
    up_count = sum(1 for p in predictions if p['direction'] == 'UP')
    down_count = sum(1 for p in predictions if p['direction'] == 'DOWN')
    flat_count = sum(1 for p in predictions if p['direction'] == 'FLAT')
    
    print("\n📊 Direction Distribution:")
    print(f"   UP:   {up_count} ({up_count/len(predictions)*100:.1f}%)")
    print(f"   DOWN: {down_count} ({down_count/len(predictions)*100:.1f}%)")
    print(f"   FLAT: {flat_count} ({flat_count/len(predictions)*100:.1f}%)")
    
    # List saved files
    print(f"\n📁 Output Files:")
    print(f"   Directory: {PLOT_DIR}")
    
    saved_files = list(PLOT_DIR.glob("*.png"))
    if saved_files:
        print(f"   Total files saved: {len(saved_files)}")
        for file in sorted(saved_files):
            size = file.stat().st_size
            print(f"   ✓ {file.name} ({size:,} bytes)")
    else:
        print("   ⚠️ No files were saved - check permissions!")
    
    print("\n✨ TimesFM stock prediction complete!")

if __name__ == "__main__":
    main()
