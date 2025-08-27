"""
FlaMinGo Loop Predictor - Runs predictions in a loop, advancing time by 5 minutes
and stacking all predictions to visualize how forecasts evolve over time.
Based on chronos_loop.py but using FlaMinGo TimesFM instead of Chronos.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))
from data_processor import DataProcessor
from config import get_config
import pytz

# Import FlaMinGo predictor
from flamingo_hf_predictor import FlaMinGoHuggingFacePredictor

# ==============================================================================
# USER CONFIGURATION - EDIT THESE VALUES
# ==============================================================================
START_TIME = '7:00'  # Starting NOW time in PT (market open)
END_TIME = '12:00'    # Ending NOW time in PT (market close)
TARGET_DATE = '2025-05-19'  # Set the date to analyze
USER_TIMEZONE = 'US/Pacific'  # Your local timezone

# Loop settings
TIME_STEP_MINUTES = 15  # Move NOW time forward by this many minutes each loop
PREDICTION_FREQUENCY = '1min'  # Resolution of predictions
ANIMATION_DELAY = 3  # Seconds to pause between predictions for visibility

# Data settings
HISTORICAL_SAMPLES = 800  # Number of historical samples to feed to FlaMinGo
                          # Increased for TimesFM 2.0's larger context capacity (up to 2048)
                          # 800 samples = ~13.3 hours of 1-minute data
USE_NORMALIZED_PRICES = True  # Use normalized prices for consistent scale

# Directional correctness thresholds (more sensitive for financial data)
FLAT_THRESHOLD_PCT = 0.01  # Movement within ±0.01% is considered FLAT

# FlaMinGo TimesFM Model Configuration
FLAMINGO_MODEL_SIZE = "500m-v2"  # Model size - TimesFM 2.0 options:
                                 # '200m-v2': TimesFM 2.0 200M (faster)
                                 # '500m-v2': TimesFM 2.0 500M (more accurate)
                                 # '200m': TimesFM 1.0 200M (legacy)
                                 # '200m-v1': TimesFM 1.0 200M (explicit legacy)
FLAMINGO_MODEL_REPO = None  # Custom model repo (None = use default for model size)

# TimesFM Core Hyperparameters (affects model architecture and performance)
# TUNING TIPS FOR FINANCIAL DATA:
# - Increase CONTEXT_LEN for more historical context (256-1024 for intraday, 128-512 for daily)
# - Decrease INPUT_PATCH_LEN for finer granularity (16-32 for minute data, 32-64 for hourly)
# - Adjust NUM_HEADS based on complexity (8-16 for simple patterns, 16-32 for complex)

TIMESFM_CONTEXT_LEN = 1024     # Max historical data points to consider
                               # TimesFM 2.0 supports up to 2048 context length!
                               # Higher = more context but slower inference
                               # Financial: 512-2048 for intraday, 256-1024 for daily data

TIMESFM_HORIZON_LEN = 128      # Internal forecast horizon (model training parameter)
                               # Should be >= your max prediction length
                               # Financial: 64-128 for short-term, 128-256 for longer horizons

TIMESFM_INPUT_PATCH_LEN = 32   # How data is chunked for processing
                               # Smaller = finer granularity, larger = more efficient
                               # Financial: 16-32 for minute data, 32-64 for hourly/daily

TIMESFM_OUTPUT_PATCH_LEN = 128 # Autoregressive decoding step size
                               # Should match or exceed HORIZON_LEN
                               # Financial: Keep at 128 unless you need longer predictions

TIMESFM_NUM_HEADS = 16         # Number of attention heads in transformer
                               # More heads = more complex patterns but slower
                               # Financial: 8-16 for simple trends, 16-32 for complex patterns

TIMESFM_RMS_NORM_EPS = 1e-6    # RMS normalization epsilon (numerical stability)
                               # Lower = more precise, higher = more stable
                               # Financial: 1e-6 to 1e-8 for price data

TIMESFM_USE_POSITIONAL_EMBEDDING = False  # Use positional embeddings
                                          # False for TimesFM 2.0 models, True for v1.0
                                          # TimesFM 2.0 uses different positional encoding

# TimesFM Processing Parameters
TIMESFM_PER_CORE_BATCH_SIZE = 32  # Batch size per core (affects memory usage)
                                  # Lower if GPU memory issues, higher for faster processing
                                  # Financial: 16-64 depending on available memory

TIMESFM_BACKEND = "auto"          # Computing backend: "auto", "cpu", "gpu", or "tpu"
                                  # "auto" chooses best available
                                  # Financial: Use "gpu" if available for faster inference

TIMESFM_POINT_FORECAST_MODE = "median"  # Point forecast aggregation: "mean" or "median"
                                        # "median" more robust to outliers
                                        # Financial: "median" for noisy data, "mean" for smooth data

# TimesFM Quantile Configuration (for uncertainty estimation)
TIMESFM_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  
# Quantiles for confidence intervals
# Financial: Include 0.05, 0.95 for 90% confidence, 0.01, 0.99 for 98% confidence

# Advanced TimesFM Parameters (rarely need adjustment)
TIMESFM_PAD_VAL = 1123581321.0    # Padding value for missing data points
TIMESFM_TOLERANCE = 1e-6          # Numerical tolerance for computations
TIMESFM_DTYPE = "bfloat32"        # Model weights precision ("float32", "bfloat32", "float16")

# ==============================================================================
# RECOMMENDED PARAMETER COMBINATIONS FOR FINANCIAL DATA (TimesFM 2.0):
# ==============================================================================
# 
# HIGH-FREQUENCY TRADING (1-minute data, short predictions):
# FLAMINGO_MODEL_SIZE = "200m-v2", TIMESFM_CONTEXT_LEN = 512
# INPUT_PATCH_LEN = 16, NUM_HEADS = 8, HORIZON_LEN = 64
# HISTORICAL_SAMPLES = 400
#
# INTRADAY TRADING (5-15 minute data, 1-4 hour predictions):
# FLAMINGO_MODEL_SIZE = "500m-v2", TIMESFM_CONTEXT_LEN = 1024  
# INPUT_PATCH_LEN = 32, NUM_HEADS = 16, HORIZON_LEN = 128 (current defaults)
# HISTORICAL_SAMPLES = 800
#
# SWING TRADING (hourly data, daily predictions):
# FLAMINGO_MODEL_SIZE = "500m-v2", TIMESFM_CONTEXT_LEN = 2048
# INPUT_PATCH_LEN = 64, NUM_HEADS = 24, HORIZON_LEN = 256
# HISTORICAL_SAMPLES = 1600
#
# MAXIMUM CONTEXT (TimesFM 2.0 full capability):
# FLAMINGO_MODEL_SIZE = "500m-v2", TIMESFM_CONTEXT_LEN = 2048
# HISTORICAL_SAMPLES = 2000, INPUT_PATCH_LEN = 32
#
# VOLATILE MARKETS (high noise, need robustness):
# POINT_FORECAST_MODE = "median", RMS_NORM_EPS = 1e-5
# QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
#
# STABLE MARKETS (low noise, need precision):
# POINT_FORECAST_MODE = "mean", RMS_NORM_EPS = 1e-8
# QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# ==============================================================================

def create_initial_plot(full_day_data, start_time_str, target_date):
    """Create the initial plot setup with two subplots showing data up to start time"""
    import matplotlib.gridspec as gridspec
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    # Set up timezone converters
    pst_tz = pytz.timezone('US/Pacific')
    est_tz = pytz.timezone('US/Eastern')
    
    # Parse start time and convert to EST
    start_hour, start_minute = map(int, start_time_str.split(':'))
    target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
    pst_datetime = pst_tz.localize(
        datetime.combine(target_date_obj.date(), datetime.min.time()).replace(hour=start_hour, minute=start_minute)
    )
    est_start_time = pst_datetime.astimezone(est_tz)
    
    # Filter data to start NOW time - last N samples
    mask = full_day_data.index <= est_start_time
    initial_data_all = full_day_data[mask]
    # Take last N samples based on user configuration
    initial_data = initial_data_all.tail(HISTORICAL_SAMPLES)
    
    # Plot only the historical data up to NOW
    if len(initial_data) > 0:
        prices = initial_data['close_norm'].values if USE_NORMALIZED_PRICES else initial_data['close'].values
        timestamps = initial_data.index
        
        ax1.plot(timestamps, prices, 'b-', linewidth=1.5, label='Historical Price', alpha=0.9)
    else:
        timestamps = []  # Initialize empty if no data
    
    if USE_NORMALIZED_PRICES:
        ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Previous EOD (1.00)')
    
    # Add vertical line at market open if it's within the data range
    market_open_et = est_tz.localize(datetime(target_date_obj.year, target_date_obj.month, target_date_obj.day, 9, 30, 0))
    if len(initial_data) > 0 and market_open_et >= initial_data.index[0] and market_open_et <= initial_data.index[-1]:
        ax1.axvline(x=market_open_et, color='green', linestyle='--', alpha=0.3, label='Market Open (6:30 AM PT)')
    
    # Format x-axis
    def format_pt(x, pos=None):
        dt = mdates.num2date(x)
        if dt.tzinfo is None:
            dt_est = pytz.timezone('US/Eastern').localize(dt)
        else:
            dt_est = dt
        dt_pt = dt_est.astimezone(pytz.timezone('US/Pacific'))
        return dt_pt.strftime('%I:%M %p')
    
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_pt))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    
    # Set x-axis limits - based on rolling window  
    if len(timestamps) > 0:
        start_time = timestamps[0]  # Start from beginning of rolling window
        end_time = timestamps[-1] + pd.Timedelta(minutes=15)
        ax1.set_xlim(start_time, end_time)
        ax2.set_xlim(start_time, end_time)
    
    # Labels for main plot
    price_label = 'Normalized Price' if USE_NORMALIZED_PRICES else 'Price ($)'
    ax1.set_ylabel(price_label, fontsize=12)
    ax1.set_title(f'FlaMinGo TimesFM Real-Time Predictions - {TARGET_DATE}', fontsize=16, fontweight='bold')
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
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    return fig, (ax1, ax2)

def update_plot(fig, ax1, ax2, prediction_entry, full_day_data, loop_count, all_predictions):
    """Update the plot with new prediction data"""
    
    # Color for this prediction
    colors = plt.cm.rainbow(np.linspace(0, 1, 20))  # Support up to 20 predictions
    color = colors[loop_count % 20]
    
    # First, extend the historical price line to current NOW time - last N samples
    start_time = prediction_entry['now_time_est']
    est_tz = pytz.timezone('US/Eastern')
    mask = full_day_data.index <= start_time
    historical_all = full_day_data[mask]
    # Take last N samples based on user configuration
    historical_to_now = historical_all.tail(HISTORICAL_SAMPLES)
    
    if len(historical_to_now) > 0:
        # Clear and redraw historical price to current point
        ax1.clear()
        prices = historical_to_now['close_norm'] if USE_NORMALIZED_PRICES else historical_to_now['close']
        ax1.plot(historical_to_now.index, prices, 
                'b-', linewidth=1.5, label='Historical Price', alpha=0.9)
        
        if USE_NORMALIZED_PRICES:
            ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Previous EOD (1.00)')
        
        # Only show market open line if it's in the current window
        market_open_et = est_tz.localize(datetime(start_time.year, start_time.month, start_time.day, 9, 30, 0))
        if market_open_et >= historical_to_now.index[0] and market_open_et <= historical_to_now.index[-1]:
            ax1.axvline(x=market_open_et, color='green', linestyle='--', alpha=0.3, label='Market Open')
    
    # Plot all previous predictions (faded)
    for i, prev_pred in enumerate(all_predictions[:-1]):
        prev_color = colors[i % 20]
        prev_timestamps = []
        prev_start = prev_pred['now_time_est']
        for j in range(len(prev_pred['prediction'])):
            prev_timestamps.append(prev_start + timedelta(minutes=j+1))
        ax1.plot(prev_timestamps, prev_pred['prediction'], 
                color=prev_color, alpha=0.3, linewidth=1.0)
        ax1.scatter(prev_start, prev_pred['last_price'], 
                   color=prev_color, s=30, alpha=0.5, zorder=4)
    
    # Plot current prediction (highlighted)
    pred_timestamps = []
    for j in range(len(prediction_entry['prediction'])):
        pred_timestamps.append(start_time + timedelta(minutes=j+1))
    
    ax1.plot(pred_timestamps, prediction_entry['prediction'], 
            color=color, alpha=0.8, linewidth=2.5,
            label=f"{prediction_entry['now_time'].strftime('%I:%M%p')} - {prediction_entry['direction']}")
    
    # Mark prediction start point
    ax1.scatter(start_time, prediction_entry['last_price'], 
               color=color, s=80, zorder=5, edgecolor='black', linewidth=1)
    
    # Add current time marker
    ax1.axvline(x=start_time, color='red', linestyle='--', alpha=0.4, linewidth=2)
    
    # Restore plot formatting
    price_label = 'Normalized Price' if USE_NORMALIZED_PRICES else 'Price ($)'
    ax1.set_ylabel(price_label, fontsize=12)
    ax1.set_title(f'FlaMinGo TimesFM Real-Time Predictions - {TARGET_DATE}', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', labelbottom=False)
    
    # Restore x-axis formatting
    def format_pt(x, pos=None):
        dt = mdates.num2date(x)
        if dt.tzinfo is None:
            dt_est = pytz.timezone('US/Eastern').localize(dt)
        else:
            dt_est = dt
        dt_pt = dt_est.astimezone(pytz.timezone('US/Pacific'))
        return dt_pt.strftime('%I:%M %p')
    
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_pt))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    
    # Update legend
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    
    # Clear and redraw all matrix bars
    ax2.clear()
    
    # Set up matrix subplot again
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Actual', 'Predicted', 'Correct'], fontsize=10)
    ax2.set_xlabel('Time (PT)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Color maps
    dir_colors = {'UP': 'green', 'DOWN': 'red', 'FLAT': 'gray'}
    bar_height = 0.35
    bar_width = timedelta(minutes=TIME_STEP_MINUTES * 0.9)
    
    # Current NOW time for evaluation
    current_now_time = prediction_entry['now_time_est']
    
    # Draw bars for ALL predictions so far
    directionally_correct = None
    for pred in all_predictions:
        pred_time = pred['now_time_est']
        # Use the actual prediction length for this prediction
        actual_pred_length = len(pred['prediction'])
        pred_end_time = pred_time + timedelta(minutes=actual_pred_length)
        
        # Can we evaluate this prediction? Only if NOW has reached 1 hour after prediction start
        one_hour_after_pred = pred_time + timedelta(minutes=60)
        can_evaluate = current_now_time >= one_hour_after_pred
        
        # Always draw the prediction bar
        ax2.barh(1, bar_width.total_seconds()/86400, 
                left=pred_time, height=bar_height, 
                color=dir_colors[pred['direction']], 
                alpha=0.7, edgecolor='black', linewidth=0.5)
        
        if can_evaluate:
            # Check if we've already evaluated this prediction
            if 'evaluated' not in pred:
                # Get actual data at exactly 1 hour after prediction start
                mask = (full_day_data.index > pred_time) & (full_day_data.index <= one_hour_after_pred)
                actual_data = full_day_data[mask]
                
                if len(actual_data) > 0:
                    # Calculate actual move
                    start_price = pred['last_price']
                    end_price = actual_data['close_norm'].iloc[-1] if USE_NORMALIZED_PRICES else actual_data['close'].iloc[-1]
                    actual_move_pct = ((end_price - start_price) / start_price) * 100
                    
                    # Determine actual direction
                    if actual_move_pct > FLAT_THRESHOLD_PCT:
                        actual_sign = 'UP'
                    elif actual_move_pct < -FLAT_THRESHOLD_PCT:
                        actual_sign = 'DOWN'
                    else:
                        actual_sign = 'FLAT'
                    
                    # Check if directionally correct
                    is_correct = False
                    if pred['direction'] == 'UP' and actual_sign == 'UP':
                        is_correct = True
                    elif pred['direction'] == 'DOWN' and actual_sign == 'DOWN':
                        is_correct = True
                    elif pred['direction'] == 'FLAT' and actual_sign == 'FLAT':
                        is_correct = True
                    
                    # Store evaluation results in the prediction
                    pred['evaluated'] = True
                    pred['actual_sign'] = actual_sign
                    pred['actual_move_pct'] = actual_move_pct
                    pred['is_correct'] = is_correct
                    
                    # Print evaluation when it happens
                    print(f"\n  Evaluation complete for {pred['now_time'].strftime('%I:%M %p')} prediction (1-hour mark):")
                    print(f"    Predicted: {pred['direction']}, Actual: {actual_sign} (move: {actual_move_pct:.3f}%)")
                    print(f"    Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            
            # If evaluated (either just now or previously), draw the actual and correctness bars
            if 'evaluated' in pred and pred['evaluated']:
                actual_sign = pred['actual_sign']
                is_correct = pred['is_correct']
                
                # Store result for current prediction if this is it
                if pred == prediction_entry:
                    directionally_correct = is_correct
                
                # Draw actual bar
                ax2.barh(0, bar_width.total_seconds()/86400, 
                        left=pred_time, height=bar_height,
                        color=dir_colors[actual_sign], 
                        alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Draw correctness bar
                correct_color = 'green' if is_correct else 'red'
                ax2.barh(2, bar_width.total_seconds()/86400,
                        left=pred_time, height=bar_height,
                        color=correct_color, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Correctness indicator text
                if is_correct:
                    ax2.text(pred_time + timedelta(minutes=TIME_STEP_MINUTES/2), 2, '✓',
                            ha='center', va='center', color='white', fontsize=6, weight='bold')
                else:
                    ax2.text(pred_time + timedelta(minutes=TIME_STEP_MINUTES/2), 2, '✗',
                            ha='center', va='center', color='white', fontsize=6, weight='bold')
    
    # Add legend for matrix
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='UP / Correct'),
        Patch(facecolor='red', edgecolor='black', label='DOWN / Wrong'),
        Patch(facecolor='gray', edgecolor='black', label='FLAT')
    ]
    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=8)
    
    # Ensure matrix x-axis formatting matches main plot
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_pt))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    
    # Update x-axis limits for both plots to show all predictions
    if len(historical_to_now) > 0 and len(pred_timestamps) > 0:
        # Extend x-axis to show all predictions plus some buffer
        xlim_start = historical_to_now.index[0]
        xlim_end = max(pred_timestamps[-1], start_time) + timedelta(minutes=5)
        ax1.set_xlim(xlim_start, xlim_end)
        ax2.set_xlim(xlim_start, xlim_end)
    
    # Update the plot
    try:
        plt.tight_layout()
    except:
        pass  # Ignore tight_layout warnings
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)  # Small pause to allow plot to update

def analyze_prediction_direction(predictions, current_price):
    """Analyze the directional prediction from FlaMinGo"""
    
    if len(predictions) == 0:
        return "UNKNOWN", 0.0, 0.0
    
    # Calculate expected move
    final_price = predictions[-1]
    price_change = final_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # Determine direction (using same threshold as chronos_loop.py)
    if price_change_pct > FLAT_THRESHOLD_PCT:
        direction = "UP"
    elif price_change_pct < -FLAT_THRESHOLD_PCT:
        direction = "DOWN"
    else:
        direction = "FLAT"
    
    # Calculate confidence based on prediction consistency
    if len(predictions) > 1:
        # Check how consistent the direction is across all predictions
        changes = np.diff(predictions)
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

def run_prediction_loop():
    """Run predictions in a loop, advancing time and collecting results"""
    
    # Initialize
    config = get_config()
    data_processor = DataProcessor(config.DATA_PATH)
    
    print(f"\nRunning FlaMinGo prediction loop")
    print(f"Date: {TARGET_DATE}")
    print(f"Time range: {START_TIME} to {END_TIME} PT")
    print(f"Step: {TIME_STEP_MINUTES} minutes")
    print(f"Horizon: Max 1 hour ahead at {PREDICTION_FREQUENCY} resolution")
    print(f"Model: FlaMinGo TimesFM {FLAMINGO_MODEL_SIZE} (TimesFM 2.0)")
    print("="*60)
    
    # Load full day data once
    full_day_data = data_processor.get_processed_trading_day(
        config.DEFAULT_DATA_FILE, 
        TARGET_DATE, 
        include_indicators=True
    )
    
    # We'll create a new predictor for each loop with dynamic horizon (max 1 hour)
    # (FlaMinGo needs to know prediction length at initialization)
    print("🚀 FlaMinGo TimesFM will be initialized for each prediction with max 1-hour horizon")
    base_predictor = None  # Will initialize per loop
    
    # Enable interactive plotting
    plt.ion()
    
    # Create initial figure
    fig, (ax1, ax2) = create_initial_plot(full_day_data, START_TIME, TARGET_DATE)
    
    # Parse start and end times
    start_hour, start_minute = map(int, START_TIME.split(':'))
    end_hour, end_minute = map(int, END_TIME.split(':'))
    
    # Create timezone objects
    pst_tz = pytz.timezone(USER_TIMEZONE)
    est_tz = pytz.timezone('US/Eastern')
    
    # Calculate market close time (1 PM PT = 4 PM EST)
    target_date_obj = datetime.strptime(TARGET_DATE, '%Y-%m-%d')
    market_close_pst = pst_tz.localize(
        datetime.combine(target_date_obj.date(), datetime.min.time()).replace(hour=13, minute=0)
    )
    market_close_est = market_close_pst.astimezone(est_tz)
    
    # Initialize lists to store results
    all_predictions = []
    
    # Loop through time
    current_time = datetime.strptime(f"{TARGET_DATE} {START_TIME}", '%Y-%m-%d %H:%M')
    end_time = datetime.strptime(f"{TARGET_DATE} {END_TIME}", '%Y-%m-%d %H:%M')
    
    loop_count = 0
    while current_time <= end_time:
        loop_count += 1
        print(f"\n--- Loop {loop_count}: NOW = {current_time.strftime('%I:%M %p')} PT ---")
        
        # Convert PT to EST for data slicing
        pst_datetime = pst_tz.localize(current_time)
        est_datetime = pst_datetime.astimezone(est_tz)
        current_datetime_str = est_datetime.strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate dynamic prediction horizon (max 1 hour ahead)
        minutes_to_close = int((market_close_est - est_datetime).total_seconds() / 60)
        dynamic_horizon = max(1, min(minutes_to_close, 60))  # Cap at 60 minutes (1 hour)
        
        print(f"Minutes until market close: {minutes_to_close}")
        print(f"Prediction horizon: {dynamic_horizon} minutes (max 1 hour)")
        
        if minutes_to_close <= 0:
            print("Market is closed, ending predictions")
            break
        
        # Slice data up to current time
        historical_data = data_processor.slice_data_to_current(full_day_data, current_datetime_str)
        
        if len(historical_data) < 100:
            print(f"Skipping - insufficient data (only {len(historical_data)} points)")
            current_time += timedelta(minutes=TIME_STEP_MINUTES)
            continue
        
        # Prepare data for prediction - take last N samples
        context_data = historical_data.tail(HISTORICAL_SAMPLES)
        timestamps = context_data.index
        
        if USE_NORMALIZED_PRICES:
            price_series = context_data['close_norm']
            current_price = context_data['close_norm'].iloc[-1]
        else:
            price_series = context_data['close']
            current_price = context_data['close'].iloc[-1]
        
        try:
            # Create predictor with dynamic horizon for this loop (max 1 hour)
            print(f"Initializing FlaMinGo for {dynamic_horizon}-minute prediction (max 1 hour)...")
            predictor = FlaMinGoHuggingFacePredictor(
                prediction_length=dynamic_horizon,
                frequency=PREDICTION_FREQUENCY,
                model_size=FLAMINGO_MODEL_SIZE,
                use_hf_models=True,
                model_repo=FLAMINGO_MODEL_REPO,
                # TimesFM hyperparameters
                context_len=TIMESFM_CONTEXT_LEN,
                horizon_len=TIMESFM_HORIZON_LEN,
                input_patch_len=TIMESFM_INPUT_PATCH_LEN,
                output_patch_len=TIMESFM_OUTPUT_PATCH_LEN,
                num_heads=TIMESFM_NUM_HEADS,
                rms_norm_eps=TIMESFM_RMS_NORM_EPS,
                use_positional_embedding=TIMESFM_USE_POSITIONAL_EMBEDDING,
                per_core_batch_size=TIMESFM_PER_CORE_BATCH_SIZE,
                backend=TIMESFM_BACKEND,
                point_forecast_mode=TIMESFM_POINT_FORECAST_MODE,
                quantiles=TIMESFM_QUANTILES,
                pad_val=TIMESFM_PAD_VAL,
                tolerance=TIMESFM_TOLERANCE,
                dtype=TIMESFM_DTYPE
            )
            
            # Generate prediction using FlaMinGo
            result = predictor.predict_forecast(price_series, timestamps)
            
            # Analyze directional prediction
            direction, expected_move_pct, confidence = analyze_prediction_direction(
                result['predictions'], current_price
            )
            
            # Store results
            prediction_entry = {
                'loop': loop_count,
                'now_time': current_time,
                'now_time_est': est_datetime,
                'last_price': current_price,
                'prediction': result['predictions'],
                'direction': direction,
                'confidence': confidence,
                'expected_move_pct': expected_move_pct
            }
            
            all_predictions.append(prediction_entry)
            
            print(f"Last price: {current_price:.4f}")
            print(f"Prediction: {direction} ({expected_move_pct:.2f}%)")
            print(f"Confidence: {confidence:.1%}")
            
            # Update plot in real-time
            update_plot(fig, ax1, ax2, prediction_entry, full_day_data, loop_count-1, all_predictions)
            
            # Add animation delay
            time.sleep(ANIMATION_DELAY)
            
        except Exception as e:
            print(f"Error generating prediction: {e}")
            import traceback
            traceback.print_exc()
        
        # Move to next time step
        current_time += timedelta(minutes=TIME_STEP_MINUTES)
    
    print(f"\nCompleted {loop_count} prediction loops")
    
    # Count how many predictions were evaluated
    evaluated_count = sum(1 for pred in all_predictions if 'evaluated' in pred and pred['evaluated'])
    print(f"Total predictions made: {len(all_predictions)}")
    print(f"Predictions evaluated: {evaluated_count}")
    if evaluated_count < len(all_predictions):
        print(f"Note: {len(all_predictions) - evaluated_count} predictions not yet evaluated (1-hour mark not reached)")
    
    # Save the final plot
    plt.savefig('flamingo_loop_predictions_realtime.png', dpi=150, bbox_inches='tight')
    print("Real-time plot saved as 'flamingo_loop_predictions_realtime.png'")
    
    # Keep plot open for viewing
    plt.ioff()  # Turn off interactive mode
    plt.show()
    
    # Also generate the static stacked predictions plot
    plot_stacked_predictions(all_predictions, full_day_data)
    
    # Analyze prediction accuracy
    analyze_predictions(all_predictions, full_day_data)
    
    return all_predictions

def plot_stacked_predictions(predictions, full_day_data):
    """Plot all predictions stacked on top of each other with accuracy matrix below"""
    
    # Create figure with subplots - main plot and matrix below
    fig = plt.figure(figsize=(18, 12))
    
    # Create gridspec for better subplot control
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    # Main plot
    ax1 = plt.subplot(gs[0])
    
    # Convert to PT for display
    pst_tz = pytz.timezone('US/Pacific')
    est_tz = pytz.timezone('US/Eastern')
    
    # Filter data to only show from market open (9:30 AM ET / 6:30 AM PT)
    target_date_obj = datetime.strptime(TARGET_DATE, '%Y-%m-%d')
    market_open_et = est_tz.localize(datetime(target_date_obj.year, target_date_obj.month, target_date_obj.day, 9, 30, 0))
    full_day_data_filtered = full_day_data[full_day_data.index >= market_open_et]
    
    # Plot actual price data
    prices = full_day_data_filtered['close_norm'] if USE_NORMALIZED_PRICES else full_day_data_filtered['close']
    timestamps = full_day_data_filtered.index
    
    # Plot historical prices in blue
    ax1.plot(timestamps, prices, 'b-', linewidth=2, label='Actual Price', alpha=0.8)
    
    # Color map for predictions
    colors = plt.cm.rainbow(np.linspace(0, 1, len(predictions)))
    
    # Plot each prediction
    for i, pred in enumerate(predictions):
        # Create timestamps for this prediction
        pred_timestamps = []
        start_time = pred['now_time_est']
        
        for j in range(len(pred['prediction'])):
            pred_timestamps.append(start_time + timedelta(minutes=j+1))
        
        # Plot prediction line
        ax1.plot(pred_timestamps, pred['prediction'], 
                color=colors[i], alpha=0.6, linewidth=1.5,
                label=f"{pred['now_time'].strftime('%I:%M%p')} - {pred['direction']}")
        
        # Mark the prediction start point
        ax1.scatter(start_time, pred['last_price'], 
                   color=colors[i], s=50, zorder=5)
    
    # Add reference lines
    if USE_NORMALIZED_PRICES:
        ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Previous EOD (1.00)')
    
    # Add vertical line at market open
    if len(timestamps) > 0:
        if market_open_et >= timestamps[0] and market_open_et <= timestamps[-1]:
            ax1.axvline(x=market_open_et, color='green', linestyle='--', alpha=0.3, label='Market Open (6:30 AM PT)')
    
    # Format x-axis for PT display
    def format_pt(x, pos=None):
        dt = mdates.num2date(x)
        if dt.tzinfo is None:
            dt_est = pytz.timezone('US/Eastern').localize(dt)
        else:
            dt_est = dt
        dt_pt = dt_est.astimezone(pytz.timezone('US/Pacific'))
        return dt_pt.strftime('%I:%M %p')
    
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_pt))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    
    # Set x-axis limits to market hours (6:30 AM PT to 1:00 PM PT for regular hours)
    if len(timestamps) > 0:
        market_close_et = timestamps[0].replace(hour=16, minute=0, second=0)
        
        # Find the actual data range within market hours - start from market open
        start_time = market_open_et  # Always start from market open
        end_time = min(timestamps[-1], market_close_et)
        
        # No padding before market open
        ax1.set_xlim(start_time, end_time + pd.Timedelta(minutes=15))
    
    # Labels and formatting for main plot
    price_label = 'Normalized Price' if USE_NORMALIZED_PRICES else 'Price ($)'
    ax1.set_ylabel(price_label, fontsize=12)
    ax1.set_title(f'FlaMinGo TimesFM Stacked Predictions - {TARGET_DATE} (Market Hours Only)', fontsize=16, fontweight='bold')
    
    # Legend - only show first few to avoid clutter
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:min(10, len(handles))], labels[:min(10, len(labels))], 
               loc='best', fontsize=9, ncol=2)
    
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', labelbottom=False)  # Hide x labels on main plot
    
    # Create matrix table subplot
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    # Calculate accuracy data for each prediction that has been evaluated
    matrix_data = []
    for pred in predictions:
        # Only include predictions that have been evaluated
        if 'evaluated' in pred and pred['evaluated']:
            matrix_data.append({
                'time': pred['now_time_est'],
                'predicted': pred['direction'],
                'actual': pred['actual_sign'],
                'actual_move': pred['actual_move_pct'],
                'correct': pred['is_correct']
            })
    
    # Plot matrix
    if matrix_data:
        times = [d['time'] for d in matrix_data]
        
        # Create color-coded bars for predictions and actuals
        bar_height = 0.35
        y_pred = 1
        y_actual = 0
        
        # Color maps
        dir_colors = {'UP': 'green', 'DOWN': 'red', 'FLAT': 'gray'}
        
        for i, data in enumerate(matrix_data):
            # Calculate bar width - make it slightly less than TIME_STEP to avoid overlap
            bar_width = timedelta(minutes=TIME_STEP_MINUTES * 0.9)
            
            # Prediction bar
            color = dir_colors[data['predicted']]
            ax2.barh(y_pred, bar_width.total_seconds()/86400, 
                    left=data['time'], height=bar_height, 
                    color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Actual bar
            color = dir_colors[data['actual']]
            ax2.barh(y_actual, bar_width.total_seconds()/86400, 
                    left=data['time'], height=bar_height,
                    color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Correctness bar
            correct_color = 'green' if data['correct'] else 'red'
            ax2.barh(2, bar_width.total_seconds()/86400,
                    left=data['time'], height=bar_height,
                    color=correct_color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add correctness indicator
            if data['correct']:
                ax2.text(data['time'] + timedelta(minutes=TIME_STEP_MINUTES/2), 2, '✓',
                        ha='center', va='center', color='white', fontsize=8, weight='bold')
            else:
                ax2.text(data['time'] + timedelta(minutes=TIME_STEP_MINUTES/2), 2, '✗',
                        ha='center', va='center', color='white', fontsize=8, weight='bold')
    
    # Format matrix subplot
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
    
    # Rotate x-axis labels
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('flamingo_loop_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nStacked predictions plot saved as 'flamingo_loop_predictions.png'")

def analyze_predictions(predictions, full_day_data):
    """Analyze prediction accuracy over time"""
    
    results = []
    
    # Only analyze predictions that have been evaluated
    for pred in predictions:
        # Skip if not evaluated yet
        if 'evaluated' not in pred or not pred['evaluated']:
            continue
            
        # Use stored evaluation results
        start_time = pred['now_time_est']
        # Use the stored evaluation results
        actual_direction = pred['actual_sign']
        actual_move_pct = pred['actual_move_pct']
        directionally_correct = pred['is_correct']
        exact_correct = pred['is_correct']  # Same as directionally correct with our threshold logic
        
        results.append({
            'time': pred['now_time'].strftime('%I:%M %p'),
            'predicted': pred['direction'],
            'actual': actual_direction,
            'predicted_move': pred['expected_move_pct'],
            'actual_move': actual_move_pct,
            'directionally_correct': directionally_correct,
            'exact_correct': exact_correct,
            'actual_move_sign': actual_direction
        })
    
    # Calculate statistics
    if results:
        df_results = pd.DataFrame(results)
        directional_accuracy = (df_results['directionally_correct'].sum() / len(df_results)) * 100
        exact_accuracy = (df_results['exact_correct'].sum() / len(df_results)) * 100
        
        print("\n" + "="*60)
        print("FLAMINGO PREDICTION ACCURACY ANALYSIS")
        print("="*60)
        print(f"Total predictions: {len(results)}")
        print(f"\nDirectional Accuracy (main metric):")
        print(f"  Correct: {df_results['directionally_correct'].sum()}/{len(results)}")
        print(f"  Accuracy: {directional_accuracy:.1f}%")
        print(f"\nExact Accuracy ({FLAT_THRESHOLD_PCT}% threshold):")
        print(f"  Correct: {df_results['exact_correct'].sum()}/{len(results)}")  
        print(f"  Accuracy: {exact_accuracy:.1f}%")
        
        # Directional accuracy by predicted direction
        print("\nDirectional Accuracy by Prediction Type:")
        for direction in ['UP', 'DOWN', 'FLAT']:
            dir_mask = df_results['predicted'] == direction
            if dir_mask.any():
                dir_correct = df_results[dir_mask]['directionally_correct'].sum()
                dir_total = dir_mask.sum()
                dir_accuracy = (dir_correct / dir_total) * 100
                print(f"  {direction}: {dir_accuracy:.1f}% ({dir_correct}/{dir_total} predictions)")
        
        # Create accuracy plot
        plt.figure(figsize=(12, 6))
        
        # Plot rolling accuracy
        window = min(5, len(df_results) // 2)
        if window > 1:
            rolling_accuracy = df_results['directionally_correct'].rolling(window=window).mean() * 100
            plt.plot(range(len(df_results)), rolling_accuracy, 'b-', linewidth=2, 
                    label=f'{window}-prediction rolling accuracy')
        
        # Plot individual results
        correct_mask = df_results['directionally_correct']
        plt.scatter(np.where(correct_mask)[0], [105]*correct_mask.sum(), 
                   color='green', marker='^', s=100, label='Correct', alpha=0.7)
        plt.scatter(np.where(~correct_mask)[0], [95]*(~correct_mask).sum(), 
                   color='red', marker='v', s=100, label='Wrong', alpha=0.7)
        
        # Add horizontal line at 50% (random guess)
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
        
        # Labels
        plt.xlabel('Prediction Number', fontsize=12)
        plt.ylabel('Accuracy %', fontsize=12)
        plt.title('FlaMinGo Prediction Accuracy Over Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 110)
        
        # Add time labels
        step = max(1, len(df_results) // 10)
        plt.xticks(range(0, len(df_results), step), 
                  [df_results.iloc[i]['time'] for i in range(0, len(df_results), step)],
                  rotation=45)
        
        plt.tight_layout()
        plt.savefig('flamingo_loop_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Accuracy plot saved as 'flamingo_loop_accuracy.png'")

if __name__ == "__main__":
    # Run the prediction loop
    predictions = run_prediction_loop()
