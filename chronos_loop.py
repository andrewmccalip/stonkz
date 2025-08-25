"""
Chronos Loop Predictor - Runs predictions in a loop, advancing time by 5 minutes
and stacking all predictions to visualize how forecasts evolve over time.
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
from src.data_processor import DataProcessor
from config import get_config
import pytz

# Import ChronosPredictor from chronos.py
from chronos import ChronosPredictor

# ==============================================================================
# USER CONFIGURATION - EDIT THESE VALUES
# ==============================================================================
START_TIME = '07:00'  # Starting NOW time in PT (market open)
END_TIME = '8:00'    # Ending NOW time in PT (reduced for demo)
TARGET_DATE = '2025-05-19'  # Set the date to analyze
USER_TIMEZONE = 'US/Pacific'  # Your local timezone

# Loop settings
TIME_STEP_MINUTES = 5  # Move NOW time forward by this many minutes each loop
PREDICTION_HORIZON_MINUTES = 15  # Predict this many minutes ahead (shorter for more accurate predictions)
PREDICTION_FREQUENCY = '1min'  # Resolution of predictions
ANIMATION_DELAY = 1.0  # Seconds to pause between predictions for visibility

# Data settings
HISTORICAL_SAMPLES = 200  # Number of historical samples to feed to Chronos (shorter for recent patterns)
USE_RAW_PRICES = False  # Use normalized prices for consistent scale

# Directional correctness thresholds (more sensitive for financial data)
FLAT_THRESHOLD_PCT = 0.01  # Movement within ±0.01% is considered FLAT

# Chronos hyperparameters (optimized for financial time series)
CHRONOS_NUM_SAMPLES = 100  # Number of samples for uncertainty estimation
CHRONOS_TEMPERATURE = 1.0  # Sampling temperature (1.0 = default, lower = more conservative)
CHRONOS_TOP_K = 50  # Top-k sampling (50 = default)
CHRONOS_TOP_P = 1.0  # Nucleus sampling threshold (1.0 = default, no filtering)
CHRONOS_MODEL = "amazon/chronos-bolt-base"  # Model to use (base/small/mini/tiny/large)

# AutoGluon settings
AUTOGLUON_TIME_LIMIT = 120  # Time limit in seconds for model fitting
AUTOGLUON_RANDOM_SEED = 42  # Random seed for reproducibility
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
    
    # Filter data to start NOW time - last 300 samples
    mask = full_day_data.index <= est_start_time
    initial_data_all = full_day_data[mask]
    # Take last N samples based on user configuration
    initial_data = initial_data_all.tail(HISTORICAL_SAMPLES)
    
    # Plot only the historical data up to NOW
    if len(initial_data) > 0:
        prices = initial_data['close_norm'].values
        timestamps = initial_data.index
        
        ax1.plot(timestamps, prices, 'b-', linewidth=1.5, label='Historical Price', alpha=0.9)
    else:
        timestamps = []  # Initialize empty if no data
    
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
    ax1.set_ylabel('Normalized Price', fontsize=12)
    ax1.set_title(f'Chronos-Bolt Real-Time Predictions - {TARGET_DATE}', fontsize=16, fontweight='bold')
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
    
    # First, extend the historical price line to current NOW time - last 300 samples
    start_time = prediction_entry['now_time_est']
    est_tz = pytz.timezone('US/Eastern')
    mask = full_day_data.index <= start_time
    historical_all = full_day_data[mask]
    # Take last N samples based on user configuration
    historical_to_now = historical_all.tail(HISTORICAL_SAMPLES)
    
    if len(historical_to_now) > 0:
        # Clear and redraw historical price to current point
        ax1.clear()
        ax1.plot(historical_to_now.index, historical_to_now['close_norm'].values, 
                'b-', linewidth=1.5, label='Historical Price', alpha=0.9)
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
            color=color, alpha=0.8, linewidth=2.0,
            label=f"{prediction_entry['now_time'].strftime('%I:%M%p')} - {prediction_entry['direction']}")
    
    # Mark prediction start point
    ax1.scatter(start_time, prediction_entry['last_price'], 
               color=color, s=80, zorder=5, edgecolor='black', linewidth=1)
    
    # Add current time marker
    ax1.axvline(x=start_time, color='red', linestyle='--', alpha=0.4, linewidth=2)
    
    # Restore plot formatting
    ax1.set_ylabel('Normalized Price', fontsize=12)
    ax1.set_title(f'Chronos-Bolt Real-Time Predictions - {TARGET_DATE}', fontsize=16, fontweight='bold')
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
        pred_end_time = pred_time + timedelta(minutes=PREDICTION_HORIZON_MINUTES)
        
        # Can we evaluate this prediction? Only if NOW has reached the end of its horizon
        can_evaluate = current_now_time >= pred_end_time
        
        # Always draw the prediction bar
        ax2.barh(1, bar_width.total_seconds()/86400, 
                left=pred_time, height=bar_height, 
                color=dir_colors[pred['direction']], 
                alpha=0.7, edgecolor='black', linewidth=0.5)
        
        if can_evaluate:
            # Check if we've already evaluated this prediction
            if 'evaluated' not in pred:
                # Get actual data for this prediction
                mask = (full_day_data.index > pred_time) & (full_day_data.index <= pred_end_time)
                actual_data = full_day_data[mask]
                
                if len(actual_data) > 0:
                    # Calculate actual move
                    start_price = pred['last_price']
                    end_price = actual_data['close_norm'].iloc[-1]
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
                    print(f"\n  Evaluation complete for {pred['now_time'].strftime('%I:%M %p')} prediction:")
                    print(f"    Predicted: {pred['direction']}, Actual: {actual_sign} (move: {actual_move_pct:.3f}%)")
                    print(f"    Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            
            # If evaluated (either just now or previously), draw the actual and correctness bars
            if 'evaluated' in pred and pred['evaluated']:
                actual_sign = pred['actual_sign']
                is_correct = pred['is_correct']
                
                # Store result for current prediction if this is it
                if pred == prediction_entry:
                    directionally_correct = is_correct
                    # Store for console output
                    current_actual_sign = actual_sign
                    current_actual_move = pred['actual_move_pct']
                
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
    
    # Don't return anything - evaluation is printed when it happens

def run_prediction_loop():
    """Run predictions in a loop, advancing time and collecting results"""
    
    # Initialize
    config = get_config()
    data_processor = DataProcessor(config.DATA_PATH)
    
    print(f"\nRunning Chronos prediction loop")
    print(f"Date: {TARGET_DATE}")
    print(f"Time range: {START_TIME} to {END_TIME} PT")
    print(f"Step: {TIME_STEP_MINUTES} minutes")
    print(f"Horizon: {PREDICTION_HORIZON_MINUTES} minutes at {PREDICTION_FREQUENCY} resolution")
    print("="*60)
    
    # Load full day data once
    full_day_data = data_processor.get_processed_trading_day(
        config.DEFAULT_DATA_FILE, 
        TARGET_DATE, 
        include_indicators=True
    )
    
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
    
    # Initialize lists to store results
    all_predictions = []
    all_results = []
    actual_prices = []
    
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
        
        # Slice data up to current time
        historical_data = data_processor.slice_data_to_current(full_day_data, current_datetime_str)
        
        if len(historical_data) < 100:
            print(f"Skipping - insufficient data (only {len(historical_data)} points)")
            current_time += timedelta(minutes=TIME_STEP_MINUTES)
            continue
        
        # Prepare data for prediction
        timestamps = historical_data.index
        
        if USE_RAW_PRICES:
            # Use raw prices for Chronos (might work better as it was trained on raw data)
            normalized_data = {
                'open': historical_data['open'].values,
                'high': historical_data['high'].values,
                'low': historical_data['low'].values,
                'close': historical_data['close'].values,
                'volume': historical_data['volume'].values
            }
            current_price = historical_data['close'].iloc[-1]
        else:
            # Use normalized prices (around 1.0)
            normalized_data = {
                'open': historical_data['open_norm'].values,
                'high': historical_data['high_norm'].values,
                'low': historical_data['low_norm'].values,
                'close': historical_data['close_norm'].values,
                'volume': historical_data['volume'].values
            }
            current_price = historical_data['close_norm'].iloc[-1]
        
        # Debug: Print data statistics
        if loop_count == 1:  # Only print on first loop
            data_type = "Raw" if USE_RAW_PRICES else "Normalized"
            print(f"\n{data_type} data check:")
            print(f"Close prices - Min: {normalized_data['close'].min():.4f}, Max: {normalized_data['close'].max():.4f}, Mean: {normalized_data['close'].mean():.4f}")
            print(f"Last 5 close prices: {normalized_data['close'][-5:]}")
        
        # Create predictor with configured hyperparameters
        predictor = ChronosPredictor(
            prediction_length=PREDICTION_HORIZON_MINUTES,
            model_name=CHRONOS_MODEL,
            frequency=PREDICTION_FREQUENCY,
            num_samples=CHRONOS_NUM_SAMPLES,
            temperature=CHRONOS_TEMPERATURE,
            top_k=CHRONOS_TOP_K,
            top_p=CHRONOS_TOP_P
        )
        
        try:
            # Generate prediction
            result = predictor.predict_directional(
                normalized_data, 
                timestamps, 
                current_price=current_price
            )
            
            # Generate time series forecast
            train_data = predictor.prepare_data_for_chronos(normalized_data, timestamps)
            forecast = predictor.generate_time_series_forecast(train_data, num_samples=100)
            
            # Store results
            last_normalized_price = historical_data['close_norm'].iloc[-1]
            
            if USE_RAW_PRICES:
                # Get the normalization base (previous day's close)
                # This is what the data was normalized by
                first_norm_price = historical_data['close_norm'].iloc[0]
                first_raw_price = historical_data['close'].iloc[0]
                normalization_base = first_raw_price / first_norm_price
                
                # Normalize the raw predictions to the same scale
                normalized_predictions = forecast['mean'][:PREDICTION_HORIZON_MINUTES] / normalization_base
            else:
                normalized_predictions = forecast['mean'][:PREDICTION_HORIZON_MINUTES]
            
            # Ensure prediction starts exactly at current price
            raw_predictions = normalized_predictions
            # Adjust predictions to start from current price
            if len(raw_predictions) > 0:
                # Calculate the offset and adjust all predictions
                prediction_offset = last_normalized_price - raw_predictions[0]
                adjusted_predictions = raw_predictions + prediction_offset
            else:
                adjusted_predictions = raw_predictions
            
            prediction_entry = {
                'loop': loop_count,
                'now_time': current_time,
                'now_time_est': est_datetime,
                'last_price': last_normalized_price,
                'prediction': adjusted_predictions,  # Use adjusted predictions
                'direction': result['direction'],
                'confidence': result['confidence'],
                'expected_move_pct': result['expected_move_pct']
            }
            
            all_predictions.append(prediction_entry)
            
            if USE_RAW_PRICES:
                print(f"Last raw price: ${current_price:.2f} (normalized: {prediction_entry['last_price']:.4f})")
                if loop_count == 1:  # Debug on first loop
                    print(f"Normalization base: ${normalization_base:.2f}")
            else:
                print(f"Last normalized price: {prediction_entry['last_price']:.4f}")
            print(f"Prediction: {result['direction']} ({result['expected_move_pct']:.2f}%)")
            print(f"Confidence: {result['confidence']:.1%}")
            
            # Update plot in real-time
            update_plot(fig, ax1, ax2, prediction_entry, full_day_data, loop_count-1, all_predictions)
            
            # Add animation delay
            time.sleep(ANIMATION_DELAY)
            
        except Exception as e:
            print(f"Error generating prediction: {e}")
        
        # Move to next time step
        current_time += timedelta(minutes=TIME_STEP_MINUTES)
    
    print(f"\nCompleted {loop_count} prediction loops")
    
    # Count how many predictions were evaluated
    evaluated_count = sum(1 for pred in all_predictions if 'evaluated' in pred and pred['evaluated'])
    print(f"Total predictions made: {len(all_predictions)}")
    print(f"Predictions evaluated: {evaluated_count}")
    if evaluated_count < len(all_predictions):
        print(f"Note: {len(all_predictions) - evaluated_count} predictions not yet evaluated (prediction horizon not reached)")
    
    # Save the final plot
    plt.savefig('chronos_loop_predictions_realtime.png', dpi=150, bbox_inches='tight')
    print("Real-time plot saved as 'chronos_loop_predictions_realtime.png'")
    
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
    prices = full_day_data_filtered['close_norm'].values
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
    ax1.set_ylabel('Normalized Price', fontsize=12)
    ax1.set_title(f'Chronos-Bolt Stacked Predictions - {TARGET_DATE} (Market Hours Only)', fontsize=16, fontweight='bold')
    
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
    plt.savefig('chronos_loop_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nStacked predictions plot saved as 'chronos_loop_predictions.png'")

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
        print("PREDICTION ACCURACY ANALYSIS")
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
        
        # Don't save CSV (per user request)
        # df_results.to_csv('chronos_loop_results.csv', index=False)
        # print("\nDetailed results saved to 'chronos_loop_results.csv'")
        
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
        plt.title('Prediction Accuracy Over Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 110)
        
        # Add time labels
        step = max(1, len(df_results) // 10)
        plt.xticks(range(0, len(df_results), step), 
                  [df_results.iloc[i]['time'] for i in range(0, len(df_results), step)],
                  rotation=45)
        
        plt.tight_layout()
        plt.savefig('chronos_loop_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Accuracy plot saved as 'chronos_loop_accuracy.png'")

if __name__ == "__main__":
    # Suppress AutoGluon verbosity for cleaner output
    import os
    os.environ['AUTOGLUON_VERBOSITY'] = '1'  # Reduce verbosity
    
    # Run the prediction loop
    predictions = run_prediction_loop()
