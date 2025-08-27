#!/usr/bin/env python3
"""
FlaMinGo Stock Demo - Real-time stock predictions using FlaMinGo TimesFM
Uses the same real daily stock data as chronos_loop.py for direct comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import sys
import pytz
from pathlib import Path

warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from data_processor import DataProcessor
from config import get_config

# Import our FlaMinGo predictor
from flamingo_hf_predictor import FlaMinGoHuggingFacePredictor

# ==============================================================================
# USER CONFIGURATION - EDIT THESE VALUES
# ==============================================================================
TARGET_DATE = '2025-05-19'  # Set the date to analyze (same as chronos_loop.py)
CURRENT_TIME = '10:30'      # Current time in PT (market time)
USER_TIMEZONE = 'US/Pacific'  # Your local timezone

# Prediction settings - predict to end of trading day
PREDICTION_FREQUENCY = '1min'    # Resolution of predictions
HISTORICAL_SAMPLES = 200         # Number of historical samples to use

# FlaMinGo model settings
MODEL_SIZE = '200m'              # '200m' or '500m'
USE_NORMALIZED_PRICES = True     # Use normalized prices (recommended)
# ==============================================================================

def load_real_stock_data(target_date, current_time_str):
    """Load real stock data using the same method as chronos_loop.py"""
    
    print(f"📊 Loading real stock data for {target_date}...")
    
    # Initialize data processor
    config = get_config()
    data_processor = DataProcessor(config.DATA_PATH)
    
    # Load full day data
    full_day_data = data_processor.get_processed_trading_day(
        config.DEFAULT_DATA_FILE, 
        target_date, 
        include_indicators=True
    )
    
    # Convert PT to EST for data slicing (same logic as chronos_loop.py)
    pst_tz = pytz.timezone(USER_TIMEZONE)
    est_tz = pytz.timezone('US/Eastern')
    
    # Parse current time
    current_hour, current_minute = map(int, current_time_str.split(':'))
    target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
    pst_datetime = pst_tz.localize(
        datetime.combine(target_date_obj.date(), datetime.min.time()).replace(
            hour=current_hour, minute=current_minute
        )
    )
    est_datetime = pst_datetime.astimezone(est_tz)
    current_datetime_str = est_datetime.strftime('%Y-%m-%d %H:%M:%S')
    
    # Calculate market close time (1 PM PT = 4 PM EST)
    market_close_pst = pst_tz.localize(
        datetime.combine(target_date_obj.date(), datetime.min.time()).replace(hour=13, minute=0)
    )
    market_close_est = market_close_pst.astimezone(est_tz)
    
    # Calculate minutes until market close
    minutes_to_close = int((market_close_est - est_datetime).total_seconds() / 60)
    
    # Slice data up to current time
    historical_data = data_processor.slice_data_to_current(full_day_data, current_datetime_str)
    
    print(f"   Loaded {len(historical_data)} data points")
    print(f"   Time range: {historical_data.index[0]} to {historical_data.index[-1]} (EST)")
    print(f"   Current time: {current_time_str} PT = {est_datetime.strftime('%H:%M')} EST")
    print(f"   Market close: 1:00 PM PT = {market_close_est.strftime('%H:%M')} EST")
    print(f"   Minutes until close: {minutes_to_close}")
    
    return historical_data, full_day_data, est_datetime, minutes_to_close

def prepare_data_for_flamingo(historical_data, use_normalized=True):
    """Prepare data for FlaMinGo prediction"""
    
    # Take last N samples for context
    context_data = historical_data.tail(HISTORICAL_SAMPLES)
    
    if use_normalized:
        # Use normalized prices (around 1.0)
        price_series = context_data['close_norm']
        data_type = "Normalized"
    else:
        # Use raw prices
        price_series = context_data['close']
        data_type = "Raw"
    
    print(f"\n🔧 Prepared {data_type} data for FlaMinGo:")
    print(f"   Context length: {len(context_data)} samples")
    print(f"   Price range: {price_series.min():.4f} to {price_series.max():.4f}")
    print(f"   Current price: {price_series.iloc[-1]:.4f}")
    
    return price_series, context_data.index

def plot_flamingo_predictions(historical_data, prediction_result, current_time_est, target_date):
    """Plot historical data and FlaMinGo predictions"""
    
    fig, ax1 = plt.subplots(1, 1, figsize=(18, 10))
    
    # Prepare data for plotting
    hist_timestamps = historical_data.index
    hist_prices = historical_data['close_norm'] if USE_NORMALIZED_PRICES else historical_data['close']
    
    # Plot 1: Full context + predictions
    ax1.plot(hist_timestamps, hist_prices, 
             label='Historical Prices', color='blue', alpha=0.8, linewidth=1.5)
    
    # Plot predictions
    pred_timestamps = prediction_result['timestamps']
    predictions = prediction_result['predictions']
    
    ax1.plot(pred_timestamps, predictions, 
             label='FlaMinGo Predictions', color='red', linewidth=2.5)
    
    # Add current time marker
    ax1.axvline(x=current_time_est, color='green', linestyle='--', alpha=0.7, linewidth=2, 
                label=f'Current Time ({CURRENT_TIME} PT)')
    
    # Add market open line
    market_open_et = current_time_est.replace(hour=9, minute=30, second=0, microsecond=0)
    if market_open_et >= hist_timestamps[0] and market_open_et <= hist_timestamps[-1]:
        ax1.axvline(x=market_open_et, color='orange', linestyle=':', alpha=0.5, 
                    label='Market Open (6:30 AM PT)')
    
    # Add reference line for normalized prices
    if USE_NORMALIZED_PRICES:
        ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Previous EOD (1.00)')
    
    # Format plot
    price_type = "Normalized Price" if USE_NORMALIZED_PRICES else "Price ($)"
    prediction_minutes = len(predictions)
    ax1.set_title(f'FlaMinGo TimesFM Stock Predictions to Market Close - {target_date}', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Time (PT)', fontsize=12)
    ax1.set_ylabel(price_type, fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis to show PT time
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
    
    # Add market close vertical line
    market_close_pst = current_time_est.replace(hour=13, minute=0, second=0, microsecond=0)
    market_close_pst = pytz.timezone('US/Pacific').localize(market_close_pst.replace(tzinfo=None))
    market_close_est = market_close_pst.astimezone(pytz.timezone('US/Eastern'))
    ax1.axvline(x=market_close_est, color='purple', linestyle='-', alpha=0.7, linewidth=2, 
                label='Market Close (1:00 PM PT)')
    
    # Rotate x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def analyze_prediction_direction(prediction_result, current_price):
    """Analyze the directional prediction from FlaMinGo"""
    
    predictions = prediction_result['predictions']
    
    if len(predictions) == 0:
        return "UNKNOWN", 0.0, 0.0
    
    # Calculate expected move
    final_price = predictions[-1]
    price_change = final_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # Determine direction (using same threshold as chronos_loop.py)
    FLAT_THRESHOLD_PCT = 0.01
    
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

def run_flamingo_stock_demo():
    """Main demo function"""
    
    print("=" * 70)
    print("FlaMinGo TimesFM Real Stock Data Demo")
    print("=" * 70)
    print(f"Target Date: {TARGET_DATE}")
    print(f"Current Time: {CURRENT_TIME} PT")
    print(f"Model Size: {MODEL_SIZE}")
    print(f"Using {'Normalized' if USE_NORMALIZED_PRICES else 'Raw'} Prices")
    
    try:
        # 1. Load real stock data
        historical_data, full_day_data, current_time_est, minutes_to_close = load_real_stock_data(TARGET_DATE, CURRENT_TIME)
        
        # Use minutes to close as prediction horizon (but cap at reasonable limit)
        prediction_horizon = min(minutes_to_close, 150)  # Cap at 2.5 hours for model stability
        print(f"Prediction Horizon: {prediction_horizon} minutes (until market close)")
        
        if prediction_horizon <= 0:
            print("❌ Market is already closed or about to close!")
            return
        
        if len(historical_data) < 100:
            print(f"❌ Insufficient data: only {len(historical_data)} points available")
            return
        
        # 2. Prepare data for FlaMinGo
        price_series, timestamps = prepare_data_for_flamingo(historical_data, USE_NORMALIZED_PRICES)
        current_price = price_series.iloc[-1]
        
        # 3. Initialize FlaMinGo predictor
        print(f"\n🚀 Initializing FlaMinGo TimesFM {MODEL_SIZE} model...")
        predictor = FlaMinGoHuggingFacePredictor(
            prediction_length=prediction_horizon,
            frequency=PREDICTION_FREQUENCY,
            model_size=MODEL_SIZE,
            use_hf_models=True
        )
        
        # Display model info
        model_info = predictor.get_model_info()
        print("\n📊 Model Information:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # 4. Generate predictions
        print(f"\n🔮 Generating {prediction_horizon}-minute predictions...")
        prediction_result = predictor.predict_forecast(price_series, timestamps)
        
        # 5. Analyze results
        print("\n📈 Prediction Results:")
        print(f"   Model: {prediction_result['model_info']['model_name']}")
        print(f"   Backend: {prediction_result['model_info']['backend']}")
        print(f"   Context length: {prediction_result['model_info']['context_length']}")
        print(f"   Predictions generated: {len(prediction_result['predictions'])}")
        print(f"   Prediction range: {prediction_result['predictions'].min():.4f} to {prediction_result['predictions'].max():.4f}")
        
        # Directional analysis
        direction, expected_move_pct, confidence = analyze_prediction_direction(prediction_result, current_price)
        print(f"\n🎯 Directional Analysis:")
        print(f"   Current price: {current_price:.4f}")
        print(f"   Predicted direction: {direction}")
        print(f"   Expected move: {expected_move_pct:+.3f}%")
        print(f"   Confidence: {confidence:.1%}")
        
        # Show first few predictions
        print(f"\n   First 5 predictions:")
        for i in range(min(5, len(prediction_result['predictions']))):
            pred_time = prediction_result['timestamps'][i]
            pred_value = prediction_result['predictions'][i]
            if isinstance(pred_time, str):
                print(f"     {i+1}. {pred_time}: {pred_value:.4f}")
            else:
                print(f"     {i+1}. {pred_time.strftime('%H:%M:%S')}: {pred_value:.4f}")
        
        # 6. Create visualization
        print("\n📊 Creating visualization...")
        fig = plot_flamingo_predictions(historical_data, prediction_result, current_time_est, TARGET_DATE)
        
        # Save plot
        plot_filename = f"flamingo_stock_predictions_{TARGET_DATE}_{CURRENT_TIME.replace(':', '')}.png"
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"   📊 Plot saved as: {plot_filename}")
        
        # Show plot
        plt.show()
        
        print("\n✅ FlaMinGo stock demo completed successfully!")
        print("\nKey Insights:")
        print(f"- FlaMinGo predicts {direction} movement of {expected_move_pct:+.3f}% until market close")
        print(f"- Model confidence: {confidence:.1%}")
        print(f"- Prediction horizon: {prediction_horizon} minutes until 1:00 PM PT")
        print(f"- Using real market data from {TARGET_DATE}")
        
        # Show end-of-day prediction
        if len(prediction_result['predictions']) > 0:
            final_price = prediction_result['predictions'][-1]
            total_move_pct = ((final_price - current_price) / current_price) * 100
            print(f"- Predicted closing price: {final_price:.4f} ({total_move_pct:+.3f}% from now)")
        
        print("\nNext steps:")
        print("- Try different times of day to see how predictions change")
        print("- Compare with Chronos predictions using chronos_loop.py")
        print("- Test on different dates to validate model performance")
        print("- Use for end-of-day trading decisions")
        
        return prediction_result
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("1. Ensure you have the required data files in databento/")
        print("2. Check that the target date has available data")
        print("3. Verify FlaMinGo model can be downloaded (internet connection)")
        print("4. Try a different current time if data is insufficient")
        print("5. Reduce HISTORICAL_SAMPLES if memory issues occur")

def compare_with_chronos():
    """Optional: Compare FlaMinGo with Chronos predictions"""
    
    print("\n" + "=" * 70)
    print("Model Comparison: FlaMinGo vs Chronos")
    print("=" * 70)
    
    try:
        # Import Chronos predictor
        from chronos import ChronosPredictor
        
        # Load the same data
        historical_data, _, current_time_est, minutes_to_close = load_real_stock_data(TARGET_DATE, CURRENT_TIME)
        price_series, timestamps = prepare_data_for_flamingo(historical_data, USE_NORMALIZED_PRICES)
        prediction_horizon = min(minutes_to_close, 150)
        
        # Test both models
        models_to_test = []
        
        # FlaMinGo
        print("\n🔮 Testing FlaMinGo...")
        flamingo_predictor = FlaMinGoHuggingFacePredictor(
            prediction_length=prediction_horizon,
            frequency=PREDICTION_FREQUENCY,
            model_size=MODEL_SIZE
        )
        flamingo_result = flamingo_predictor.predict_forecast(price_series, timestamps)
        models_to_test.append(('FlaMinGo', flamingo_result))
        
        # Chronos
        print("\n🔮 Testing Chronos...")
        chronos_predictor = ChronosPredictor(
            prediction_length=prediction_horizon,
            frequency=PREDICTION_FREQUENCY,
            model_name="amazon/chronos-bolt-base"
        )
        
        # Prepare data for Chronos (it expects OHLCV format)
        chronos_data = {
            'close': price_series.values,
            'open': price_series.values,  # Simplified
            'high': price_series.values,
            'low': price_series.values,
            'volume': np.ones(len(price_series))  # Dummy volume
        }
        
        chronos_result = chronos_predictor.predict_directional(
            chronos_data, timestamps, current_price=price_series.iloc[-1]
        )
        models_to_test.append(('Chronos', chronos_result))
        
        # Compare results
        print(f"\n📊 Model Comparison Results:")
        for model_name, result in models_to_test:
            if model_name == 'FlaMinGo':
                direction, move_pct, confidence = analyze_prediction_direction(result, price_series.iloc[-1])
                print(f"   {model_name}:")
                print(f"     Direction: {direction}")
                print(f"     Expected move: {move_pct:+.3f}%")
                print(f"     Confidence: {confidence:.1%}")
                print(f"     Predictions: {len(result['predictions'])}")
            else:  # Chronos
                print(f"   {model_name}:")
                print(f"     Direction: {result['direction']}")
                print(f"     Expected move: {result['expected_move_pct']:+.3f}%")
                print(f"     Confidence: {result['confidence']:.1%}")
        
    except ImportError:
        print("❌ Chronos not available for comparison")
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

if __name__ == "__main__":
    # Run the main demo
    result = run_flamingo_stock_demo()
    
    # Optionally run model comparison
    if result:
        user_input = input("\nWould you like to compare with Chronos? (y/n): ")
        if user_input.lower().startswith('y'):
            compare_with_chronos()
    
    print("\nDemo finished. Thank you for trying FlaMinGo with real stock data!")
