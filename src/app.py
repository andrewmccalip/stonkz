"""Flask web application for AI Stock Prediction System"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import pandas as pd
import json
import logging

from src.data_processor import DataProcessor
from src.indicators import add_all_indicators
from config import get_config

# Initialize Flask app
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

# Load configuration
config = get_config()
app.config.from_object(config)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize data processor
data_processor = DataProcessor(config.DATA_PATH)

# Cache for loaded data
data_cache = {}


@app.route('/')
def index():
    """Main page with date selector and visualization"""
    return render_template('index.html')


@app.route('/api/dates')
def get_available_dates():
    """Get list of available dates from the data files"""
    try:
        # Load the data file if not in cache
        if 'dates' not in data_cache:
            df = data_processor.load_csv_data(config.DEFAULT_DATA_FILE)
            dates = data_processor.get_available_dates(df)
            data_cache['dates'] = dates
        
        return jsonify({
            'success': True,
            'dates': data_cache['dates']
        })
    
    except Exception as e:
        logger.error(f"Error getting dates: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/data/<date>')
def get_historical_data(date):
    """Get historical data for a specific date"""
    try:
        # Get optional parameters
        current_time = request.args.get('current_time', None)
        
        # Use cached processing - get full trading day
        full_day_data = data_processor.get_processed_trading_day(
            config.DEFAULT_DATA_FILE, 
            date, 
            include_indicators=True
        )
        
        # Get base price for response
        df_full = data_processor.load_csv_data(config.DEFAULT_DATA_FILE)
        prev_close = data_processor.get_previous_close(df_full, date)
        
        # Split data into historical and ground truth if current_time is specified
        if current_time:
            historical_data = data_processor.slice_data_to_current(full_day_data, current_time)
            
            # Get ground truth data (after current_time)
            if ':' in current_time and len(current_time) <= 5:
                # Time only provided, use the date from the data
                from datetime import datetime
                import pytz
                eastern_tz = pytz.timezone('US/Eastern')
                # Use the date from the last timestamp in historical data
                data_date = historical_data.index[-1].date()
                hour, minute = map(int, current_time.split(':'))
                current_dt = eastern_tz.localize(datetime.combine(data_date, datetime.min.time()).replace(hour=hour, minute=minute))
            else:
                current_dt = pd.to_datetime(current_time)
                if current_dt.tzinfo is None:
                    import pytz
                    eastern_tz = pytz.timezone('US/Eastern')
                    current_dt = eastern_tz.localize(current_dt)
            
            ground_truth_data = full_day_data[full_day_data.index > current_dt]
        else:
            historical_data = full_day_data
            ground_truth_data = pd.DataFrame()  # Empty
        
        # Prepare response data for historical
        response_data = {
            'timestamps': historical_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'ohlc': {
                'open': historical_data['open_norm'].tolist(),
                'high': historical_data['high_norm'].tolist(),
                'low': historical_data['low_norm'].tolist(),
                'close': historical_data['close_norm'].tolist()
            },
            'volume': historical_data['volume'].tolist(),
            'indicators': {},
            'base_price': prev_close,
            'current_price': historical_data['close'].iloc[-1] if not historical_data.empty else None,
            'current_time': current_time
        }
        
        # Add ground truth data if available
        if not ground_truth_data.empty:
            response_data['ground_truth'] = {
                'timestamps': ground_truth_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'ohlc': {
                    'open': ground_truth_data['open_norm'].tolist(),
                    'high': ground_truth_data['high_norm'].tolist(),
                    'low': ground_truth_data['low_norm'].tolist(),
                    'close': ground_truth_data['close_norm'].tolist()
                },
                'volume': ground_truth_data['volume'].tolist()
            }
        
        # Add indicators to response
        indicator_columns = ['rsi', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 
                           'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
                           'obv', 'vwap', 'stoch_k', 'stoch_d']
        
        for col in indicator_columns:
            if col in historical_data.columns:
                values = historical_data[col].fillna(0).tolist()
                response_data['indicators'][col] = values
        
        return jsonify({
            'success': True,
            'data': response_data
        })
    
    except Exception as e:
        logger.error(f"Error getting historical data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Trigger AI predictions for the current state"""
    try:
        # Get request data
        req_data = request.get_json()
        date = req_data.get('date')
        current_time = req_data.get('current_time')
        
        if not date:
            return jsonify({
                'success': False,
                'error': 'Date parameter is required'
            }), 400
        
        # Use cached processing
        with_indicators = data_processor.get_processed_trading_day(
            config.DEFAULT_DATA_FILE, 
            date, 
            include_indicators=True
        )
        
        # Get base price for metadata
        df_full = data_processor.load_csv_data(config.DEFAULT_DATA_FILE)
        prev_close = data_processor.get_previous_close(df_full, date)
        
        # Slice to current time
        if current_time:
            print(f"\nDEBUG: Slicing data to current_time = {current_time}")
            print(f"DEBUG: Data before slicing - Start: {with_indicators.index[0]}, End: {with_indicators.index[-1]}, Count: {len(with_indicators)}")
            
            # Need to build the correct datetime for slicing
            # The date context should be from the target date
            from datetime import datetime
            target_date_obj = datetime.strptime(date, '%Y-%m-%d')
            hour, minute = map(int, current_time.split(':'))
            current_datetime = datetime.combine(target_date_obj.date(), datetime.min.time()).replace(hour=hour, minute=minute)
            current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"DEBUG: Slicing to datetime: {current_datetime_str}")
            with_indicators = data_processor.slice_data_to_current(with_indicators, current_datetime_str)
            print(f"DEBUG: Data after slicing - Start: {with_indicators.index[0]}, End: {with_indicators.index[-1]}, Count: {len(with_indicators)}\n")
        
        # Prepare data for prediction - use all data up to current time
        prediction_data = data_processor.prepare_data_for_prediction(with_indicators, lookback_periods=None)
        
        # Print normalized open prices to console
        from datetime import datetime
        import pytz
        
        # Convert timestamps to EST AM/PM format
        est = pytz.timezone('US/Eastern')
        start_time = datetime.strptime(prediction_data['timestamps'][0], '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(prediction_data['timestamps'][-1], '%Y-%m-%d %H:%M:%S')
        
        # Format as AM/PM
        start_formatted = start_time.strftime('%Y-%m-%d %I:%M:%S %p')
        end_formatted = end_time.strftime('%Y-%m-%d %I:%M:%S %p')
        
        print("\n" + "="*80)
        print("NORMALIZED OPEN PRICES")
        print("="*80)
        print(f"Data from: {start_formatted} EST (previous day market close)")
        print(f"Data to:   {end_formatted} EST (NOW - your selected time)")
        print(f"Current time selection: {current_time if current_time else 'Not specified'}")
        print(f"Total data points: {len(prediction_data['prices']['open'])}")
        print("-"*80)
        open_prices = [f"{p:.6f}" for p in prediction_data['prices']['open']]
        print(open_prices)
        print("="*80 + "\n")
        
        # For now, return mock predictions
        # TODO: Integrate with AI predictor module
        mock_predictions = generate_mock_predictions(prediction_data)
        
        return jsonify({
            'success': True,
            'predictions': mock_predictions,
            'metadata': {
                'prediction_time': prediction_data['current_time'],
                'market_close': prediction_data['market_close'],
                'base_price': prev_close,
                'normalized_base': 1.00
            }
        })
    
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def generate_mock_predictions(data):
    """Generate mock directional predictions with confidence intervals"""
    import numpy as np
    
    current_price = data['current_price']
    current_time = pd.to_datetime(data['current_time'])
    market_close = pd.to_datetime(data['market_close'])
    
    # Generate predictions for all 15-minute intervals from market open to current time
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # If current time is before market open, no predictions
    if current_time < market_open:
        return {
            'intervals': [],
            'aggregate': [],
            'metadata': {
                'generated_at': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'market_close': market_close.strftime('%Y-%m-%d %H:%M:%S'),
                'intervals_count': 0,
                'message': 'Current time is before market open'
            }
        }
    
    time_intervals = []
    
    # Start from 15 minutes after market open
    interval_time = market_open + timedelta(minutes=15)
    
    # Generate all 15-minute intervals up to current time
    while interval_time <= current_time and interval_time <= market_close:
        time_intervals.append(interval_time)
        interval_time += timedelta(minutes=15)
    
    # If no intervals generated (current time too close to market open)
    if not time_intervals:
        return {
            'intervals': [],
            'aggregate': [],
            'metadata': {
                'generated_at': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'market_close': market_close.strftime('%Y-%m-%d %H:%M:%S'),
                'intervals_count': 0,
                'message': 'Not enough time has passed since market open for predictions'
            }
        }
    
    models = ['GPT-4', 'Claude 3', 'Gemini Pro', 'Mistral Large', 'Command R+']
    
    # Model characteristics (bias and consistency)
    model_profiles = {
        'GPT-4': {'up_bias': 0.55, 'confidence_mean': 0.75, 'confidence_std': 0.1},
        'Claude 3': {'up_bias': 0.35, 'confidence_mean': 0.72, 'confidence_std': 0.12},
        'Gemini Pro': {'up_bias': 0.65, 'confidence_mean': 0.78, 'confidence_std': 0.08},
        'Mistral Large': {'up_bias': 0.50, 'confidence_mean': 0.70, 'confidence_std': 0.15},
        'Command R+': {'up_bias': 0.40, 'confidence_mean': 0.68, 'confidence_std': 0.13}
    }
    
    predictions = []
    
    for time_point in time_intervals:
        interval_predictions = {
            'time': time_point.strftime('%I:%M %p'),
            'timestamp': time_point.strftime('%Y-%m-%d %H:%M:%S'),
            'minutes_ahead': int((time_point - current_time).total_seconds() / 60),
            'predictions': []
        }
        
        # Generate predictions for each model
        for model in models:
            profile = model_profiles[model]
            np.random.seed(hash(f"{model}{time_point}") % 2**32)
            
            # Generate direction probability with some randomness
            up_prob = np.clip(np.random.normal(profile['up_bias'], 0.15), 0, 1)
            
            # Determine prediction
            rand = np.random.random()
            if rand < up_prob:
                direction = 'UP'
                # Expected move: 0.1% to 2% for up
                expected_move = np.random.uniform(0.1, 2.0)
            elif rand < up_prob + 0.2:  # 20% chance of flat
                direction = 'FLAT'
                expected_move = np.random.uniform(-0.1, 0.1)
            else:
                direction = 'DOWN'
                # Expected move: -0.1% to -2% for down
                expected_move = np.random.uniform(-2.0, -0.1)
            
            # Generate confidence level
            confidence = np.clip(
                np.random.normal(profile['confidence_mean'], profile['confidence_std']),
                0.4, 0.95
            )
            
            # Calculate standard deviation for the prediction
            # Higher confidence = lower std dev
            std_dev = (1 - confidence) * abs(expected_move) * 0.5
            
            interval_predictions['predictions'].append({
                'model': model,
                'direction': direction,
                'confidence': round(confidence * 100, 1),
                'expected_move': round(expected_move, 2),
                'std_dev': round(std_dev, 3),
                'upper_bound': round(expected_move + 2 * std_dev, 2),
                'lower_bound': round(expected_move - 2 * std_dev, 2)
            })
        
        predictions.append(interval_predictions)
    
    # Calculate aggregate statistics
    aggregate_stats = calculate_aggregate_predictions(predictions)
    
    return {
        'intervals': predictions,
        'aggregate': aggregate_stats,
        'metadata': {
            'generated_at': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'market_close': market_close.strftime('%Y-%m-%d %H:%M:%S'),
            'intervals_count': len(predictions)
        }
    }


def calculate_aggregate_predictions(predictions):
    """Calculate aggregate statistics from all model predictions"""
    if not predictions:
        return {}
    
    aggregate = []
    
    for interval in predictions:
        up_count = sum(1 for p in interval['predictions'] if p['direction'] == 'UP')
        down_count = sum(1 for p in interval['predictions'] if p['direction'] == 'DOWN')
        flat_count = sum(1 for p in interval['predictions'] if p['direction'] == 'FLAT')
        
        # Calculate weighted average expected move
        total_weight = sum(p['confidence'] for p in interval['predictions'])
        weighted_move = sum(p['expected_move'] * p['confidence'] for p in interval['predictions']) / total_weight if total_weight > 0 else 0
        
        # Calculate consensus
        total = len(interval['predictions'])
        if up_count > total / 2:
            consensus = 'BULLISH'
            consensus_strength = (up_count / total) * 100
        elif down_count > total / 2:
            consensus = 'BEARISH'
            consensus_strength = (down_count / total) * 100
        else:
            consensus = 'NEUTRAL'
            consensus_strength = (flat_count / total) * 100 if flat_count > 0 else 50
        
        aggregate.append({
            'time': interval['time'],
            'consensus': consensus,
            'consensus_strength': round(consensus_strength, 1),
            'up_votes': up_count,
            'down_votes': down_count,
            'flat_votes': flat_count,
            'expected_move': round(weighted_move, 2),
            'probability_up': round((up_count / total) * 100, 1),
            'probability_down': round((down_count / total) * 100, 1),
            'probability_flat': round((flat_count / total) * 100, 1)
        })
    
    return aggregate


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
