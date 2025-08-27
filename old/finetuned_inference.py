#!/usr/bin/env python3
"""
Finetuned FlaMinGo TimesFM Inference Script
Uses a finetuned FlaMinGo model to make predictions on stock market data
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json

warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

# Import project modules
from data_processor import DataProcessor
from config import get_config
import pytz

# Import FlaMinGo modules
try:
    from timesfm import TimesFmHparams
    from timesfm.timesfm_torch import TimesFmTorch
    from torch_classification_timesfm.model import TimesFmWithClassifier
    FLAMINGO_AVAILABLE = True
    print("✅ FlaMinGo TimesFM modules imported successfully")
except ImportError as e:
    FLAMINGO_AVAILABLE = False
    print(f"❌ FlaMinGo TimesFM not available: {e}")

# Import the model class from finetuning script
try:
    from finetuning import StockTimesFmModel
    print("✅ Finetuned model class imported successfully")
except ImportError as e:
    print(f"❌ Could not import finetuned model class: {e}")
    print("Make sure finetuning.py is in the same directory")

class FinetunedFlaMinGoPredictor:
    """
    Predictor using a finetuned FlaMinGo TimesFM model
    """
    
    def __init__(self, model_path: str = None, device: str = "auto", use_latest_checkpoint: bool = False):
        """
        Initialize the finetuned predictor
        
        Args:
            model_path: Path to the saved finetuned model (optional if use_latest_checkpoint=True)
            device: Device to run inference on ("auto", "cpu", "cuda")
            use_latest_checkpoint: If True, automatically use the latest checkpoint from model_checkpoints/
        """
        if use_latest_checkpoint:
            self.model_path = self._find_latest_checkpoint()
        elif model_path is None:
            # Try to find latest checkpoint as fallback
            try:
                self.model_path = self._find_latest_checkpoint()
                print("🔍 No model path specified, using latest checkpoint")
            except FileNotFoundError:
                # Fallback to default model path
                self.model_path = "finetuned_flamingo_stock_model.pth"
                print("🔍 No checkpoints found, using default model path")
        else:
            self.model_path = model_path
            
        self.device = self._get_device(device)
        self.model = None
        self.model_config = None
        self.label_encoder = None
        
        # Load the model
        self._load_model()
        
    def _get_device(self, device: str) -> str:
        """Determine the device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _find_latest_checkpoint(self) -> str:
        """Find the latest checkpoint file in model_checkpoints/ directory"""
        checkpoint_dir = Path("model_checkpoints")
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError("model_checkpoints/ directory not found")
        
        # Get all checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint files found in model_checkpoints/")
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        latest_checkpoint = checkpoint_files[0]
        print(f"🎯 Found latest checkpoint: {latest_checkpoint}")
        
        # Show checkpoint info
        size_mb = latest_checkpoint.stat().st_size / 1024 / 1024
        modified = datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return str(latest_checkpoint)
    
    def _load_model(self):
        """Load the finetuned model"""
        print(f"Loading finetuned model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'label_encoder' in checkpoint:
            # Standard model format (final saved model)
            self.model_config = checkpoint['model_config']
            self.label_encoder = checkpoint['label_encoder']
            training_config = checkpoint.get('training_config', {})
            model_state_dict = checkpoint['model_state_dict']
        elif 'epoch' in checkpoint:
            # Training checkpoint format
            print(f"📦 Loading from training checkpoint (epoch {checkpoint['epoch']})")
            self.model_config = checkpoint['model_config']
            # For checkpoints, we need to create a simple label encoder
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(['DOWN', 'FLAT', 'UP'])  # Standard classes
            training_config = checkpoint['model_config']
            model_state_dict = checkpoint['model_state_dict']
            
            # Show checkpoint metadata
            if 'val_loss' in checkpoint:
                print(f"   Validation loss: {checkpoint['val_loss']:.6f}")
            if 'timestamp' in checkpoint:
                print(f"   Timestamp: {checkpoint['timestamp']}")
        else:
            raise ValueError("Unknown checkpoint format")
        
        print(f"Model configuration: {self.model_config}")
        print(f"Training configuration: {training_config}")
        
        # Create TimesFM hyperparameters (use defaults if not saved)
        timesfm_hparams = TimesFmHparams(
            context_len=self.model_config.get('context_length', 512),
            horizon_len=128,  # Standard horizon length
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,  # 20 for 200M model
            num_heads=16,
            model_dims=1280,  # 1280 for 200M model
            per_core_batch_size=16,
            backend="cpu",
            quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            point_forecast_mode="median"
        )
        
        # Create model
        self.model = StockTimesFmModel(
            timesfm_hparams=timesfm_hparams,
            num_classes=self.model_config.get('num_classes', 3),
            feature_dim=self.model_config.get('feature_dim', 1)
        )
        
        # Load state dict with strict=False to handle dynamic layers
        missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys in model: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys in checkpoint: {unexpected_keys}")
            # Handle feature_projection layer that might be created dynamically
            if any('feature_projection' in key for key in unexpected_keys):
                print("   Note: feature_projection layer will be recreated dynamically if needed")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"Classes: {self.label_encoder.classes_}")
    
    def predict(self, data: pd.DataFrame, return_forecast: bool = True, 
                return_classification: bool = True) -> Dict:
        """
        Make predictions using the finetuned model
        
        Args:
            data: DataFrame with stock data (should have same format as training data)
            return_forecast: Whether to return price forecasts
            return_classification: Whether to return directional classification
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Prepare input data
        context_length = self.model_config['context_length']
        prediction_horizon = self.model_config['prediction_horizon']
        
        if len(data) < context_length:
            raise ValueError(f"Need at least {context_length} data points, got {len(data)}")
        
        # Take the last context_length points
        context_data = data.tail(context_length)
        
        # Prepare features (optimized for TimesFM - close prices only)
        if 'close' in context_data.columns:
            features = context_data['close'].values.reshape(-1, 1)
        elif 'close_norm' in context_data.columns:
            features = context_data['close_norm'].values.reshape(-1, 1)
        else:
            raise ValueError("No suitable price column found (need 'close' or 'close_norm')")
        
        # Convert to tensor
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Create paddings and frequency tensors
        paddings = torch.zeros((1, context_length + 128), dtype=torch.float32).to(self.device)
        freq = torch.zeros((1, 1), dtype=torch.int64).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(
                input_ts=input_tensor,
                paddings=paddings,
                freq=freq,
                horizon_len=prediction_horizon,
                return_forecast=return_forecast,
                return_classification=return_classification
            )
        
        results = {}
        
        if return_forecast and 'forecast' in outputs:
            forecast = outputs['forecast'].cpu().numpy().squeeze()
            results['forecast'] = forecast
            
            # Calculate expected move
            current_price = features[-1, 0]  # Last price from first feature
            final_price = forecast[-1]
            price_change_pct = ((final_price - current_price) / current_price) * 100
            results['expected_move_pct'] = price_change_pct
        
        if return_classification and 'classification' in outputs:
            classification_logits = outputs['classification'].cpu().numpy().squeeze()
            predicted_class_idx = np.argmax(classification_logits)
            predicted_class = self.label_encoder.classes_[predicted_class_idx]
            confidence = torch.softmax(outputs['classification'], dim=1).cpu().numpy().squeeze()
            
            results['predicted_direction'] = predicted_class
            results['classification_confidence'] = confidence
            results['direction_confidence'] = confidence[predicted_class_idx]
        
        return results
    
    def predict_realtime(self, data_processor: DataProcessor, target_date: str, 
                        current_time: str, save_plot: bool = True) -> Dict:
        """
        Make real-time prediction for a specific date and time
        
        Args:
            data_processor: DataProcessor instance
            target_date: Date string 'YYYY-MM-DD'
            current_time: Time string 'HH:MM' or full datetime string
            save_plot: Whether to save prediction plot
            
        Returns:
            Dictionary with prediction results
        """
        print(f"Making real-time prediction for {target_date} at {current_time}")
        
        # Load full day data
        config = get_config()
        full_day_data = data_processor.get_processed_trading_day(
            config.DEFAULT_DATA_FILE,
            target_date,
            include_indicators=True
        )
        
        # Slice data to current time
        if ':' in current_time and len(current_time) <= 5:
            # Time only provided
            from datetime import datetime
            import pytz
            eastern_tz = pytz.timezone('US/Eastern')
            data_date = datetime.strptime(target_date, '%Y-%m-%d').date()
            hour, minute = map(int, current_time.split(':'))
            current_dt = eastern_tz.localize(
                datetime.combine(data_date, datetime.min.time()).replace(hour=hour, minute=minute)
            )
        else:
            current_dt = pd.to_datetime(current_time)
            if current_dt.tzinfo is None:
                import pytz
                eastern_tz = pytz.timezone('US/Eastern')
                current_dt = eastern_tz.localize(current_dt)
        
        # Get historical data up to current time
        historical_data = data_processor.slice_data_to_current(full_day_data, current_dt.strftime('%Y-%m-%d %H:%M:%S'))
        
        if len(historical_data) < self.model_config['context_length']:
            raise ValueError(f"Insufficient historical data: {len(historical_data)} points")
        
        # Make prediction
        prediction_results = self.predict(historical_data)
        
        # Add metadata
        prediction_results.update({
            'target_date': target_date,
            'current_time': current_time,
            'current_datetime': current_dt,
            'historical_data_points': len(historical_data),
            'last_price': historical_data['close_norm'].iloc[-1] if 'close_norm' in historical_data.columns else historical_data['close'].iloc[-1]
        })
        
        # Create visualization if requested
        if save_plot:
            self._plot_prediction(historical_data, prediction_results, target_date, current_time)
        
        return prediction_results
    
    def _plot_prediction(self, historical_data: pd.DataFrame, prediction_results: Dict, 
                        target_date: str, current_time: str):
        """Create and save prediction plot"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
        
        # Plot historical data
        price_col = 'close_norm' if 'close_norm' in historical_data.columns else 'close'
        ax1.plot(historical_data.index, historical_data[price_col], 'b-', linewidth=1.5, label='Historical Price')
        
        # Plot forecast if available
        if 'forecast' in prediction_results:
            forecast = prediction_results['forecast']
            current_dt = prediction_results['current_datetime']
            
            # Create future timestamps
            future_timestamps = [current_dt + timedelta(minutes=i+1) for i in range(len(forecast))]
            
            ax1.plot(future_timestamps, forecast, 'r-', linewidth=2, alpha=0.8, label='Forecast')
            ax1.scatter(current_dt, prediction_results['last_price'], color='red', s=100, zorder=5, label='Prediction Start')
        
        # Add reference lines
        if 'close_norm' in historical_data.columns:
            ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Previous EOD (1.00)')
        
        # Format plot
        ax1.set_ylabel('Normalized Price' if 'close_norm' in historical_data.columns else 'Price ($)')
        ax1.set_title(f'Finetuned FlaMinGo Prediction - {target_date} at {current_time}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Classification results
        if 'predicted_direction' in prediction_results:
            direction = prediction_results['predicted_direction']
            confidence = prediction_results['direction_confidence']
            
            # Color based on direction
            colors = {'UP': 'green', 'DOWN': 'red', 'FLAT': 'gray'}
            color = colors.get(direction, 'blue')
            
            ax2.bar(['Prediction'], [confidence], color=color, alpha=0.7)
            ax2.set_ylabel('Confidence')
            ax2.set_title(f'Direction: {direction} (Confidence: {confidence:.1%})')
            ax2.set_ylim(0, 1)
            
            # Add text annotation
            if 'expected_move_pct' in prediction_results:
                move_pct = prediction_results['expected_move_pct']
                ax2.text(0, confidence/2, f'{move_pct:+.2f}%', ha='center', va='center', 
                        fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"finetuned_prediction_{target_date}_{current_time.replace(':', '')}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Prediction plot saved as {plot_filename}")

def run_inference_demo():
    """Run a demonstration of the finetuned model inference"""
    
    # Configuration
    MODEL_PATH = "finetuned_flamingo_stock_model.pth"
    TARGET_DATE = "2025-05-19"
    CURRENT_TIME = "10:30"  # 10:30 AM ET
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please run finetuning.py first to create the model.")
        return
    
    print("🚀 Running Finetuned FlaMinGo Inference Demo")
    print("="*50)
    
    # Initialize predictor
    try:
        predictor = FinetunedFlaMinGoPredictor(MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Initialize data processor
    config = get_config()
    data_processor = DataProcessor(config.DATA_PATH)
    
    # Make real-time prediction
    try:
        results = predictor.predict_realtime(
            data_processor, TARGET_DATE, CURRENT_TIME, save_plot=True
        )
        
        print("\n📊 Prediction Results:")
        print(f"Date: {results['target_date']}")
        print(f"Time: {results['current_time']}")
        print(f"Historical data points: {results['historical_data_points']}")
        print(f"Last price: {results['last_price']:.4f}")
        
        if 'predicted_direction' in results:
            print(f"\n🎯 Classification:")
            print(f"Direction: {results['predicted_direction']}")
            print(f"Confidence: {results['direction_confidence']:.1%}")
        
        if 'expected_move_pct' in results:
            print(f"\n📈 Forecast:")
            print(f"Expected move: {results['expected_move_pct']:+.2f}%")
        
        print("\n✅ Inference completed successfully!")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()

def batch_inference(model_path: str, target_dates: List[str], time_points: List[str]):
    """
    Run batch inference on multiple dates and times
    
    Args:
        model_path: Path to the finetuned model
        target_dates: List of dates to test
        time_points: List of times to test for each date
    """
    print(f"🔄 Running batch inference on {len(target_dates)} dates and {len(time_points)} time points")
    
    # Initialize predictor
    predictor = FinetunedFlaMinGoPredictor(model_path)
    
    # Initialize data processor
    config = get_config()
    data_processor = DataProcessor(config.DATA_PATH)
    
    results = []
    
    for date in target_dates:
        for time_point in time_points:
            try:
                print(f"Processing {date} at {time_point}...")
                
                result = predictor.predict_realtime(
                    data_processor, date, time_point, save_plot=False
                )
                
                results.append({
                    'date': date,
                    'time': time_point,
                    'direction': result.get('predicted_direction', 'N/A'),
                    'confidence': result.get('direction_confidence', 0.0),
                    'expected_move': result.get('expected_move_pct', 0.0),
                    'last_price': result.get('last_price', 0.0)
                })
                
            except Exception as e:
                print(f"❌ Error processing {date} at {time_point}: {e}")
                results.append({
                    'date': date,
                    'time': time_point,
                    'direction': 'ERROR',
                    'confidence': 0.0,
                    'expected_move': 0.0,
                    'last_price': 0.0
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('batch_inference_results.csv', index=False)
    print(f"✅ Batch inference completed. Results saved to batch_inference_results.csv")
    
    # Print summary
    print("\n📊 Batch Inference Summary:")
    print(f"Total predictions: {len(results)}")
    direction_counts = results_df['direction'].value_counts()
    for direction, count in direction_counts.items():
        print(f"{direction}: {count} ({count/len(results)*100:.1f}%)")
    
    return results_df

def run_random_day_inference():
    """Run inference on a random day from the dataset and create plots"""
    
    # Try to find latest checkpoint first
    try:
        predictor = FinetunedFlaMinGoPredictor(use_latest_checkpoint=True)
    except FileNotFoundError:
        print("❌ No checkpoints found. Please run finetuning.py first.")
        return
    
    # Pick a random day from the large dataset
    import random
    from pathlib import Path
    
    # Use the large dataset path from finetuning configuration
    large_dataset_path = Path("databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv")
    
    if not large_dataset_path.exists():
        print(f"❌ Large dataset not found: {large_dataset_path}")
        print("Please ensure the dataset is available.")
        return
    
    print("🔍 Scanning dataset for available dates...")
    
    # Read a sample of the dataset to find available dates
    import pandas as pd
    sample_data = pd.read_csv(large_dataset_path, nrows=100000)  # Read first 100k rows
    sample_data['ts_event'] = pd.to_datetime(sample_data['ts_event'])
    
    # Get unique dates
    available_dates = sample_data['ts_event'].dt.date.unique()
    available_dates = sorted([str(date) for date in available_dates])
    
    # Pick a random date
    random_date = random.choice(available_dates)
    print(f"🎲 Selected random date: {random_date}")
    
    # Create a simple prediction loop similar to flamingo_loop.py
    run_finetuned_prediction_loop(predictor, random_date, large_dataset_path)

def run_finetuned_prediction_loop(predictor, target_date, dataset_path):
    """Run a prediction loop similar to flamingo_loop.py but using finetuned model"""
    
    print(f"\n🚀 Running finetuned model prediction loop for {target_date}")
    print("="*60)
    
    # Load data for the target date
    import pandas as pd
    
    print("📊 Loading data for target date...")
    
    # Read the dataset in chunks to find data for our target date
    chunk_size = 50000
    day_data = []
    
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
        chunk['ts_event'] = pd.to_datetime(chunk['ts_event'])
        chunk['date'] = chunk['ts_event'].dt.date
        
        # Filter for our target date
        target_date_obj = pd.to_datetime(target_date).date()
        day_chunk = chunk[chunk['date'] == target_date_obj]
        
        if len(day_chunk) > 0:
            day_data.append(day_chunk)
    
    if not day_data:
        print(f"❌ No data found for {target_date}")
        return
    
    # Combine all chunks for the day
    full_day_data = pd.concat(day_data, ignore_index=True)
    
    # Filter for ES futures symbols (4-letter codes)
    es_symbols = full_day_data[
        (full_day_data['symbol'].str.len() == 4) & 
        (full_day_data['symbol'].str.startswith('ES'))
    ]
    
    if len(es_symbols) == 0:
        print(f"❌ No ES futures data found for {target_date}")
        return
    
    # Pick the most active symbol for the day
    symbol_counts = es_symbols['symbol'].value_counts()
    most_active_symbol = symbol_counts.index[0]
    
    symbol_data = es_symbols[es_symbols['symbol'] == most_active_symbol].copy()
    symbol_data = symbol_data.sort_values('ts_event').reset_index(drop=True)
    
    print(f"📈 Using symbol: {most_active_symbol} ({len(symbol_data):,} data points)")
    
    # Set up time parameters
    start_time = '09:45'  # Start predictions at 9:45 AM ET
    end_time = '15:30'    # End at 3:30 PM ET
    time_step_minutes = 30  # Make predictions every 30 minutes
    
    # Convert to datetime index
    symbol_data['ts_event'] = pd.to_datetime(symbol_data['ts_event'])
    symbol_data.set_index('ts_event', inplace=True)
    
    # Create prediction loop similar to flamingo_loop.py
    create_finetuned_prediction_plot(predictor, symbol_data, target_date, start_time, end_time, time_step_minutes)

def create_finetuned_prediction_plot(predictor, data, target_date, start_time, end_time, time_step_minutes):
    """Create prediction plot using finetuned model"""
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    import pytz
    
    # Enable interactive plotting
    plt.ion()
    
    # Create figure similar to flamingo_loop.py
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[3, 1])
    
    # Parse times
    start_hour, start_minute = map(int, start_time.split(':'))
    end_hour, end_minute = map(int, end_time.split(':'))
    
    # Create timezone objects
    est_tz = pytz.timezone('US/Eastern')
    
    # Initialize prediction storage
    all_predictions = []
    
    # Loop through time
    target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
    current_time = est_tz.localize(
        datetime.combine(target_date_obj.date(), datetime.min.time()).replace(hour=start_hour, minute=start_minute)
    )
    end_datetime = est_tz.localize(
        datetime.combine(target_date_obj.date(), datetime.min.time()).replace(hour=end_hour, minute=end_minute)
    )
    
    loop_count = 0
    colors = plt.cm.rainbow(np.linspace(0, 1, 20))
    
    while current_time <= end_datetime:
        loop_count += 1
        print(f"\n--- Loop {loop_count}: NOW = {current_time.strftime('%I:%M %p')} ET ---")
        
        # Get historical data up to current time
        historical_mask = data.index <= current_time
        historical_data = data[historical_mask]
        
        if len(historical_data) < predictor.model_config['context_length']:
            print(f"Skipping - insufficient data ({len(historical_data)} points)")
            current_time += timedelta(minutes=time_step_minutes)
            continue
        
        # Take last context_length points
        context_data = historical_data.tail(predictor.model_config['context_length'])
        current_price = context_data['close'].iloc[-1]
        
        try:
            # Create a simple DataFrame with close prices for prediction
            pred_data = pd.DataFrame({
                'close': context_data['close']
            })
            
            # Make prediction
            results = predictor.predict(pred_data)
            
            print(f"Last price: ${current_price:.2f}")
            
            if 'predicted_direction' in results:
                print(f"Direction: {results['predicted_direction']}")
                print(f"Confidence: {results['direction_confidence']:.1%}")
            
            if 'expected_move_pct' in results:
                print(f"Expected move: {results['expected_move_pct']:+.2f}%")
            
            # Store prediction
            prediction_entry = {
                'loop': loop_count,
                'now_time': current_time,
                'last_price': current_price,
                'prediction': results.get('forecast', []),
                'direction': results.get('predicted_direction', 'UNKNOWN'),
                'confidence': results.get('direction_confidence', 0.0),
                'expected_move_pct': results.get('expected_move_pct', 0.0)
            }
            
            all_predictions.append(prediction_entry)
            
            # Update plot
            update_finetuned_plot(fig, ax1, ax2, prediction_entry, data, loop_count-1, all_predictions, colors)
            
            # Small delay for visualization
            plt.pause(0.5)
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            import traceback
            traceback.print_exc()
        
        # Move to next time step
        current_time += timedelta(minutes=time_step_minutes)
    
    print(f"\n✅ Completed {loop_count} prediction loops")
    
    # Save the final plot
    plot_filename = f'finetuned_inference_{target_date.replace("-", "")}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved as {plot_filename}")
    
    # Keep plot open
    plt.ioff()
    plt.show()
    
    return all_predictions

def update_finetuned_plot(fig, ax1, ax2, prediction_entry, full_data, loop_count, all_predictions, colors):
    """Update the plot with new prediction data"""
    
    color = colors[loop_count % 20]
    current_time = prediction_entry['now_time']
    
    # Clear and redraw main plot
    ax1.clear()
    
    # Plot historical data up to current time
    historical_mask = full_data.index <= current_time
    historical_data = full_data[historical_mask]
    
    if len(historical_data) > 0:
        ax1.plot(historical_data.index, historical_data['close'], 
                'b-', linewidth=1.5, label='Historical Price', alpha=0.9)
    
    # Plot all previous predictions (faded)
    for i, prev_pred in enumerate(all_predictions[:-1]):
        prev_color = colors[i % 20]
        if len(prev_pred['prediction']) > 0:
            # Create timestamps for forecast
            forecast_times = []
            start_time = prev_pred['now_time']
            for j in range(len(prev_pred['prediction'])):
                forecast_times.append(start_time + timedelta(minutes=j+1))
            
            ax1.plot(forecast_times, prev_pred['prediction'], 
                    color=prev_color, alpha=0.3, linewidth=1.0)
            ax1.scatter(start_time, prev_pred['last_price'], 
                       color=prev_color, s=30, alpha=0.5, zorder=4)
    
    # Plot current prediction (highlighted)
    if len(prediction_entry['prediction']) > 0:
        forecast_times = []
        for j in range(len(prediction_entry['prediction'])):
            forecast_times.append(current_time + timedelta(minutes=j+1))
        
        ax1.plot(forecast_times, prediction_entry['prediction'], 
                color=color, alpha=0.8, linewidth=2.5,
                label=f"{current_time.strftime('%I:%M%p')} - {prediction_entry['direction']}")
    
    # Mark prediction start point
    ax1.scatter(current_time, prediction_entry['last_price'], 
               color=color, s=80, zorder=5, edgecolor='black', linewidth=1)
    
    # Add current time marker
    ax1.axvline(x=current_time, color='red', linestyle='--', alpha=0.4, linewidth=2)
    
    # Format plot
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'Finetuned FlaMinGo Predictions - {current_time.strftime("%Y-%m-%d")}', 
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=8)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    # Simple prediction summary in bottom subplot
    ax2.clear()
    ax2.text(0.5, 0.5, f"Latest Prediction: {prediction_entry['direction']}\n"
                       f"Confidence: {prediction_entry['confidence']:.1%}\n"
                       f"Expected Move: {prediction_entry['expected_move_pct']:+.2f}%",
             ha='center', va='center', transform=ax2.transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Finetuned FlaMinGo Inference (default: random day with plotting)')
    parser.add_argument('--model', default=None, 
                       help='Path to finetuned model (default: use latest checkpoint)')
    parser.add_argument('--date', default='2025-05-19', 
                       help='Target date for single inference (YYYY-MM-DD)')
    parser.add_argument('--time', default='10:30', 
                       help='Current time for single inference (HH:MM)')
    parser.add_argument('--batch', action='store_true', 
                       help='Run batch inference instead of random day')
    parser.add_argument('--random-day', action='store_true',
                       help='Explicitly run random day inference (default behavior)')
    parser.add_argument('--use-checkpoint', action='store_true',
                       help='Use latest checkpoint instead of final model')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch inference
        test_dates = ['2025-05-19', '2025-05-20', '2025-05-21']
        test_times = ['09:45', '10:30', '11:15', '14:00']
        model_path = args.model if not args.use_checkpoint else None
        batch_inference(model_path, test_dates, test_times)
    elif args.date != '2025-05-19' or args.time != '10:30':
        # Single inference demo (only if user specified custom date/time)
        run_inference_demo()
    else:
        # Default: Random day inference with plotting
        print("🎲 Running random day inference by default (use --batch for batch mode)")
        run_random_day_inference()
