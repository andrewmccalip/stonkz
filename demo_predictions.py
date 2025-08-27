#!/usr/bin/env python3
"""
Demo Predictions - Showcase both TimesFM and Kronos prediction systems with plotting.
Demonstrates how to use the prediction interfaces and visualization in a real scenario.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import prediction systems
try:
    from prediction_timesfm import predict_timesfm, PredictionError as TimesFMError, get_error_message as get_timesfm_error
    TIMESFM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ TimesFM not available: {e}")
    TIMESFM_AVAILABLE = False

try:
    from prediction_kronos import predict_kronos, PredictionError as KronosError, get_error_message as get_kronos_error
    KRONOS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Kronos not available: {e}")
    KRONOS_AVAILABLE = False

# Import plotting
from plotting import plot_prediction_results, create_comparison_plot

# Configuration
SCRIPT_DIR = Path(__file__).parent
DATASETS_DIR = SCRIPT_DIR / "datasets" / "ES"
CONTEXT_LENGTH = 416
HORIZON_LENGTH = 96

def load_random_dataset():
    """Load a random dataset for testing"""
    
    csv_files = list(DATASETS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {DATASETS_DIR}")
        return None, None
    
    # Pick random file
    random_file = random.choice(csv_files)
    print(f"📁 Selected dataset: {random_file.name}")
    
    # Load data
    df = pd.read_csv(random_file)
    df['timestamp_pt'] = pd.to_datetime(df['timestamp_pt'])
    
    # Find 7 AM data
    df_7am = df[df['timestamp_pt'].dt.hour == 7]
    if len(df_7am) == 0:
        print("⚠️ No 7 AM data found, using first available data")
        start_idx = CONTEXT_LENGTH
    else:
        start_idx = df_7am.index[0]
    
    if start_idx < CONTEXT_LENGTH:
        print(f"❌ Not enough historical data (need {CONTEXT_LENGTH}, have {start_idx})")
        return None, None
    
    print(f"🕰️ Starting prediction at: {df.iloc[start_idx]['timestamp_pt']}")
    
    return df, start_idx

def demo_timesfm_prediction(df, start_idx):
    """Demo TimesFM prediction with plotting"""
    
    if not TIMESFM_AVAILABLE:
        print("⚠️ TimesFM not available, skipping...")
        return None
    
    print("\n" + "="*60)
    print("🔮 TimesFM Prediction Demo")
    print("="*60)
    
    # Extract close prices (already normalized)
    historical_data = df['close'].iloc[start_idx - CONTEXT_LENGTH:start_idx].values
    
    print(f"📈 Historical data: {len(historical_data)} points")
    print(f"   Range: [{historical_data.min():.6f}, {historical_data.max():.6f}]")
    
    # Make prediction
    predictions, error_code = predict_timesfm(historical_data, verbose=True)
    
    if error_code == TimesFMError.SUCCESS:
        print(f"\n✅ TimesFM prediction successful!")
        
        # Get ground truth if available
        future_end_idx = start_idx + HORIZON_LENGTH
        actual_future = None
        if future_end_idx <= len(df):
            actual_future = df['close'].iloc[start_idx:future_end_idx].values
        
        # Create plot
        plot_result = plot_prediction_results(
            context_data=historical_data,
            prediction_data=predictions,
            ground_truth_data=actual_future,
            title="TimesFM Prediction Demo",
            model_name="TimesFM",
            show_plot=False,
            verbose=True
        )
        
        return {
            'model': 'TimesFM',
            'context': historical_data,
            'prediction': predictions,
            'ground_truth': actual_future,
            'plot_path': plot_result['plot_path'],
            'metrics': plot_result['metrics']
        }
    else:
        print(f"❌ TimesFM prediction failed: {get_timesfm_error(error_code)}")
        return None

def demo_kronos_prediction(df, start_idx):
    """Demo Kronos prediction with plotting"""
    
    if not KRONOS_AVAILABLE:
        print("⚠️ Kronos not available, skipping...")
        return None
    
    print("\n" + "="*60)
    print("🤖 Kronos Prediction Demo")
    print("="*60)
    
    # Extract OHLCV data (already normalized)
    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
    historical_data = df[ohlcv_columns].iloc[start_idx - CONTEXT_LENGTH:start_idx]
    
    print(f"📈 Historical OHLCV data: {len(historical_data)} rows")
    print(f"   Close range: [{historical_data['close'].min():.6f}, {historical_data['close'].max():.6f}]")
    print(f"   Volume range: [{historical_data['volume'].min():.0f}, {historical_data['volume'].max():.0f}]")
    
    # Make prediction
    predictions, error_code = predict_kronos(historical_data, verbose=True)
    
    if error_code == KronosError.SUCCESS:
        print(f"\n✅ Kronos prediction successful!")
        
        # Get ground truth if available
        future_end_idx = start_idx + HORIZON_LENGTH
        actual_future = None
        if future_end_idx <= len(df):
            actual_future = df['close'].iloc[start_idx:future_end_idx].values
        
        # Create plot with volume
        plot_result = plot_prediction_results(
            context_data=historical_data,  # Full OHLCV DataFrame
            prediction_data=predictions,
            ground_truth_data=actual_future,
            title="Kronos Prediction Demo",
            model_name="Kronos",
            show_plot=False,
            verbose=True
        )
        
        return {
            'model': 'Kronos',
            'context': historical_data['close'].values,  # Just close for comparison
            'prediction': predictions,
            'ground_truth': actual_future,
            'plot_path': plot_result['plot_path'],
            'metrics': plot_result['metrics']
        }
    else:
        print(f"❌ Kronos prediction failed: {get_kronos_error(error_code)}")
        return None

def create_model_comparison(timesfm_result, kronos_result):
    """Create a comparison plot between TimesFM and Kronos"""
    
    if timesfm_result is None and kronos_result is None:
        print("❌ No successful predictions to compare")
        return None
    
    print("\n" + "="*60)
    print("📊 Model Comparison")
    print("="*60)
    
    comparison_data = {}
    
    if timesfm_result is not None:
        comparison_data['TimesFM'] = {
            'context': timesfm_result['context'],
            'prediction': timesfm_result['prediction'],
            'ground_truth': timesfm_result['ground_truth']
        }
    
    if kronos_result is not None:
        comparison_data['Kronos'] = {
            'context': kronos_result['context'],
            'prediction': kronos_result['prediction'],
            'ground_truth': kronos_result['ground_truth']
        }
    
    if len(comparison_data) > 1:
        comparison_path = create_comparison_plot(
            comparison_data,
            title="TimesFM vs Kronos Prediction Comparison"
        )
        print(f"📊 Comparison plot saved to: {comparison_path}")
        return comparison_path
    else:
        print("⚠️ Need at least 2 models for comparison")
        return None

def print_results_summary(timesfm_result, kronos_result):
    """Print a summary of results"""
    
    print("\n" + "="*60)
    print("📋 Results Summary")
    print("="*60)
    
    if timesfm_result is not None:
        metrics = timesfm_result['metrics']
        print(f"\n🔮 TimesFM Results:")
        print(f"   Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
        print(f"   Correlation: {metrics['correlation']:.3f}")
        print(f"   MSE: {metrics['mse']:.6f}")
        print(f"   MAE: {metrics['mae']:.6f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        print(f"   Plot: {timesfm_result['plot_path']}")
    
    if kronos_result is not None:
        metrics = kronos_result['metrics']
        print(f"\n🤖 Kronos Results:")
        print(f"   Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
        print(f"   Correlation: {metrics['correlation']:.3f}")
        print(f"   MSE: {metrics['mse']:.6f}")
        print(f"   MAE: {metrics['mae']:.6f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        print(f"   Plot: {kronos_result['plot_path']}")
    
    # Compare models if both available
    if timesfm_result is not None and kronos_result is not None:
        print(f"\n🏆 Model Comparison:")
        
        timesfm_acc = timesfm_result['metrics']['directional_accuracy']
        kronos_acc = kronos_result['metrics']['directional_accuracy']
        
        if timesfm_acc > kronos_acc:
            print(f"   Best Directional Accuracy: TimesFM ({timesfm_acc:.1f}% vs {kronos_acc:.1f}%)")
        elif kronos_acc > timesfm_acc:
            print(f"   Best Directional Accuracy: Kronos ({kronos_acc:.1f}% vs {timesfm_acc:.1f}%)")
        else:
            print(f"   Directional Accuracy: Tie ({timesfm_acc:.1f}%)")
        
        timesfm_corr = timesfm_result['metrics']['correlation']
        kronos_corr = kronos_result['metrics']['correlation']
        
        if timesfm_corr > kronos_corr:
            print(f"   Best Correlation: TimesFM ({timesfm_corr:.3f} vs {kronos_corr:.3f})")
        elif kronos_corr > timesfm_corr:
            print(f"   Best Correlation: Kronos ({kronos_corr:.3f} vs {timesfm_corr:.3f})")
        else:
            print(f"   Correlation: Similar ({timesfm_corr:.3f} vs {kronos_corr:.3f})")

def main():
    """Main demo function"""
    
    print("🚀 Prediction Systems Demo")
    print("=" * 60)
    print(f"Available models:")
    print(f"  TimesFM: {'✅' if TIMESFM_AVAILABLE else '❌'}")
    print(f"  Kronos:  {'✅' if KRONOS_AVAILABLE else '❌'}")
    
    if not TIMESFM_AVAILABLE and not KRONOS_AVAILABLE:
        print("❌ No prediction models available!")
        return
    
    # Load random dataset
    df, start_idx = load_random_dataset()
    if df is None:
        return
    
    # Run predictions
    timesfm_result = None
    kronos_result = None
    
    if TIMESFM_AVAILABLE:
        timesfm_result = demo_timesfm_prediction(df, start_idx)
    
    if KRONOS_AVAILABLE:
        kronos_result = demo_kronos_prediction(df, start_idx)
    
    # Create comparison if both models worked
    comparison_path = create_model_comparison(timesfm_result, kronos_result)
    
    # Print summary
    print_results_summary(timesfm_result, kronos_result)
    
    print(f"\n📁 All plots saved to: prediction_plots/")
    print("🎉 Demo completed!")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    main()
