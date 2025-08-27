#!/usr/bin/env python3
"""
Plotting Module - Comprehensive visualization functions for prediction results.
Provides consistent, publication-quality plots following established best practices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration & Constants
# ==============================================================================

# Color Scheme (consistent with existing codebase)
COLORS = {
    'historical': '#1f77b4',    # Blue - historical/context data
    'ground_truth': '#2ca02c',  # Green - actual future data  
    'prediction': '#ff7f0e',    # Orange - model predictions
    'current_time': '#d62728',  # Red - current time marker
    'reference': '#7f7f7f',     # Gray - reference lines
    'volume': '#9467bd',        # Purple - volume data
    'secondary': '#8c564b'      # Brown - secondary data
}

# Default styling
DEFAULT_STYLE = {
    'figure_size': (16, 10),
    'title_fontsize': 14,
    'label_fontsize': 12,
    'legend_fontsize': 11,
    'grid_alpha': 0.3,
    'line_alpha': 0.8,
    'line_width': 2,
    'marker_size': 50,
    'dpi': 150
}

# Plot directory
SCRIPT_DIR = Path(__file__).parent
PLOTS_DIR = SCRIPT_DIR / "prediction_plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ==============================================================================
# Main Plotting Functions
# ==============================================================================

def plot_comprehensive_dashboard(
    context_data,
    prediction_data=None,
    ground_truth_data=None,
    title="Comprehensive Prediction Dashboard",
    model_name="Model",
    save_path=None,
    show_plot=True,
    figsize=None,
    normalize_to_start=True,
    epoch=0,
    training_history=None,
    verbose=False
):
    """
    Create a comprehensive prediction dashboard with simplified layout.
    
    Layout:
    - Top row (3-wide): Sample vs Prediction comparison
    - Middle row (3-wide): Training and Validation Loss vs Epoch
    - Bottom row: Additional analysis plots (simplified)
    
    Args:
        context_data (array-like): Historical/context data
        prediction_data (array-like): Model predictions  
        ground_truth_data (array-like): Actual future values
        title (str): Plot title
        model_name (str): Name of the model
        save_path (str/Path, optional): Path to save plot
        show_plot (bool): Whether to display the plot
        figsize (tuple, optional): Figure size (width, height)
        normalize_to_start (bool): Whether to normalize data to start at 1.0
        epoch (int): Current training epoch
        training_history (dict, optional): Training history with 'train_loss' and 'val_loss' lists
        verbose (bool): Print detailed information
    
    Returns:
        dict: Dictionary containing plot information and metrics
    """
    
    if verbose:
        print(f"🎨 Creating comprehensive dashboard: {title}")
    
    # Prepare data
    plot_data = _prepare_plot_data(
        context_data, prediction_data, ground_truth_data, None,
        None, normalize_to_start, verbose
    )
    
    if plot_data is None:
        print("❌ Failed to prepare plot data")
        return None
    
    # Set figure size (increased height for 3 rows)
    figsize = figsize or (20, 16)
    
    # Create figure with subplot layout: 3 rows, 3 columns
    fig = plt.figure(figsize=figsize)
    
    # Top row: 3-wide sample vs prediction plot
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=3, fig=fig)
    
    # Middle row: Training and validation loss vs epoch (3-wide)
    ax_loss = plt.subplot2grid((3, 3), (1, 0), colspan=3, fig=fig)
    
    # Bottom row: Additional plots (simplified)
    ax_metrics = plt.subplot2grid((3, 3), (2, 0), fig=fig)
    ax_residuals = plt.subplot2grid((3, 3), (2, 1), fig=fig)
    ax_correlation = plt.subplot2grid((3, 3), (2, 2), fig=fig)
    
    # Plot main prediction comparison (3-wide top panel)
    _plot_comprehensive_main(ax_main, plot_data, model_name, epoch, verbose)
    
    # Plot training and validation loss (3-wide middle panel)
    _plot_training_loss(ax_loss, training_history, epoch, verbose)
    
    # Plot simplified bottom panels
    metrics = {}
    if plot_data['prediction'] is not None and plot_data['ground_truth'] is not None:
        metrics = _calculate_metrics(plot_data['prediction'], plot_data['ground_truth'])
        _plot_metrics_summary(ax_metrics, metrics, model_name)
        _plot_residuals_analysis(ax_residuals, plot_data, metrics)
        _plot_correlation_analysis(ax_correlation, plot_data, metrics)
    
    # Format the overall plot
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = PLOTS_DIR / f"{model_name.lower()}_dashboard_{timestamp}.png"
    else:
        save_path = Path(save_path)
    
    plt.savefig(save_path, dpi=DEFAULT_STYLE['dpi'], bbox_inches='tight', facecolor='white')
    
    if verbose:
        print(f"💾 Dashboard saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Return results
    result = {
        'plot_path': save_path,
        'metrics': metrics,
        'plot_data': plot_data,
        'figure_size': figsize
    }
    
    if verbose:
        print(f"✅ Dashboard creation completed")
    
    return result

def plot_prediction_results(
    context_data,
    prediction_data=None,
    ground_truth_data=None,
    volume_data=None,
    timestamps=None,
    title="Prediction Results",
    model_name="Model",
    save_path=None,
    show_plot=True,
    figsize=None,
    normalize_to_start=True,
    add_metrics=True,
    verbose=False
):
    """
    Create a comprehensive prediction visualization plot.
    
    Args:
        context_data (array-like): Historical/context data (required)
                                  Can be 1D array of close prices or 2D OHLCV data
        prediction_data (array-like, optional): Model predictions
        ground_truth_data (array-like, optional): Actual future values for comparison
        volume_data (array-like, optional): Volume data for context period
        timestamps (array-like, optional): Timestamps for x-axis. If None, uses indices
        title (str): Plot title
        model_name (str): Name of the model for labeling
        save_path (str/Path, optional): Path to save plot. If None, auto-generates
        show_plot (bool): Whether to display the plot
        figsize (tuple, optional): Figure size (width, height)
        normalize_to_start (bool): Whether to normalize all data to start at 1.0
        add_metrics (bool): Whether to add performance metrics text box
        verbose (bool): Print detailed information
    
    Returns:
        dict: Dictionary containing plot information and calculated metrics
    
    Example:
        >>> # Basic usage with TimesFM output
        >>> result = plot_prediction_results(
        ...     context_data=historical_close_prices,
        ...     prediction_data=timesfm_predictions,
        ...     ground_truth_data=actual_future_prices,
        ...     title="TimesFM Prediction",
        ...     model_name="TimesFM"
        ... )
        
        >>> # Usage with Kronos OHLCV output
        >>> result = plot_prediction_results(
        ...     context_data=ohlcv_data,  # m×5 array
        ...     prediction_data=kronos_predictions,
        ...     volume_data=ohlcv_data[:, 4],  # Volume column
        ...     title="Kronos Prediction",
        ...     model_name="Kronos"
        ... )
    """
    
    if verbose:
        print(f"🎨 Creating prediction plot: {title}")
    
    # Validate and prepare data
    plot_data = _prepare_plot_data(
        context_data, prediction_data, ground_truth_data, volume_data,
        timestamps, normalize_to_start, verbose
    )
    
    if plot_data is None:
        print("❌ Failed to prepare plot data")
        return None
    
    # Create figure and subplots
    figsize = figsize or DEFAULT_STYLE['figure_size']
    
    # Determine subplot layout based on available data
    has_volume = plot_data['volume'] is not None
    if has_volume:
        fig, (ax_main, ax_volume) = plt.subplots(2, 1, figsize=figsize, 
                                                height_ratios=[3, 1], 
                                                sharex=True)
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=figsize)
        ax_volume = None
    
    # Plot main price data
    _plot_main_data(ax_main, plot_data, model_name, verbose)
    
    # Plot volume data if available
    if has_volume and ax_volume is not None:
        _plot_volume_data(ax_volume, plot_data, verbose)
    
    # Add formatting and styling
    _format_plot(ax_main, ax_volume, title, plot_data, add_metrics)
    
    # Calculate and display metrics
    metrics = {}
    if add_metrics and plot_data['prediction'] is not None and plot_data['ground_truth'] is not None:
        metrics = _calculate_metrics(plot_data['prediction'], plot_data['ground_truth'])
        _add_metrics_text(ax_main, metrics, model_name)
    
    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = PLOTS_DIR / f"{model_name.lower()}_prediction_{timestamp}.png"
    else:
        save_path = Path(save_path)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DEFAULT_STYLE['dpi'], bbox_inches='tight', facecolor='white')
    
    if verbose:
        print(f"💾 Plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Return results
    result = {
        'plot_path': save_path,
        'metrics': metrics,
        'plot_data': plot_data,
        'figure_size': figsize
    }
    
    if verbose:
        print(f"✅ Plot creation completed")
        if metrics:
            print(f"📊 Calculated metrics: {list(metrics.keys())}")
    
    return result

# ==============================================================================
# Data Preparation Functions
# ==============================================================================

def _prepare_plot_data(context_data, prediction_data, ground_truth_data, volume_data, 
                      timestamps, normalize_to_start, verbose):
    """Prepare and validate all data for plotting"""
    
    try:
        # Process context data
        context_array = np.array(context_data)
        
        # Handle different context data formats
        if context_array.ndim == 1:
            # 1D array - assume close prices
            context_close = context_array
            context_volume = volume_data
        elif context_array.ndim == 2:
            # 2D array - assume OHLCV format
            if context_array.shape[1] >= 4:
                context_close = context_array[:, 3]  # Close prices (4th column)
                if context_array.shape[1] >= 5 and volume_data is None:
                    context_volume = context_array[:, 4]  # Volume (5th column)
                else:
                    context_volume = volume_data
            else:
                print("❌ 2D context data must have at least 4 columns (OHLC)")
                return None
        else:
            print("❌ Context data must be 1D or 2D array")
            return None
        
        # Process prediction data
        pred_array = None
        if prediction_data is not None:
            pred_array = np.array(prediction_data)
            if pred_array.ndim != 1:
                print("❌ Prediction data must be 1D array")
                return None
        
        # Process ground truth data
        gt_array = None
        if ground_truth_data is not None:
            gt_array = np.array(ground_truth_data)
            if gt_array.ndim != 1:
                print("❌ Ground truth data must be 1D array")
                return None
        
        # Process volume data
        vol_array = None
        if context_volume is not None:
            vol_array = np.array(context_volume)
            if len(vol_array) != len(context_close):
                print("⚠️ Volume data length doesn't match context data")
                vol_array = None
        
        # Create timestamps if not provided
        if timestamps is None:
            total_length = len(context_close)
            if pred_array is not None:
                total_length += len(pred_array)
            timestamps = np.arange(total_length)
        else:
            timestamps = np.array(timestamps)
        
        # Split timestamps
        context_len = len(context_close)
        context_timestamps = timestamps[:context_len]
        
        pred_timestamps = None
        if pred_array is not None:
            pred_start_idx = context_len
            pred_end_idx = pred_start_idx + len(pred_array)
            if len(timestamps) >= pred_end_idx:
                pred_timestamps = timestamps[pred_start_idx:pred_end_idx]
            else:
                # Generate prediction timestamps
                if isinstance(timestamps[0], (datetime, pd.Timestamp)):
                    last_time = timestamps[-1]
                    pred_timestamps = pd.date_range(
                        start=last_time + timedelta(minutes=1),
                        periods=len(pred_array),
                        freq='1min'
                    )
                else:
                    pred_timestamps = np.arange(context_len, context_len + len(pred_array))
        
        # Normalization
        base_value = 1.0
        if normalize_to_start and len(context_close) > 0:
            base_value = context_close[0]
            context_close = context_close / base_value
            
            if pred_array is not None:
                pred_array = pred_array / base_value
            
            if gt_array is not None:
                gt_array = gt_array / base_value
        
        if verbose:
            print(f"📊 Data prepared:")
            print(f"   Context: {len(context_close)} points")
            print(f"   Prediction: {len(pred_array) if pred_array is not None else 0} points")
            print(f"   Ground truth: {len(gt_array) if gt_array is not None else 0} points")
            print(f"   Volume: {'Yes' if vol_array is not None else 'No'}")
            print(f"   Normalized: {normalize_to_start} (base: {base_value:.6f})")
        
        return {
            'context': context_close,
            'context_timestamps': context_timestamps,
            'prediction': pred_array,
            'prediction_timestamps': pred_timestamps,
            'ground_truth': gt_array,
            'volume': vol_array,
            'base_value': base_value,
            'normalized': normalize_to_start
        }
        
    except Exception as e:
        print(f"❌ Error preparing plot data: {e}")
        return None

# ==============================================================================
# Plotting Functions
# ==============================================================================

def _plot_main_data(ax, plot_data, model_name, verbose):
    """Plot the main price data on the primary axis"""
    
    # Plot context (historical) data
    ax.plot(plot_data['context_timestamps'], plot_data['context'], 
           color=COLORS['historical'], linewidth=DEFAULT_STYLE['line_width'], 
           label='Historical Data', alpha=DEFAULT_STYLE['line_alpha'])
    
    # Add reference line for normalized data
    if plot_data['normalized']:
        ax.axhline(y=1.0, color=COLORS['reference'], linestyle=':', 
                  alpha=0.5, linewidth=1, label='Normalized Start (1.00)')
    
    # Plot prediction data
    if plot_data['prediction'] is not None and plot_data['prediction_timestamps'] is not None:
        ax.plot(plot_data['prediction_timestamps'], plot_data['prediction'],
               color=COLORS['prediction'], linewidth=DEFAULT_STYLE['line_width'],
               label=f'{model_name} Prediction', alpha=DEFAULT_STYLE['line_alpha'])
        
        # Connect context to prediction with thin line
        if len(plot_data['context']) > 0:
            last_context_time = plot_data['context_timestamps'][-1]
            last_context_price = plot_data['context'][-1]
            first_pred_time = plot_data['prediction_timestamps'][0]
            first_pred_price = plot_data['prediction'][0]
            
            ax.plot([last_context_time, first_pred_time], 
                   [last_context_price, first_pred_price],
                   color=COLORS['prediction'], linewidth=1, alpha=0.3)
    
    # Plot ground truth data
    if plot_data['ground_truth'] is not None and plot_data['prediction_timestamps'] is not None:
        # Use same timestamps as prediction for comparison
        gt_timestamps = plot_data['prediction_timestamps'][:len(plot_data['ground_truth'])]
        
        ax.plot(gt_timestamps, plot_data['ground_truth'],
               color=COLORS['ground_truth'], linewidth=DEFAULT_STYLE['line_width'] + 1,
               label='Ground Truth', alpha=DEFAULT_STYLE['line_alpha'])
        
        # Connect context to ground truth
        if len(plot_data['context']) > 0:
            last_context_time = plot_data['context_timestamps'][-1]
            last_context_price = plot_data['context'][-1]
            first_gt_time = gt_timestamps[0]
            first_gt_price = plot_data['ground_truth'][0]
            
            ax.plot([last_context_time, first_gt_time], 
                   [last_context_price, first_gt_price],
                   color=COLORS['ground_truth'], linewidth=1, alpha=0.3)
    
    # Add vertical line at prediction start
    if plot_data['prediction'] is not None and len(plot_data['context']) > 0:
        pred_start_time = plot_data['context_timestamps'][-1]
        ax.axvline(x=pred_start_time, color=COLORS['current_time'], 
                  linestyle='--', alpha=0.6, linewidth=2, label='Prediction Start')

def _plot_volume_data(ax, plot_data, verbose):
    """Plot volume data on the secondary axis"""
    
    if plot_data['volume'] is None:
        return
    
    # Plot volume bars
    ax.bar(plot_data['context_timestamps'], plot_data['volume'],
          color=COLORS['volume'], alpha=0.6, width=0.8, label='Volume')
    
    ax.set_ylabel('Volume', fontsize=DEFAULT_STYLE['label_fontsize'])
    ax.tick_params(axis='y', labelcolor=COLORS['volume'])

def _format_plot(ax_main, ax_volume, title, plot_data, add_metrics):
    """Apply formatting and styling to the plot"""
    
    # Main axis formatting
    ax_main.set_title(title, fontsize=DEFAULT_STYLE['title_fontsize'], fontweight='bold')
    
    if plot_data['normalized']:
        ax_main.set_ylabel('Normalized Price', fontsize=DEFAULT_STYLE['label_fontsize'])
    else:
        ax_main.set_ylabel('Price', fontsize=DEFAULT_STYLE['label_fontsize'])
    
    ax_main.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
    ax_main.legend(loc='upper left', fontsize=DEFAULT_STYLE['legend_fontsize'])
    
    # Format x-axis based on timestamp type
    if len(plot_data['context_timestamps']) > 0:
        if isinstance(plot_data['context_timestamps'][0], (datetime, pd.Timestamp)):
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)
        
        if ax_volume is None:
            ax_main.set_xlabel('Time', fontsize=DEFAULT_STYLE['label_fontsize'])
    
    # Volume axis formatting
    if ax_volume is not None:
        ax_volume.set_xlabel('Time', fontsize=DEFAULT_STYLE['label_fontsize'])
        ax_volume.grid(True, alpha=DEFAULT_STYLE['grid_alpha'], axis='x')
        
        # Format volume axis timestamps
        if isinstance(plot_data['context_timestamps'][0], (datetime, pd.Timestamp)):
            ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_volume.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45)

# ==============================================================================
# Metrics and Analysis
# ==============================================================================

def _calculate_metrics(predictions, ground_truth):
    """Calculate performance metrics for predictions vs ground truth"""
    
    # Ensure same length
    min_len = min(len(predictions), len(ground_truth))
    pred = predictions[:min_len]
    gt = ground_truth[:min_len]
    
    metrics = {}
    
    # Basic error metrics
    metrics['mse'] = np.mean((pred - gt) ** 2)
    metrics['mae'] = np.mean(np.abs(pred - gt))
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Percentage errors
    mape = np.mean(np.abs((gt - pred) / gt)) * 100
    metrics['mape'] = mape if not np.isnan(mape) else 0.0
    
    # Directional accuracy
    if len(pred) > 1:
        pred_direction = np.sign(np.diff(pred))
        gt_direction = np.sign(np.diff(gt))
        dir_accuracy = np.mean(pred_direction == gt_direction) * 100
        metrics['directional_accuracy'] = dir_accuracy
    else:
        metrics['directional_accuracy'] = 50.0
    
    # Correlation
    if np.std(pred) > 1e-6 and np.std(gt) > 1e-6:
        correlation = np.corrcoef(pred, gt)[0, 1]
        metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
    else:
        metrics['correlation'] = 0.0
    
    # Price change comparison
    pred_change = (pred[-1] - pred[0]) / pred[0] * 100 if pred[0] != 0 else 0
    gt_change = (gt[-1] - gt[0]) / gt[0] * 100 if gt[0] != 0 else 0
    metrics['pred_change_pct'] = pred_change
    metrics['gt_change_pct'] = gt_change
    
    return metrics

def _add_metrics_text(ax, metrics, model_name):
    """Add metrics text box to the plot"""
    
    metrics_text = f"""{model_name} Performance Metrics:

MSE: {metrics['mse']:.6f}
MAE: {metrics['mae']:.6f}
RMSE: {metrics['rmse']:.6f}
MAPE: {metrics['mape']:.2f}%

Directional Accuracy: {metrics['directional_accuracy']:.1f}%
Correlation: {metrics['correlation']:.3f}

Price Changes:
  Predicted: {metrics['pred_change_pct']:.2f}%
  Actual: {metrics['gt_change_pct']:.2f}%"""
    
    ax.text(0.5, 0.98, metrics_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='center', fontsize=10, fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))

# ==============================================================================
# Utility Functions
# ==============================================================================

def create_comparison_plot(results_dict, title="Model Comparison", save_path=None):
    """
    Create a comparison plot for multiple model results.
    
    Args:
        results_dict: Dictionary with model names as keys and prediction data as values
                     Each value should be a dict with 'context', 'prediction', 'ground_truth'
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Path to saved plot
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    
    # Colors for different models
    model_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    context_plotted = False
    gt_plotted = False
    
    for i, (model_name, data) in enumerate(results_dict.items()):
        color = model_colors[i % len(model_colors)]
        
        # Plot context data (only once)
        if not context_plotted and 'context' in data:
            context_len = len(data['context'])
            context_x = np.arange(context_len)
            ax.plot(context_x, data['context'], color=COLORS['historical'],
                   linewidth=2, label='Historical Data', alpha=0.8)
            context_plotted = True
        
        # Plot ground truth (only once)
        if not gt_plotted and 'ground_truth' in data and data['ground_truth'] is not None:
            gt_len = len(data['ground_truth'])
            gt_x = np.arange(context_len, context_len + gt_len)
            ax.plot(gt_x, data['ground_truth'], color=COLORS['ground_truth'],
                   linewidth=3, label='Ground Truth', alpha=0.9)
            gt_plotted = True
        
        # Plot prediction
        if 'prediction' in data and data['prediction'] is not None:
            pred_len = len(data['prediction'])
            pred_x = np.arange(context_len, context_len + pred_len)
            ax.plot(pred_x, data['prediction'], color=color,
                   linewidth=2, label=f'{model_name} Prediction', alpha=0.8)
    
    # Add prediction start line
    if context_plotted:
        ax.axvline(x=context_len, color=COLORS['current_time'],
                  linestyle='--', alpha=0.6, linewidth=2, label='Prediction Start')
    
    # Formatting
    ax.set_title(title, fontsize=DEFAULT_STYLE['title_fontsize'], fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=DEFAULT_STYLE['label_fontsize'])
    ax.set_ylabel('Normalized Price', fontsize=DEFAULT_STYLE['label_fontsize'])
    ax.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
    ax.legend(loc='upper left', fontsize=DEFAULT_STYLE['legend_fontsize'])
    
    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = PLOTS_DIR / f"model_comparison_{timestamp}.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DEFAULT_STYLE['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path

def clear_plots_directory():
    """Clear the plots directory"""
    if PLOTS_DIR.exists():
        plot_files = list(PLOTS_DIR.glob("*.png")) + list(PLOTS_DIR.glob("*.jpg"))
        for file in plot_files:
            try:
                file.unlink()
            except Exception as e:
                print(f"⚠️ Failed to delete {file.name}: {e}")
        print(f"🧹 Cleared {len(plot_files)} plot files from {PLOTS_DIR}")

# ==============================================================================
# Example Usage and Testing
# ==============================================================================

if __name__ == "__main__":
    print("🎨 Plotting Module Test")
    print("=" * 50)
    
    # Generate sample data for testing
    np.random.seed(42)
    
    # Sample context data (normalized close prices)
    context_length = 416
    context_data = np.ones(context_length)
    for i in range(1, context_length):
        change = np.random.normal(0, 0.002)  # Small random changes
        context_data[i] = context_data[i-1] * (1 + change)
    
    # Sample prediction data
    prediction_length = 96
    prediction_data = np.ones(prediction_length)
    prediction_data[0] = context_data[-1] * 1.001  # Start slightly above context end
    for i in range(1, prediction_length):
        change = np.random.normal(0.0001, 0.003)  # Slight upward trend with noise
        prediction_data[i] = prediction_data[i-1] * (1 + change)
    
    # Sample ground truth data
    ground_truth_data = np.ones(prediction_length)
    ground_truth_data[0] = context_data[-1] * 1.0005  # Start close to context end
    for i in range(1, prediction_length):
        change = np.random.normal(0.0002, 0.0025)  # Similar but different pattern
        ground_truth_data[i] = ground_truth_data[i-1] * (1 + change)
    
    # Sample volume data
    volume_data = np.random.randint(500, 2000, context_length)
    
    print("📊 Testing basic plotting function...")
    
    # Test 1: Basic plot with all data
    result1 = plot_prediction_results(
        context_data=context_data,
        prediction_data=prediction_data,
        ground_truth_data=ground_truth_data,
        volume_data=volume_data,
        title="Test Plot - All Data",
        model_name="TestModel",
        show_plot=False,
        verbose=True
    )
    
    print(f"✅ Test 1 completed: {result1['plot_path']}")
    
    # Test 2: Plot with only context and prediction
    result2 = plot_prediction_results(
        context_data=context_data,
        prediction_data=prediction_data,
        title="Test Plot - Prediction Only",
        model_name="TestModel",
        show_plot=False,
        add_metrics=False
    )
    
    print(f"✅ Test 2 completed: {result2['plot_path']}")
    
    # Test 3: OHLCV format input
    ohlcv_data = np.column_stack([
        context_data * 0.999,  # Open
        context_data * 1.002,  # High  
        context_data * 0.998,  # Low
        context_data,          # Close
        volume_data           # Volume
    ])
    
    result3 = plot_prediction_results(
        context_data=ohlcv_data,
        prediction_data=prediction_data,
        ground_truth_data=ground_truth_data,
        title="Test Plot - OHLCV Input",
        model_name="TestModel",
        show_plot=False
    )
    
    print(f"✅ Test 3 completed: {result3['plot_path']}")
    
    # Test 4: Model comparison
    comparison_data = {
        'Model A': {
            'context': context_data,
            'prediction': prediction_data,
            'ground_truth': ground_truth_data
        },
        'Model B': {
            'context': context_data,
            'prediction': prediction_data * 1.001 + np.random.normal(0, 0.001, len(prediction_data)),
            'ground_truth': ground_truth_data
        }
    }
    
    comparison_path = create_comparison_plot(
        comparison_data,
        title="Model Comparison Test"
    )
    
    print(f"✅ Test 4 completed: {comparison_path}")
    
    print(f"\n📁 All test plots saved to: {PLOTS_DIR}")
    print("🎨 Plotting module ready for use!")

# ==============================================================================
# Comprehensive Dashboard Helper Functions
# ==============================================================================

def _plot_comprehensive_main(ax, plot_data, model_name, epoch, verbose):
    """Plot the main 3-wide prediction comparison"""
    
    context = plot_data['context']
    prediction = plot_data['prediction']
    ground_truth = plot_data['ground_truth']
    timestamps = plot_data['timestamps']
    
    # Create time axis
    context_len = len(context)
    pred_len = len(prediction) if prediction is not None else 0
    
    if timestamps is not None:
        context_time = timestamps[:context_len]
        if prediction is not None:
            pred_time = timestamps[context_len:context_len + pred_len]
        else:
            pred_time = []
    else:
        context_time = np.arange(context_len)
        pred_time = np.arange(context_len, context_len + pred_len)
    
    # Plot context (historical data) - larger, more prominent
    ax.plot(context_time, context, color=COLORS['historical'], 
            linewidth=3, label='Historical Context', alpha=0.9)
    
    # Plot predictions and ground truth if available
    if prediction is not None:
        ax.plot(pred_time, prediction, color=COLORS['prediction'], 
                linewidth=3, label=f'{model_name} Prediction', alpha=0.9)
        
        if ground_truth is not None:
            ax.plot(pred_time, ground_truth, color=COLORS['ground_truth'], 
                    linewidth=3, label='Ground Truth', alpha=0.9)
    
    # Add vertical line to separate context from prediction
    if pred_len > 0:
        ax.axvline(x=context_time[-1] if timestamps is not None else context_len, 
                   color=COLORS['current_time'], linestyle='--', linewidth=2, 
                   alpha=0.7, label='Prediction Start')
    
    # Formatting
    ax.set_title(f'Prediction Comparison - Epoch {epoch}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Normalized Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    
    # Set y-axis to focus on the data range
    if prediction is not None and ground_truth is not None:
        all_values = np.concatenate([context, prediction, ground_truth])
        y_margin = (np.max(all_values) - np.min(all_values)) * 0.05
        ax.set_ylim(np.min(all_values) - y_margin, np.max(all_values) + y_margin)

def _plot_training_loss(ax, training_history, epoch, verbose):
    """Plot training and validation loss over epochs"""
    
    if training_history is None or not training_history:
        ax.text(0.5, 0.5, 'No training history\navailable', 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, alpha=0.6)
        ax.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        return
    
    train_losses = training_history.get('train_loss', [])
    val_losses = training_history.get('val_loss', [])
    
    if not train_losses and not val_losses:
        ax.text(0.5, 0.5, 'No loss data\navailable', 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, alpha=0.6)
        ax.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        return
    
    # Create epoch axis
    epochs = np.arange(1, len(train_losses) + 1) if train_losses else np.arange(1, len(val_losses) + 1)
    
    # Plot training loss
    if train_losses:
        ax.plot(epochs[:len(train_losses)], train_losses, 'b-', linewidth=2, 
                label='Training Loss', alpha=0.8)
    
    # Plot validation loss
    if val_losses:
        val_epochs = np.arange(1, len(val_losses) + 1)
        ax.plot(val_epochs, val_losses, 'r-', linewidth=2, 
                label='Validation Loss', alpha=0.8)
    
    # Highlight current epoch
    if epoch < len(train_losses):
        current_train_loss = train_losses[epoch]
        ax.plot(epoch + 1, current_train_loss, 'bo', markersize=8, 
                label=f'Current Epoch ({epoch + 1})')
    
    # Formatting
    ax.set_title('Training & Validation Loss vs Epoch', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')  # Log scale for better visualization
    
    # Add text with current values
    if train_losses and val_losses:
        current_train = train_losses[-1] if train_losses else 0
        current_val = val_losses[-1] if val_losses else 0
        best_val = min(val_losses) if val_losses else 0
        
        info_text = f"Current Train: {current_train:.6f}\nCurrent Val: {current_val:.6f}\nBest Val: {best_val:.6f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def _plot_metrics_summary(ax, metrics, model_name):
    """Plot a summary of key metrics"""
    
    ax.axis('off')  # Turn off axes for text display
    
    # Create metrics text
    metrics_text = f"{model_name} Performance Metrics\n" + "="*35 + "\n\n"
    
    if 'mse' in metrics:
        metrics_text += f"MSE: {metrics['mse']:.6f}\n"
    if 'mae' in metrics:
        metrics_text += f"MAE: {metrics['mae']:.6f}\n"
    if 'rmse' in metrics:
        metrics_text += f"RMSE: {metrics['rmse']:.6f}\n"
    if 'directional_accuracy' in metrics:
        metrics_text += f"Directional Accuracy: {metrics['directional_accuracy']:.1%}\n"
    if 'correlation' in metrics:
        metrics_text += f"Correlation: {metrics['correlation']:.4f}\n"
    
    # Add the text to the plot
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))

def _plot_residuals_analysis(ax, plot_data, metrics):
    """Plot residuals analysis"""
    
    prediction = plot_data['prediction']
    ground_truth = plot_data['ground_truth']
    
    if prediction is not None and ground_truth is not None:
        residuals = prediction - ground_truth
        
        # Plot residuals over time
        ax.plot(residuals, color='red', linewidth=2, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axhline(y=np.mean(residuals), color='blue', linestyle='--', alpha=0.7, 
                   label=f'Mean: {np.mean(residuals):.4f}')
        
        ax.set_title('Prediction Residuals', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Residual (Pred - Truth)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No residuals data\navailable', 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, alpha=0.6)

def _plot_correlation_analysis(ax, plot_data, metrics):
    """Plot correlation analysis"""
    
    prediction = plot_data['prediction']
    ground_truth = plot_data['ground_truth']
    
    if prediction is not None and ground_truth is not None:
        # Scatter plot of predictions vs ground truth
        ax.scatter(ground_truth, prediction, alpha=0.6, color=COLORS['prediction'])
        
        # Add perfect prediction line
        min_val = min(np.min(ground_truth), np.min(prediction))
        max_val = max(np.max(ground_truth), np.max(prediction))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, 
                label='Perfect Prediction')
        
        # Add correlation info
        if 'correlation' in metrics:
            ax.text(0.05, 0.95, f"Correlation: {metrics['correlation']:.4f}", 
                    transform=ax.transAxes, fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title('Prediction vs Ground Truth', fontsize=12, fontweight='bold')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No correlation data\navailable', 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, alpha=0.6)
