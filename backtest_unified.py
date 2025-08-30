#!/usr/bin/env python3
"""
Unified Backtesting Framework for Time Series Prediction Models

This comprehensive script performs systematic backtesting of prediction models with:
1. Running predictions on multiple historical cases
2. Evaluating directional accuracy (up/down classification)
3. Computing comprehensive performance metrics
4. Generating detailed multi-panel visualizations
5. Producing reports and saving results
"""

import os
import sys
import json
import random
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from tqdm import tqdm

# Try to import scipy for KDE, but make it optional
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy not available, some visualizations will be simplified")

# Set matplotlib style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

# Add project paths
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

# Import prediction models
# from prediction_sundial import predict_sundial, get_model_info as get_sundial_info
from prediction_timesfm_v2 import predict_timesfm_v2, get_model_info as get_timesfm_info
from prediction_fft import predict_fft, get_model_info as get_fft_info
# from prediction_chronos import predict_chronos, get_model_info as get_chronos_info

# ==============================================================================
# Configuration
# ==============================================================================

# Backtesting parameters
NUM_TEST_CASES = 2000           # Number of test cases to run
CONTEXT_LENGTH = 416           # Historical context in minutes (~6.9 hours)
HORIZON_LENGTH = 96            # Prediction horizon in minutes (~1.6 hours)
RANDOM_SEED = 42              # For reproducibility

# Dataset configuration
DATASETS_DIR = SCRIPT_DIR / "datasets" / "ES"
RESULTS_DIR = SCRIPT_DIR / "backtest_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Model registry - add new models here
MODEL_REGISTRY = {
    # 'sundial': {
    #     'predict_fn': predict_sundial,
    #     'info_fn': get_sundial_info,
    #     'display_name': 'Sundial Base 128M',
    #     'color': '#FF6B6B'
    # },
    'timesfm_v2': {
        'predict_fn': predict_timesfm_v2,
        'info_fn': get_timesfm_info,
        'display_name': 'TimesFM 2.0',
        'color': '#4ECDC4'
    },
    'fft_similarity': {
        'predict_fn': predict_fft,
        'info_fn': get_fft_info,
        'display_name': 'FFT Similarity Matching',
        'color': '#9B59B6'
    },
}

# ==============================================================================
# Metrics Calculation
# ==============================================================================

def calculate_directional_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate the percentage of correct directional predictions (up/down).
    
    Args:
        predictions: Predicted values
        actuals: Actual values
    
    Returns:
        Directional accuracy as a percentage
    """
    # Calculate differences (returns)
    pred_returns = np.diff(predictions)
    actual_returns = np.diff(actuals)
    
    # Determine directions (1 for up, -1 for down, 0 for no change)
    pred_directions = np.sign(pred_returns)
    actual_directions = np.sign(actual_returns)
    
    # Calculate accuracy (excluding no-change cases)
    non_zero_mask = actual_directions != 0
    if np.sum(non_zero_mask) == 0:
        return 0.0
    
    correct_predictions = pred_directions[non_zero_mask] == actual_directions[non_zero_mask]
    accuracy = np.mean(correct_predictions) * 100
    
    return accuracy

def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict:
    """
    Calculate comprehensive metrics for prediction evaluation.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
    
    Returns:
        Dictionary containing various metrics
    """
    # Basic error metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    
    # Percentage error metrics
    # Avoid division by zero
    mask = actuals != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
    else:
        mape = np.nan
    
    # Directional accuracy
    directional_acc = calculate_directional_accuracy(predictions, actuals)
    
    # Correlation
    if len(predictions) > 1:
        correlation = np.corrcoef(predictions, actuals)[0, 1]
    else:
        correlation = np.nan
    
    # Additional directional metrics
    pred_returns = np.diff(predictions)
    actual_returns = np.diff(actuals)
    
    # True positive rate (correctly predicted ups)
    actual_ups = actual_returns > 0
    pred_ups = pred_returns > 0
    if np.sum(actual_ups) > 0:
        tpr = np.sum(actual_ups & pred_ups) / np.sum(actual_ups)
    else:
        tpr = np.nan
    
    # True negative rate (correctly predicted downs)
    actual_downs = actual_returns < 0
    pred_downs = pred_returns < 0
    if np.sum(actual_downs) > 0:
        tnr = np.sum(actual_downs & pred_downs) / np.sum(actual_downs)
    else:
        tnr = np.nan
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': directional_acc,
        'correlation': correlation,
        'true_positive_rate': tpr,
        'true_negative_rate': tnr,
        'num_predictions': len(predictions)
    }

# ==============================================================================
# Backtesting Engine
# ==============================================================================

class BacktestEngine:
    """Engine for running systematic backtests on prediction models."""
    
    def __init__(self, model_name: str, num_cases: int = NUM_TEST_CASES, 
                 random_seed: Optional[int] = RANDOM_SEED):
        """
        Initialize the backtest engine.
        
        Args:
            model_name: Name of the model to test (from MODEL_REGISTRY)
            num_cases: Number of test cases to run
            random_seed: Random seed for reproducibility
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
        
        self.model_name = model_name
        self.model_config = MODEL_REGISTRY[model_name]
        self.num_cases = num_cases
        self.random_seed = random_seed
        
        # Set random seeds
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Storage for results
        self.test_cases = []
        self.results = []
        
    def select_test_cases(self) -> List[Dict]:
        """
        Select test cases from available datasets.
        
        Returns:
            List of test case configurations
        """
        print(f"🔍 Selecting {self.num_cases} test cases...")
        
        # Find all CSV files
        csv_files = list(DATASETS_DIR.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {DATASETS_DIR}")
        
        print(f"📁 Found {len(csv_files)} dataset files")
        
        test_cases = []
        attempts = 0
        max_attempts = self.num_cases * 10  # Prevent infinite loop
        
        while len(test_cases) < self.num_cases and attempts < max_attempts:
            attempts += 1
            
            # Select random file
            csv_file = random.choice(csv_files)
            
            # Load data
            try:
                df = pd.read_csv(csv_file)
                
                # Need enough data for context + horizon
                min_required = CONTEXT_LENGTH + HORIZON_LENGTH + 1
                if len(df) < min_required:
                    continue
                
                # Select random starting point
                max_start = len(df) - min_required
                start_idx = random.randint(CONTEXT_LENGTH, max_start)
                
                # Create test case
                test_case = {
                    'file': csv_file.name,
                    'start_idx': start_idx,
                    'context_start': start_idx - CONTEXT_LENGTH,
                    'context_end': start_idx,
                    'horizon_start': start_idx,
                    'horizon_end': start_idx + HORIZON_LENGTH,
                    'timestamp': df.iloc[start_idx]['timestamp_pt'] if 'timestamp_pt' in df.columns else None
                }
                
                test_cases.append(test_case)
                
            except Exception as e:
                print(f"⚠️  Failed to process {csv_file.name}: {e}")
                continue
        
        if len(test_cases) < self.num_cases:
            print(f"⚠️  Only found {len(test_cases)} valid test cases")
        
        self.test_cases = test_cases
        return test_cases
    
    def run_single_test(self, test_case: Dict) -> Dict:
        """
        Run a single test case.
        
        Args:
            test_case: Test case configuration
        
        Returns:
            Dictionary containing results
        """
        try:
            # Load data
            df = pd.read_csv(DATASETS_DIR / test_case['file'])
            
            # Extract context and actual future data
            context_data = df['close'].iloc[test_case['context_start']:test_case['context_end']].values
            actual_data = df['close'].iloc[test_case['horizon_start']:test_case['horizon_end']].values
            
            # Run prediction
            predictions, error_code = self.model_config['predict_fn'](context_data, verbose=False)
            
            if error_code != 0:  # Assuming 0 is success
                return {
                    'success': False,
                    'error': f"Prediction failed with code {error_code}",
                    'test_case': test_case
                }
            
            # Calculate metrics
            metrics = calculate_metrics(predictions, actual_data)
            
            # Store results
            result = {
                'success': True,
                'test_case': test_case,
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                'actuals': actual_data.tolist(),
                'metrics': metrics,
                'context_end_value': context_data[-1],
                'actual_start_value': actual_data[0]
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'test_case': test_case
            }
    
    def run_backtest(self):
        """Run the complete backtest with graceful keyboard interrupt handling."""
        print(f"\n🚀 Starting backtest for {self.model_config['display_name']}")
        print(f"   Model: {self.model_name}")
        print(f"   Test cases: {self.num_cases}")
        print(f"   Context length: {CONTEXT_LENGTH} minutes")
        print(f"   Horizon length: {HORIZON_LENGTH} minutes")
        
        # Select test cases if not already done
        if not self.test_cases:
            self.select_test_cases()
        
        # Run tests with progress bar and keyboard interrupt handling
        print(f"\n📊 Running predictions...")
        print(f"   💡 Press Ctrl+C to stop and analyze current progress")
        self.results = []
        
        try:
            for test_case in tqdm(self.test_cases, desc="Processing"):
                result = self.run_single_test(test_case)
                self.results.append(result)
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Keyboard interrupt detected!")
            print(f"   📊 Stopping analysis and processing current results...")
            print(f"   🔢 Completed {len(self.results)} out of {len(self.test_cases)} test cases")
            
            if len(self.results) == 0:
                print(f"   ❌ No test cases completed - cannot generate analysis")
                return
            
            print(f"   ✅ Proceeding with analysis of {len(self.results)} completed cases...")
        
        # Calculate summary statistics
        self.calculate_summary()
        
    def calculate_summary(self):
        """Calculate summary statistics from results."""
        successful_results = [r for r in self.results if r['success']]
        
        print(f"\n📈 Backtest Summary")
        print(f"   Total tests: {len(self.results)}")
        print(f"   Successful: {len(successful_results)}")
        print(f"   Failed: {len(self.results) - len(successful_results)}")
        
        if successful_results:
            # Aggregate metrics
            metrics_list = [r['metrics'] for r in successful_results]
            
            # Calculate mean and std for each metric
            summary_stats = {}
            for metric_name in metrics_list[0].keys():
                values = [m[metric_name] for m in metrics_list if not np.isnan(m[metric_name])]
                if values:
                    summary_stats[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            # Print key metrics
            print(f"\n🎯 Directional Accuracy:")
            if 'directional_accuracy' in summary_stats:
                da = summary_stats['directional_accuracy']
                print(f"   Mean: {da['mean']:.2f}% ± {da['std']:.2f}%")
                print(f"   Range: [{da['min']:.2f}%, {da['max']:.2f}%]")
            
            print(f"\n📊 Error Metrics:")
            for metric in ['mae', 'rmse', 'mape']:
                if metric in summary_stats:
                    s = summary_stats[metric]
                    print(f"   {metric.upper()}: {s['mean']:.6f} ± {s['std']:.6f}")
            
            self.summary_stats = summary_stats
        else:
            print("❌ No successful predictions to summarize")
            self.summary_stats = None
    
    def save_results(self):
        """Save backtest results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = RESULTS_DIR / f"backtest_{self.model_name}_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        save_data = {
            'model_name': self.model_name,
            'model_info': self.model_config['info_fn']() if self.model_config['info_fn'] else None,
            'timestamp': timestamp,
            'num_cases': self.num_cases,
            'random_seed': self.random_seed,
            'results': convert_to_serializable(self.results),
            'summary_stats': convert_to_serializable(self.summary_stats)
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Save summary CSV
        if self.summary_stats:
            summary_df = pd.DataFrame([
                {
                    'metric': metric,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'count': stats['count']
                }
                for metric, stats in self.summary_stats.items()
            ])
            
            summary_file = RESULTS_DIR / f"summary_{self.model_name}_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"📊 Summary saved to: {summary_file}")
    
    def plot_results(self, save_plots: bool = True, comprehensive: bool = True):
        """
        Generate visualization plots for the backtest results.
        
        Args:
            save_plots: Whether to save plots to disk
            comprehensive: Whether to use comprehensive multi-panel visualization
        """
        if not self.summary_stats:
            print("❌ No results to plot")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if comprehensive:
            # Use comprehensive visualization
            plot_file = RESULTS_DIR / f"backtest_comprehensive_{self.model_name}_{timestamp}.png" if save_plots else None
            self.create_comprehensive_plot(save_path=plot_file)
        else:
            # Use original simple visualization
            plot_file = RESULTS_DIR / f"backtest_plot_{self.model_name}_{timestamp}.png" if save_plots else None
            self.create_simple_plot(save_path=plot_file)
    
    def create_simple_plot(self, save_path: Optional[Path] = None):
        """Create the original simple 4-panel visualization."""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Backtest Results - {self.model_config["display_name"]}', fontsize=16)
        
        # Extract data
        successful_results = [r for r in self.results if r['success']]
        dir_accs = [r['metrics']['directional_accuracy'] for r in successful_results]
        
        # 1. Directional Accuracy Distribution
        ax = axes[0, 0]
        ax.hist(dir_accs, bins=20, alpha=0.7, color=self.model_config['color'], edgecolor='black')
        ax.axvline(np.mean(dir_accs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dir_accs):.2f}%')
        ax.axvline(50, color='black', linestyle=':', linewidth=2, label='Random: 50%')
        ax.set_xlabel('Directional Accuracy (%)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Directional Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Error Metrics Box Plot
        ax = axes[0, 1]
        error_data = []
        error_labels = []
        
        for metric in ['mae', 'rmse', 'mape']:
            if metric in self.summary_stats:
                values = [r['metrics'][metric] for r in successful_results if not np.isnan(r['metrics'][metric])]
                if values:
                    error_data.append(values)
                    error_labels.append(metric.upper())
        
        if error_data:
            bp = ax.boxplot(error_data, labels=error_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(self.model_config['color'])
                patch.set_alpha(0.7)
            ax.set_ylabel('Error Value')
            ax.set_title('Error Metrics Distribution')
            ax.grid(True, alpha=0.3)
        
        # 3. Correlation Scatter Plot
        ax = axes[1, 0]
        correlations = [r['metrics']['correlation'] for r in successful_results if not np.isnan(r['metrics']['correlation'])]
        dir_accs_for_corr = [r['metrics']['directional_accuracy'] for r in successful_results if not np.isnan(r['metrics']['correlation'])]
        
        if correlations and dir_accs_for_corr:
            ax.scatter(correlations, dir_accs_for_corr, alpha=0.6, color=self.model_config['color'], edgecolor='black')
            ax.set_xlabel('Correlation')
            ax.set_ylabel('Directional Accuracy (%)')
            ax.set_title('Correlation vs Directional Accuracy')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(correlations, dir_accs_for_corr, 1)
            p = np.poly1d(z)
            ax.plot(sorted(correlations), p(sorted(correlations)), "r--", alpha=0.8)
        
        # 4. Cumulative Accuracy Over Time
        ax = axes[1, 1]
        cumulative_correct = []
        total_predictions = 0
        
        for i, result in enumerate(successful_results):
            if result['success']:
                # Calculate directional accuracy for this result
                acc = result['metrics']['directional_accuracy']
                total_predictions += 1
                
                if i == 0:
                    cumulative_correct.append(acc)
                else:
                    # Update cumulative average
                    prev_sum = cumulative_correct[-1] * (i)
                    new_sum = prev_sum + acc
                    cumulative_correct.append(new_sum / (i + 1))
        
        if cumulative_correct:
            ax.plot(range(1, len(cumulative_correct) + 1), cumulative_correct, 
                   color=self.model_config['color'], linewidth=2)
            ax.axhline(y=50, color='black', linestyle=':', linewidth=2, label='Random baseline')
            ax.set_xlabel('Test Case Number')
            ax.set_ylabel('Cumulative Directional Accuracy (%)')
            ax.set_title('Cumulative Performance Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")
        
        plt.show()
    
    def create_comprehensive_plot(self, save_path: Optional[Path] = None):
        """Create comprehensive multi-panel visualization of backtest results."""
        # Filter successful results
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            print("❌ No successful results to plot")
            return
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(24, 20))
        
        # Create grid specification
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.25)
        
        # Title
        fig.suptitle(f'Comprehensive Backtest Analysis - {self.model_config["display_name"]}', 
                     fontsize=20, fontweight='bold')
        
        # Extract data for plotting
        dir_accs = [r['metrics']['directional_accuracy'] for r in successful_results]
        maes = [r['metrics']['mae'] for r in successful_results if not np.isnan(r['metrics']['mae'])]
        rmses = [r['metrics']['rmse'] for r in successful_results if not np.isnan(r['metrics']['rmse'])]
        mapes = [r['metrics']['mape'] for r in successful_results if not np.isnan(r['metrics']['mape'])]
        correlations = [r['metrics']['correlation'] for r in successful_results if not np.isnan(r['metrics']['correlation'])]
        tprs = [r['metrics']['true_positive_rate'] for r in successful_results if not np.isnan(r['metrics']['true_positive_rate'])]
        tnrs = [r['metrics']['true_negative_rate'] for r in successful_results if not np.isnan(r['metrics']['true_negative_rate'])]
        
        # ==============================================================================
        # 1. Directional Accuracy Distribution with Statistics
        # ==============================================================================
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Create histogram
        n, bins, patches = ax1.hist(dir_accs, bins=25, alpha=0.7, color=self.model_config['color'], 
                                   edgecolor='black', density=True)
        
        # Add KDE curve if scipy available
        if HAS_SCIPY:
            kde = stats.gaussian_kde(dir_accs)
            x_range = np.linspace(min(dir_accs), max(dir_accs), 100)
            ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Add statistical lines
        mean_acc = np.mean(dir_accs)
        median_acc = np.median(dir_accs)
        ax1.axvline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.2f}%')
        ax1.axvline(median_acc, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_acc:.2f}%')
        ax1.axvline(50, color='black', linestyle=':', linewidth=2, label='Random: 50%')
        
        # Shade regions
        ax1.axvspan(0, 50, alpha=0.1, color='red', label='Below Random')
        ax1.axvspan(50, 100, alpha=0.1, color='green', label='Above Random')
        
        ax1.set_xlabel('Directional Accuracy (%)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Distribution of Directional Accuracy', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'n = {len(dir_accs)}\nμ = {mean_acc:.2f}%\nσ = {np.std(dir_accs):.2f}%\nmin = {min(dir_accs):.2f}%\nmax = {max(dir_accs):.2f}%'
        ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ==============================================================================
        # 2. Confusion Matrix for Directional Predictions
        # ==============================================================================
        ax2 = fig.add_subplot(gs[0, 2:])
        
        # Calculate confusion matrix components
        total_ups_predicted = 0
        total_downs_predicted = 0
        total_actual_ups = 0
        total_actual_downs = 0
        correct_ups = 0
        correct_downs = 0
        
        for result in successful_results:
            predictions = np.array(result['predictions'])
            actuals = np.array(result['actuals'])
            
            pred_returns = np.diff(predictions)
            actual_returns = np.diff(actuals)
            
            pred_ups = pred_returns > 0
            pred_downs = pred_returns < 0
            actual_ups = actual_returns > 0
            actual_downs = actual_returns < 0
            
            total_ups_predicted += np.sum(pred_ups)
            total_downs_predicted += np.sum(pred_downs)
            total_actual_ups += np.sum(actual_ups)
            total_actual_downs += np.sum(actual_downs)
            correct_ups += np.sum(pred_ups & actual_ups)
            correct_downs += np.sum(pred_downs & actual_downs)
        
        # Create confusion matrix
        confusion_matrix = np.array([[correct_downs, total_actual_downs - correct_downs],
                                    [total_actual_ups - correct_ups, correct_ups]])
        
        # Plot confusion matrix
        im = ax2.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Count', fontsize=10)
        
        # Set ticks and labels
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['Predicted Down', 'Predicted Up'])
        ax2.set_yticklabels(['Actual Down', 'Actual Up'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax2.text(j, i, f'{confusion_matrix[i, j]:,}', 
                              ha="center", va="center", color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black",
                              fontsize=12, fontweight='bold')
        
        # Add percentages
        total = confusion_matrix.sum()
        for i in range(2):
            for j in range(2):
                pct = confusion_matrix[i, j] / total * 100
                text = ax2.text(j, i + 0.3, f'({pct:.1f}%)', 
                              ha="center", va="center", color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black",
                              fontsize=10)
        
        ax2.set_title('Direction Prediction Confusion Matrix', fontsize=14, fontweight='bold')
        
        # ==============================================================================
        # 3. Error Metrics Comparison
        # ==============================================================================
        ax3 = fig.add_subplot(gs[1, :2])
        
        # Prepare data for box plots
        error_data = []
        error_labels = []
        
        if maes:
            error_data.append(maes)
            error_labels.append(f'MAE\n(μ={np.mean(maes):.4f})')
        if rmses:
            error_data.append(rmses)
            error_labels.append(f'RMSE\n(μ={np.mean(rmses):.4f})')
        if mapes and len([m for m in mapes if m < 10]) > 10:  # Filter extreme MAPE values
            filtered_mapes = [m for m in mapes if m < 10]
            error_data.append(filtered_mapes)
            error_labels.append(f'MAPE %\n(μ={np.mean(filtered_mapes):.2f})')
        
        if error_data:
            bp = ax3.boxplot(error_data, labels=error_labels, patch_artist=True, notch=True, showmeans=True)
            
            # Customize box plots
            for patch in bp['boxes']:
                patch.set_facecolor(self.model_config['color'])
                patch.set_alpha(0.7)
            
            # Customize other elements
            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                if element in bp:
                    plt.setp(bp[element], color='black')
            
            ax3.set_ylabel('Error Value', fontsize=12)
            ax3.set_title('Error Metrics Distribution', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # ==============================================================================
        # 4. Correlation Analysis
        # ==============================================================================
        ax4 = fig.add_subplot(gs[1, 2:])
        
        if correlations and dir_accs:
            # Create 2D histogram
            h = ax4.hist2d(correlations, dir_accs[:len(correlations)], bins=20, cmap='YlOrRd')
            
            # Add colorbar
            cbar = plt.colorbar(h[3], ax=ax4)
            cbar.set_label('Count', fontsize=10)
            
            # Add regression line
            z = np.polyfit(correlations, dir_accs[:len(correlations)], 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(correlations), max(correlations), 100)
            ax4.plot(x_line, p(x_line), "b--", linewidth=2, label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
            
            # Add reference lines
            ax4.axhline(50, color='black', linestyle=':', linewidth=1, alpha=0.5)
            ax4.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
            
            ax4.set_xlabel('Correlation', fontsize=12)
            ax4.set_ylabel('Directional Accuracy (%)', fontsize=12)
            ax4.set_title('Correlation vs Directional Accuracy', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # ==============================================================================
        # 5. Cumulative Performance Over Time
        # ==============================================================================
        ax5 = fig.add_subplot(gs[2, :])
        
        # Calculate cumulative metrics
        cumulative_accuracy = []
        cumulative_mae = []
        rolling_accuracy = []
        window_size = 10
        
        for i in range(len(successful_results)):
            # Cumulative accuracy
            accs_so_far = [r['metrics']['directional_accuracy'] for r in successful_results[:i+1]]
            cumulative_accuracy.append(np.mean(accs_so_far))
            
            # Cumulative MAE
            maes_so_far = [r['metrics']['mae'] for r in successful_results[:i+1] if not np.isnan(r['metrics']['mae'])]
            if maes_so_far:
                cumulative_mae.append(np.mean(maes_so_far))
            
            # Rolling accuracy
            if i >= window_size - 1:
                window_accs = [r['metrics']['directional_accuracy'] for r in successful_results[i-window_size+1:i+1]]
                rolling_accuracy.append(np.mean(window_accs))
        
        # Plot on primary axis
        x_range = range(1, len(cumulative_accuracy) + 1)
        ax5.plot(x_range, cumulative_accuracy, 'b-', linewidth=2, label='Cumulative Accuracy')
        
        if len(rolling_accuracy) > 0:
            x_rolling = range(window_size, len(cumulative_accuracy) + 1)
            ax5.plot(x_rolling, rolling_accuracy, 'g-', linewidth=1.5, alpha=0.7, 
                    label=f'Rolling Accuracy (window={window_size})')
        
        ax5.axhline(50, color='black', linestyle=':', linewidth=2, label='Random Baseline')
        ax5.fill_between(x_range, 50, cumulative_accuracy, where=[acc > 50 for acc in cumulative_accuracy],
                        color='green', alpha=0.2, label='Above Random')
        ax5.fill_between(x_range, 50, cumulative_accuracy, where=[acc <= 50 for acc in cumulative_accuracy],
                        color='red', alpha=0.2, label='Below Random')
        
        ax5.set_xlabel('Test Case Number', fontsize=12)
        ax5.set_ylabel('Directional Accuracy (%)', fontsize=12)
        ax5.set_title('Cumulative and Rolling Performance', fontsize=14, fontweight='bold')
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        
        # Add secondary axis for MAE
        if cumulative_mae:
            ax5_2 = ax5.twinx()
            ax5_2.plot(range(1, len(cumulative_mae) + 1), cumulative_mae, 'r--', 
                      linewidth=1.5, alpha=0.7, label='Cumulative MAE')
            ax5_2.set_ylabel('MAE', fontsize=12, color='red')
            ax5_2.tick_params(axis='y', labelcolor='red')
        
        # ==============================================================================
        # 6. Prediction Horizon Analysis
        # ==============================================================================
        ax6 = fig.add_subplot(gs[3, :2])
        
        # Calculate accuracy at different horizons
        horizon_accuracies = []
        horizons = [1, 5, 10, 20, 30, 60, 96]  # Minutes into the future
        
        for h in horizons:
            if h <= 96:  # Our prediction horizon
                h_accs = []
                for result in successful_results:
                    predictions = np.array(result['predictions'])
                    actuals = np.array(result['actuals'])
                    
                    if len(predictions) >= h and len(actuals) >= h:
                        # Calculate direction at horizon h
                        pred_dir = np.sign(predictions[h-1] - predictions[0]) if h > 1 else 0
                        actual_dir = np.sign(actuals[h-1] - actuals[0]) if h > 1 else 0
                        
                        if actual_dir != 0:  # Exclude no-change cases
                            h_accs.append(1 if pred_dir == actual_dir else 0)
                
                if h_accs:
                    horizon_accuracies.append(np.mean(h_accs) * 100)
                else:
                    horizon_accuracies.append(50)  # Default to random
        
        # Plot horizon analysis
        ax6.plot(horizons[:len(horizon_accuracies)], horizon_accuracies, 'o-', 
                color=self.model_config['color'], linewidth=2, markersize=8)
        ax6.axhline(50, color='black', linestyle=':', linewidth=2)
        ax6.fill_between(horizons[:len(horizon_accuracies)], 50, horizon_accuracies,
                        where=[acc > 50 for acc in horizon_accuracies], 
                        color='green', alpha=0.2)
        ax6.fill_between(horizons[:len(horizon_accuracies)], 50, horizon_accuracies,
                        where=[acc <= 50 for acc in horizon_accuracies], 
                        color='red', alpha=0.2)
        
        ax6.set_xlabel('Prediction Horizon (minutes)', fontsize=12)
        ax6.set_ylabel('Directional Accuracy (%)', fontsize=12)
        ax6.set_title('Accuracy vs Prediction Horizon', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.set_xticks(horizons[:len(horizon_accuracies)])
        
        # ==============================================================================
        # 7. True Positive vs True Negative Rate
        # ==============================================================================
        ax7 = fig.add_subplot(gs[3, 2:])
        
        if tprs and tnrs:
            # Create scatter plot
            ax7.scatter(tnrs, tprs, alpha=0.6, s=50, color=self.model_config['color'], edgecolor='black')
            
            # Add diagonal line (perfect balance)
            ax7.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect Balance')
            
            # Add mean point
            mean_tnr = np.mean(tnrs)
            mean_tpr = np.mean(tprs)
            ax7.scatter(mean_tnr, mean_tpr, s=200, color='red', marker='*', 
                       edgecolor='black', linewidth=2, label=f'Mean (TNR={mean_tnr:.2f}, TPR={mean_tpr:.2f})')
            
            # Add quadrant labels
            ax7.text(0.25, 0.75, 'Good at\nDowns', ha='center', va='center', fontsize=10, alpha=0.5)
            ax7.text(0.75, 0.25, 'Good at\nUps', ha='center', va='center', fontsize=10, alpha=0.5)
            ax7.text(0.75, 0.75, 'Good at\nBoth', ha='center', va='center', fontsize=10, alpha=0.5)
            ax7.text(0.25, 0.25, 'Poor at\nBoth', ha='center', va='center', fontsize=10, alpha=0.5)
            
            ax7.set_xlabel('True Negative Rate', fontsize=12)
            ax7.set_ylabel('True Positive Rate', fontsize=12)
            ax7.set_title('Classification Performance Balance', fontsize=14, fontweight='bold')
            ax7.set_xlim(0, 1)
            ax7.set_ylim(0, 1)
            ax7.grid(True, alpha=0.3)
            ax7.legend()
        
        # Sample prediction subplot removed as requested
        
        # Add performance summary text
        summary_text = f'Model: {self.model_config["display_name"]}\n'
        summary_text += f'Total Cases: {len(self.results)}\n'
        summary_text += f'Successful: {len(successful_results)}\n'
        if self.summary_stats and 'directional_accuracy' in self.summary_stats:
            da = self.summary_stats['directional_accuracy']
            summary_text += f'Dir. Accuracy: {da["mean"]:.2f}% ± {da["std"]:.2f}%'
        
        fig.text(0.99, 0.01, summary_text, transform=fig.transFigure, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comprehensive plot saved to: {save_path}")
        
        plt.show()

# ==============================================================================
# Model Comparison Functions
# ==============================================================================

def compare_models(model_names: List[str], num_cases: int = NUM_TEST_CASES):
    """
    Compare multiple models on the same test cases.
    
    Args:
        model_names: List of model names to compare
        num_cases: Number of test cases to run
    """
    print(f"🔄 Comparing models: {model_names}")
    
    # First, select common test cases
    engine = BacktestEngine(model_names[0], num_cases)
    test_cases = engine.select_test_cases()
    
    # Run each model on the same test cases
    results = {}
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")
        
        engine = BacktestEngine(model_name, num_cases)
        engine.test_cases = test_cases  # Use same test cases
        engine.run_backtest()
        engine.save_results()
        
        results[model_name] = {
            'engine': engine,
            'summary': engine.summary_stats
        }
    
    # Create comparison visualization
    create_model_comparison_plot(results)
    
    return results

def create_model_comparison_plot(results_dict: Dict[str, Dict], save_path: Optional[Path] = None):
    """
    Create a comprehensive comparison plot for multiple models.
    
    Args:
        results_dict: Dictionary with model names as keys and result data as values
        save_path: Optional path to save the plot
    """
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Multi-Model Backtest Comparison', fontsize=20, fontweight='bold')
    
    # Prepare data for each model
    model_stats = {}
    for model_name, data in results_dict.items():
        if data['summary'] and 'directional_accuracy' in data['summary']:
            model_stats[model_name] = {
                'display_name': MODEL_REGISTRY[model_name]['display_name'],
                'color': MODEL_REGISTRY[model_name]['color'],
                'dir_acc_mean': data['summary']['directional_accuracy']['mean'],
                'dir_acc_std': data['summary']['directional_accuracy']['std'],
                'mae_mean': data['summary'].get('mae', {}).get('mean', np.nan),
                'mae_std': data['summary'].get('mae', {}).get('std', np.nan),
                'tpr_mean': data['summary'].get('true_positive_rate', {}).get('mean', np.nan),
                'tnr_mean': data['summary'].get('true_negative_rate', {}).get('mean', np.nan),
                'correlation_mean': data['summary'].get('correlation', {}).get('mean', np.nan),
                'successful_results': [r for r in data['engine'].results if r['success']]
            }
    
    # 1. Directional Accuracy Comparison
    ax = axes[0, 0]
    model_names = list(model_stats.keys())
    x_pos = np.arange(len(model_names))
    
    means = [model_stats[m]['dir_acc_mean'] for m in model_names]
    stds = [model_stats[m]['dir_acc_std'] for m in model_names]
    colors = [model_stats[m]['color'] for m in model_names]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7)
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.axhline(50, color='black', linestyle=':', linewidth=2, label='Random baseline')
    ax.set_ylabel('Directional Accuracy (%)')
    ax.set_title('Directional Accuracy Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model_stats[m]['display_name'] for m in model_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Error Metrics Comparison
    ax = axes[0, 1]
    mae_means = [model_stats[m]['mae_mean'] for m in model_names if not np.isnan(model_stats[m]['mae_mean'])]
    mae_stds = [model_stats[m]['mae_std'] for m in model_names if not np.isnan(model_stats[m]['mae_std'])]
    mae_names = [m for m in model_names if not np.isnan(model_stats[m]['mae_mean'])]
    
    if mae_means:
        x_pos = np.arange(len(mae_names))
        bars = ax.bar(x_pos, mae_means, yerr=mae_stds, capsize=10, alpha=0.7)
        for bar, name in zip(bars, mae_names):
            bar.set_color(model_stats[name]['color'])
        
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('MAE Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([model_stats[m]['display_name'] for m in mae_names], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    # 3. TPR vs TNR Scatter
    ax = axes[1, 0]
    for model_name, stats in model_stats.items():
        if not np.isnan(stats['tpr_mean']) and not np.isnan(stats['tnr_mean']):
            ax.scatter(stats['tnr_mean'], stats['tpr_mean'], 
                      s=200, color=stats['color'], label=stats['display_name'],
                      edgecolor='black', linewidth=2, alpha=0.7)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('True Negative Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Classification Balance Comparison')
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.3, 0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Distribution Comparison (Violin plot)
    ax = axes[1, 1]
    all_accuracies = []
    labels = []
    
    for model_name, stats in model_stats.items():
        if stats['successful_results']:
            accs = [r['metrics']['directional_accuracy'] for r in stats['successful_results']]
            all_accuracies.append(accs)
            labels.append(stats['display_name'])
    
    if all_accuracies:
        parts = ax.violinplot(all_accuracies, positions=range(len(all_accuracies)), 
                             showmeans=True, showmedians=True)
        
        # Color the violins
        for pc, (model_name, stats) in zip(parts['bodies'], model_stats.items()):
            pc.set_facecolor(stats['color'])
            pc.set_alpha(0.7)
        
        ax.axhline(50, color='black', linestyle=':', linewidth=2)
        ax.set_ylabel('Directional Accuracy (%)')
        ax.set_title('Accuracy Distribution Comparison')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    # 5. Cumulative Performance Comparison
    ax = axes[2, 0]
    for model_name, stats in model_stats.items():
        results = stats['successful_results']
        if results:
            cumulative = []
            for i in range(len(results)):
                accs = [r['metrics']['directional_accuracy'] for r in results[:i+1]]
                cumulative.append(np.mean(accs))
            
            ax.plot(range(1, len(cumulative) + 1), cumulative, 
                   linewidth=2, label=stats['display_name'], color=stats['color'])
    
    ax.axhline(50, color='black', linestyle=':', linewidth=2)
    ax.set_xlabel('Test Case Number')
    ax.set_ylabel('Cumulative Directional Accuracy (%)')
    ax.set_title('Cumulative Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Performance Summary Table
    ax = axes[2, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Model', 'Dir. Acc. (%)', 'MAE', 'TPR', 'TNR']
    
    for model_name, stats in model_stats.items():
        row = [
            stats['display_name'],
            f"{stats['dir_acc_mean']:.2f} ± {stats['dir_acc_std']:.2f}",
            f"{stats['mae_mean']:.4f}" if not np.isnan(stats['mae_mean']) else 'N/A',
            f"{stats['tpr_mean']:.2f}" if not np.isnan(stats['tpr_mean']) else 'N/A',
            f"{stats['tnr_mean']:.2f}" if not np.isnan(stats['tnr_mean']) else 'N/A'
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color code the cells
    for i, (model_name, stats) in enumerate(model_stats.items()):
        table[(i+1, 0)].set_facecolor(stats['color'])
        table[(i+1, 0)].set_alpha(0.3)
    
    ax.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved to: {save_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = RESULTS_DIR / f"comparison_plot_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved to: {save_path}")
    
    plt.show()

# ==============================================================================
# Main Execution Functions
# ==============================================================================

def run_backtest(model_name: str = 'sundial', num_cases: int = NUM_TEST_CASES, 
                 plot_results: bool = True, save_results: bool = True,
                 comprehensive_plots: bool = True):
    """
    Run a complete backtest for a specified model with graceful interrupt handling.
    
    Args:
        model_name: Name of the model to test
        num_cases: Number of test cases to run
        plot_results: Whether to generate plots
        save_results: Whether to save results to disk
        comprehensive_plots: Whether to use comprehensive visualization
    
    Returns:
        BacktestEngine instance with results
    """
    # Create backtest engine
    engine = BacktestEngine(model_name, num_cases)
    
    try:
        # Run backtest
        engine.run_backtest()
        
        # Save results if we have any
        if save_results and len(engine.results) > 0:
            engine.save_results()
        
        # Generate plots if we have any successful results
        if plot_results and len([r for r in engine.results if r['success']]) > 0:
            engine.plot_results(save_plots=save_results, comprehensive=comprehensive_plots)
            
    except KeyboardInterrupt:
        # This shouldn't happen since we handle it in run_backtest, but just in case
        print(f"\n⚠️  Interrupt detected at top level - exiting gracefully...")
        
    return engine

# ==============================================================================
# CLI Interface
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified backtest framework for time series prediction models')
    parser.add_argument('--model', type=str, default='timesfm_v2',
                      choices=list(MODEL_REGISTRY.keys()),
                      help='Model to test')
    parser.add_argument('--cases', type=int, default=NUM_TEST_CASES,
                      help='Number of test cases')
    parser.add_argument('--compare', nargs='+', 
                      help='Compare multiple models')
    parser.add_argument('--no-plots', action='store_true',
                      help='Skip generating plots')
    parser.add_argument('--simple-plots', action='store_true',
                      help='Use simple plots instead of comprehensive')
    parser.add_argument('--no-save', action='store_true',
                      help='Skip saving results')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("🎯 Unified Backtest Framework for Time Series Predictions")
    print("=" * 60)
    
    try:
        if args.compare:
            # Compare multiple models
            compare_models(args.compare, args.cases)
        else:
            # Run single model backtest
            run_backtest(
                model_name=args.model,
                num_cases=args.cases,
                plot_results=not args.no_plots,
                save_results=not args.no_save,
                comprehensive_plots=not args.simple_plots
            )
        
        print("\n✅ Backtest complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Final keyboard interrupt - exiting...")
        print("✅ Results from completed test cases have been processed and saved.")
        sys.exit(0)
