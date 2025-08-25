#!/usr/bin/env python3
"""
Debug data preprocessing and normalization for TimesFM training.
Visualize how data is being processed and identify any issues.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import pytz

# Configuration
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "databento/ES/glbx-mdp3-20100606-20250822.ohlcv-1m.csv"
CONTEXT_LENGTH = 512
HORIZON_LENGTH = 128
INPUT_PATCH_LEN = 32
SEQUENCE_STRIDE = 64

def load_and_analyze_data():
    """Load data and analyze normalization patterns."""
    print(f"📊 Loading data from {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.dropna(subset=['close'])
    
    print(f"\n📈 Data Overview:")
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Close price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    
    # Extract a sample day for analysis
    sample_date = '2024-01-15'  # Pick a recent trading day
    day_mask = df['timestamp'].dt.date.astype(str) == sample_date
    day_data = df[day_mask].copy()
    
    if len(day_data) == 0:
        print(f"No data for {sample_date}, using most recent day")
        latest_date = df['timestamp'].dt.date.max()
        day_mask = df['timestamp'].dt.date == latest_date
        day_data = df[day_mask].copy()
        sample_date = str(latest_date)
    
    print(f"\n📅 Analyzing day: {sample_date}")
    print(f"Day data points: {len(day_data)}")
    
    return df, day_data, sample_date

def normalize_prices_method1(prices):
    """Method 1: Simple mean/std normalization (current implementation)."""
    mean = np.mean(prices)
    std = np.std(prices) + 1e-6
    return (prices - mean) / std, mean, std

def normalize_prices_method2(prices, base_price=None):
    """Method 2: Normalize to previous day close (like chronos_loop.py)."""
    if base_price is None:
        base_price = prices[0]  # Use first price as base
    return prices / base_price, base_price, 1.0

def normalize_prices_method3(prices):
    """Method 3: Min-max normalization to [0, 1]."""
    min_price = np.min(prices)
    max_price = np.max(prices)
    range_price = max_price - min_price + 1e-6
    return (prices - min_price) / range_price, min_price, range_price

def visualize_normalization_methods(day_data):
    """Compare different normalization methods."""
    prices = day_data['close'].values
    timestamps = day_data['timestamp']
    
    # Apply different normalization methods
    norm1, mean1, std1 = normalize_prices_method1(prices)
    norm2, base2, _ = normalize_prices_method2(prices)
    norm3, min3, range3 = normalize_prices_method3(prices)
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    
    # Original prices
    axes[0].plot(timestamps, prices, 'b-', linewidth=2, label='Original')
    axes[0].set_ylabel('Price ($)')
    axes[0].set_title('Original Prices')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Method 1: Mean/Std
    axes[1].plot(timestamps, norm1, 'g-', linewidth=2, label=f'Mean={mean1:.2f}, Std={std1:.2f}')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Normalized')
    axes[1].set_title('Method 1: Mean/Std Normalization (Current)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Method 2: Base price
    axes[2].plot(timestamps, norm2, 'r-', linewidth=2, label=f'Base=${base2:.2f}')
    axes[2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Base (1.0)')
    axes[2].set_ylabel('Normalized')
    axes[2].set_title('Method 2: Relative to Base Price (chronos_loop style)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Method 3: Min-Max
    axes[3].plot(timestamps, norm3, 'm-', linewidth=2, label=f'Min=${min3:.2f}, Max=${min3+range3:.2f}')
    axes[3].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Normalized')
    axes[3].set_title('Method 3: Min-Max Normalization')
    axes[3].set_xlabel('Time')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ Saved normalization comparison to 'normalization_comparison.png'")

def visualize_sequence_creation(day_data):
    """Visualize how sequences are created for training."""
    prices = day_data['close'].values
    timestamps = day_data['timestamp'].values
    
    # Normalize using method 2 (relative to base)
    normalized_prices, base_price, _ = normalize_prices_method2(prices)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    
    # Plot full day data
    ax1.plot(timestamps, normalized_prices, 'b-', alpha=0.3, linewidth=1, label='Full Day Data')
    
    # Show a few example sequences
    num_examples = 5
    colors = plt.cm.rainbow(np.linspace(0, 1, num_examples))
    
    total_len = CONTEXT_LENGTH + HORIZON_LENGTH
    
    for i in range(num_examples):
        start_idx = i * SEQUENCE_STRIDE * 2  # Space them out
        if start_idx + total_len > len(normalized_prices):
            break
        
        # Context window
        context_end = start_idx + CONTEXT_LENGTH
        ax1.plot(timestamps[start_idx:context_end], 
                normalized_prices[start_idx:context_end],
                color=colors[i], linewidth=2, alpha=0.8,
                label=f'Context {i+1}')
        
        # Prediction window
        pred_end = start_idx + total_len
        ax1.plot(timestamps[context_end:pred_end],
                normalized_prices[context_end:pred_end],
                color=colors[i], linewidth=2, alpha=0.8,
                linestyle='--',
                label=f'Target {i+1}')
        
        # Mark transition point
        ax1.axvline(x=timestamps[context_end], color=colors[i], 
                   linestyle=':', alpha=0.5)
    
    ax1.set_ylabel('Normalized Price')
    ax1.set_title(f'Sequence Creation Example (Context={CONTEXT_LENGTH}, Horizon={HORIZON_LENGTH}, Stride={SEQUENCE_STRIDE})')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Show patch creation for first sequence
    ax2.set_title(f'Patch Creation Example (Patch Length={INPUT_PATCH_LEN})')
    
    # Get first sequence
    first_seq = normalized_prices[:CONTEXT_LENGTH]
    num_patches = CONTEXT_LENGTH // INPUT_PATCH_LEN
    
    # Plot each patch with different colors
    patch_colors = plt.cm.viridis(np.linspace(0, 1, num_patches))
    for i in range(num_patches):
        start = i * INPUT_PATCH_LEN
        end = start + INPUT_PATCH_LEN
        patch_times = np.arange(start, end)
        
        ax2.plot(patch_times, first_seq[start:end], 
                color=patch_colors[i], linewidth=2,
                marker='o', markersize=4,
                label=f'Patch {i+1}')
    
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Normalized Price')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis for top plot
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    
    plt.tight_layout()
    plt.savefig('sequence_creation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved sequence creation visualization to 'sequence_creation.png'")

def analyze_patch_statistics(day_data):
    """Analyze statistics of patches to understand feature engineering."""
    prices = day_data['close'].values
    normalized_prices, _, _ = normalize_prices_method2(prices)
    
    # Create patches from first sequence
    first_seq = normalized_prices[:CONTEXT_LENGTH]
    num_patches = CONTEXT_LENGTH // INPUT_PATCH_LEN
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # Analyze each patch
    all_features = []
    
    for i in range(min(num_patches, 20)):  # Limit to first 20 patches
        start = i * INPUT_PATCH_LEN
        end = start + INPUT_PATCH_LEN
        patch = first_seq[start:end]
        
        # Calculate features (similar to what's in the training script)
        features = {
            'mean': np.mean(patch),
            'std': np.std(patch),
            'change': patch[-1] - patch[0],
            'max': np.max(patch),
            'min': np.min(patch),
            'range': np.max(patch) - np.min(patch)
        }
        all_features.append(features)
    
    # Convert to DataFrame for easy plotting
    features_df = pd.DataFrame(all_features)
    
    # Plot 1: Patch means over time
    axes[0].plot(features_df['mean'], 'b-', marker='o')
    axes[0].set_title('Patch Means')
    axes[0].set_xlabel('Patch Index')
    axes[0].set_ylabel('Mean Value')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Patch standard deviations
    axes[1].plot(features_df['std'], 'r-', marker='o')
    axes[1].set_title('Patch Standard Deviations')
    axes[1].set_xlabel('Patch Index')
    axes[1].set_ylabel('Std Dev')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Patch changes (end - start)
    axes[2].bar(range(len(features_df)), features_df['change'], 
                color=['green' if x > 0 else 'red' for x in features_df['change']])
    axes[2].set_title('Patch Changes (End - Start)')
    axes[2].set_xlabel('Patch Index')
    axes[2].set_ylabel('Change')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 4: Patch ranges
    axes[3].plot(features_df['range'], 'g-', marker='o')
    axes[3].set_title('Patch Ranges (Max - Min)')
    axes[3].set_xlabel('Patch Index')
    axes[3].set_ylabel('Range')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('patch_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved patch statistics to 'patch_statistics.png'")

def compare_with_chronos_approach(day_data):
    """Compare our approach with chronos_loop.py normalization."""
    prices = day_data['close'].values
    timestamps = day_data['timestamp']
    
    # Simulate chronos_loop.py approach
    # They use 'close_norm' which appears to be normalized to previous day's close
    # Let's simulate this
    prev_day_close = prices[0]  # Assume first price is "previous close"
    chronos_norm = prices / prev_day_close
    
    # Our current approach
    our_norm = (prices - np.mean(prices)) / (np.std(prices) + 1e-6)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Original prices
    ax1.plot(timestamps, prices, 'b-', linewidth=2)
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Original Prices')
    ax1.grid(True, alpha=0.3)
    
    # Chronos-style normalization
    ax2.plot(timestamps, chronos_norm, 'g-', linewidth=2, label='Chronos-style')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Base (1.0)')
    ax2.set_ylabel('Normalized')
    ax2.set_title('Chronos-style Normalization (Relative to Base)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Our normalization
    ax3.plot(timestamps, our_norm, 'r-', linewidth=2, label='Our method')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero mean')
    ax3.set_ylabel('Normalized')
    ax3.set_title('Our Normalization (Z-score)')
    ax3.set_xlabel('Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Format x-axis
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    
    plt.tight_layout()
    plt.savefig('chronos_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved chronos comparison to 'chronos_comparison.png'")

def main():
    """Run all analysis."""
    print("🔍 Debugging Data Preprocessing for TimesFM")
    print("=" * 60)
    
    # Load data
    full_data, day_data, sample_date = load_and_analyze_data()
    
    # Run visualizations
    print("\n📊 Creating visualizations...")
    
    print("\n1. Comparing normalization methods...")
    visualize_normalization_methods(day_data)
    
    print("\n2. Visualizing sequence creation...")
    visualize_sequence_creation(day_data)
    
    print("\n3. Analyzing patch statistics...")
    analyze_patch_statistics(day_data)
    
    print("\n4. Comparing with chronos approach...")
    compare_with_chronos_approach(day_data)
    
    print("\n✅ Analysis complete! Check the generated PNG files:")
    print("   - normalization_comparison.png")
    print("   - sequence_creation.png")
    print("   - patch_statistics.png")
    print("   - chronos_comparison.png")
    
    # Print recommendations
    print("\n💡 Recommendations based on chronos_loop.py:")
    print("1. Use relative normalization (price / base_price) instead of z-score")
    print("2. Base price should be previous day's close or first price in sequence")
    print("3. This keeps values around 1.0, which is more interpretable")
    print("4. Consider using the exact same preprocessing as your inference code")

if __name__ == "__main__":
    main()
