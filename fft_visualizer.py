#!/usr/bin/env python3
"""
FFT Signature Visualizer - Comprehensive analysis and visualization of FFT signatures

This script randomly selects FFT signatures from the database and creates detailed
visualizations showing:
1. Time series data (context + outcome vs prediction horizon)
2. FFT magnitude spectrum with frequency band highlights
3. FFT phase spectrum 
4. Dominant frequency analysis
5. Spectral energy distribution
6. Pattern characteristics and statistics

Helps validate the quality and understand the characteristics of our FFT database.
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
# Try to import seaborn for better styling (optional)
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
if HAS_SEABORN:
    sns.set_palette("husl")

# Add project paths
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

# ==============================================================================
# Configuration
# ==============================================================================

# Paths
FFT_DIR = SCRIPT_DIR / "datasets" / "fft"
SIGNATURES_DIR = FFT_DIR / "signatures"
METADATA_DIR = FFT_DIR / "metadata"
PLOTS_DIR = SCRIPT_DIR / "fft_visualization_plots"
PLOTS_DIR.mkdir(exist_ok=True)

# FFT Analysis parameters (matching database builder)
CONTEXT_LENGTH = 416
HORIZON_LENGTH = 96
SAMPLING_RATE = 1.0  # 1 sample per minute
NYQUIST_FREQ = SAMPLING_RATE / 2

# Frequency bands (from database)
FREQ_BANDS = {
    'high': (1/5, 1/1),      # 1-5 minute patterns (0.2 to 1.0 cycles/min)
    'medium': (1/30, 1/15),  # 15-30 minute patterns (0.033 to 0.067 cycles/min) 
    'low': (1/90, 1/60)      # 60-90 minute patterns (0.011 to 0.017 cycles/min)
}

# Visualization colors
COLORS = {
    'context': '#2E86AB',
    'outcome': '#A23B72', 
    'prediction': '#F18F01',
    'high_freq': '#C73E1D',
    'medium_freq': '#F18F01',
    'low_freq': '#2E86AB',
    'magnitude': '#4A90E2',
    'phase': '#7B68EE'
}

# ==============================================================================
# Data Loading Functions
# ==============================================================================

def load_database_metadata() -> Dict:
    """Load database metadata"""
    metadata_path = METADATA_DIR / "database_info.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Database metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)

def get_random_signature_files(num_files: int = 5) -> List[Path]:
    """Get random signature files from the database"""
    signature_files = list(SIGNATURES_DIR.glob("*.csv"))
    
    if not signature_files:
        raise FileNotFoundError(f"No signature files found in {SIGNATURES_DIR}")
    
    if len(signature_files) < num_files:
        print(f"⚠️ Only {len(signature_files)} files available, selecting all")
        return signature_files
    
    return random.sample(signature_files, num_files)

def load_signature(signature_path: Path) -> Dict:
    """Load and parse a signature file"""
    try:
        df = pd.read_csv(signature_path)
        
        if len(df) != 1:
            raise ValueError(f"Expected 1 row, got {len(df)}")
        
        row = df.iloc[0]
        
        # Parse arrays from string representation
        context_data = np.array(eval(row['context_data']))
        outcome_data = np.array(eval(row['outcome_data']))
        fft_magnitude = np.array(eval(row['fft_magnitude']))
        fft_phase = np.array(eval(row['fft_phase']))
        
        # Parse dominant frequencies
        dom_freq_high = eval(row['dominant_freq_high']) if pd.notna(row['dominant_freq_high']) else []
        dom_freq_medium = eval(row['dominant_freq_medium']) if pd.notna(row['dominant_freq_medium']) else []
        dom_freq_low = eval(row['dominant_freq_low']) if pd.notna(row['dominant_freq_low']) else []
        
        signature = {
            'metadata': {
                'sequence_id': row['sequence_id'],
                'source_file': row['source_file'],
                'start_idx': row['start_idx'],
                'end_idx': row['end_idx'],
                'timestamp_start': row['timestamp_start'],
                'timestamp_end': row['timestamp_end'],
                'priority_score': row['priority_score'],
                'processing_timestamp': row['processing_timestamp']
            },
            'data': {
                'context_data': context_data,
                'outcome_data': outcome_data,
                'fft_magnitude': fft_magnitude,
                'fft_phase': fft_phase
            },
            'analysis': {
                'dominant_freq_high': dom_freq_high,
                'dominant_freq_medium': dom_freq_medium,
                'dominant_freq_low': dom_freq_low,
                'energy_high': row['energy_high'],
                'energy_medium': row['energy_medium'],
                'energy_low': row['energy_low'],
                'total_energy': row['total_energy']
            }
        }
        
        return signature
        
    except Exception as e:
        raise ValueError(f"Failed to load signature {signature_path.name}: {e}")

# ==============================================================================
# Analysis Functions
# ==============================================================================

def get_frequency_bins(n_samples: int) -> np.ndarray:
    """Get frequency bins for FFT analysis"""
    return np.fft.fftfreq(n_samples, d=1.0)[:n_samples//2]

def analyze_signature_characteristics(signature: Dict) -> Dict:
    """Analyze key characteristics of the signature"""
    context_data = signature['data']['context_data']
    outcome_data = signature['data']['outcome_data']
    fft_magnitude = signature['data']['fft_magnitude']
    
    # Time series statistics
    context_stats = {
        'mean': np.mean(context_data),
        'std': np.std(context_data),
        'min': np.min(context_data),
        'max': np.max(context_data),
        'range': np.max(context_data) - np.min(context_data),
        'trend': np.polyfit(range(len(context_data)), context_data, 1)[0]  # Linear trend slope
    }
    
    outcome_stats = {
        'mean': np.mean(outcome_data),
        'std': np.std(outcome_data),
        'min': np.min(outcome_data),
        'max': np.max(outcome_data),
        'range': np.max(outcome_data) - np.min(outcome_data),
        'trend': np.polyfit(range(len(outcome_data)), outcome_data, 1)[0]
    }
    
    # Price movement analysis
    context_returns = np.diff(context_data)
    outcome_returns = np.diff(outcome_data)
    
    movement_stats = {
        'context_volatility': np.std(context_returns),
        'outcome_volatility': np.std(outcome_returns),
        'context_total_return': (context_data[-1] - context_data[0]) / context_data[0] * 100,
        'outcome_total_return': (outcome_data[-1] - outcome_data[0]) / outcome_data[0] * 100,
        'direction_change': np.sign(context_data[-1] - context_data[0]) != np.sign(outcome_data[-1] - outcome_data[0])
    }
    
    # FFT characteristics
    freq_bins = get_frequency_bins(len(fft_magnitude))
    spectral_stats = {
        'peak_frequency': freq_bins[np.argmax(fft_magnitude[:len(freq_bins)])],
        'spectral_centroid': np.sum(freq_bins * fft_magnitude[:len(freq_bins)]) / np.sum(fft_magnitude[:len(freq_bins)]),
        'spectral_rolloff': calculate_spectral_rolloff(fft_magnitude[:len(freq_bins)], freq_bins),
        'spectral_flatness': calculate_spectral_flatness(fft_magnitude[:len(freq_bins)])
    }
    
    return {
        'context_stats': context_stats,
        'outcome_stats': outcome_stats,
        'movement_stats': movement_stats,
        'spectral_stats': spectral_stats
    }

def calculate_spectral_rolloff(magnitude: np.ndarray, freq_bins: np.ndarray, rolloff_percent: float = 0.85) -> float:
    """Calculate spectral rolloff frequency"""
    total_energy = np.sum(magnitude ** 2)
    cumulative_energy = np.cumsum(magnitude ** 2)
    rolloff_idx = np.where(cumulative_energy >= rolloff_percent * total_energy)[0]
    
    if len(rolloff_idx) > 0:
        return freq_bins[rolloff_idx[0]]
    else:
        return freq_bins[-1]

def calculate_spectral_flatness(magnitude: np.ndarray) -> float:
    """Calculate spectral flatness (Wiener entropy)"""
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    magnitude = magnitude + epsilon
    
    geometric_mean = np.exp(np.mean(np.log(magnitude)))
    arithmetic_mean = np.mean(magnitude)
    
    return geometric_mean / arithmetic_mean

# ==============================================================================
# Visualization Functions
# ==============================================================================

def create_comprehensive_visualization(signature: Dict, characteristics: Dict, save_path: Optional[Path] = None):
    """Create comprehensive visualization of FFT signature"""
    
    # Extract data
    context_data = signature['data']['context_data']
    outcome_data = signature['data']['outcome_data']
    fft_magnitude = signature['data']['fft_magnitude']
    fft_phase = signature['data']['fft_phase']
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title with metadata
    metadata = signature['metadata']
    title = f"FFT Signature Analysis: {metadata['sequence_id']}\n"
    title += f"Source: {metadata['source_file']} | Priority: {metadata['priority_score']:.2f}"
    if metadata['timestamp_start']:
        start_time = pd.to_datetime(metadata['timestamp_start']).strftime('%Y-%m-%d %H:%M')
        title += f" | Time: {start_time}"
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # ==============================================================================
    # 1. Time Series Data (Context + Outcome)
    # ==============================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot context data
    context_x = np.arange(len(context_data))
    outcome_x = np.arange(len(context_data), len(context_data) + len(outcome_data))
    
    ax1.plot(context_x, context_data, color=COLORS['context'], linewidth=2, label=f'Context ({len(context_data)} min)', alpha=0.8)
    ax1.plot(outcome_x, outcome_data, color=COLORS['outcome'], linewidth=2, label=f'Outcome ({len(outcome_data)} min)', alpha=0.8)
    
    # Add vertical line at prediction boundary
    ax1.axvline(x=len(context_data), color='red', linestyle='--', linewidth=2, alpha=0.7, label='Prediction Point')
    
    # Highlight end-of-day if high priority
    if metadata['priority_score'] > 0.7:
        ax1.fill_between(context_x[-60:], ax1.get_ylim()[0], ax1.get_ylim()[1], 
                        alpha=0.1, color='gold', label='End-of-day period')
    
    ax1.set_xlabel('Time (minutes)', fontsize=12)
    ax1.set_ylabel('Normalized Price', fontsize=12)
    ax1.set_title('Time Series: Context → Outcome', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text box
    context_stats = characteristics['context_stats']
    movement_stats = characteristics['movement_stats']
    stats_text = f"Context: μ={context_stats['mean']:.6f}, σ={context_stats['std']:.6f}, trend={context_stats['trend']:.8f}\n"
    stats_text += f"Movement: {movement_stats['context_total_return']:.3f}% → {movement_stats['outcome_total_return']:.3f}%"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ==============================================================================
    # 2. FFT Magnitude Spectrum
    # ==============================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    freq_bins = get_frequency_bins(len(fft_magnitude))
    magnitude_positive = fft_magnitude[:len(freq_bins)]
    
    # Plot magnitude spectrum
    ax2.plot(freq_bins, magnitude_positive, color=COLORS['magnitude'], linewidth=1.5, alpha=0.8)
    ax2.fill_between(freq_bins, 0, magnitude_positive, color=COLORS['magnitude'], alpha=0.3)
    
    # Highlight frequency bands
    for band_name, (min_freq, max_freq) in FREQ_BANDS.items():
        band_mask = (freq_bins >= min_freq) & (freq_bins <= max_freq)
        if np.any(band_mask):
            color = COLORS[f'{band_name}_freq']
            ax2.fill_between(freq_bins[band_mask], 0, magnitude_positive[band_mask], 
                           color=color, alpha=0.5, label=f'{band_name.capitalize()} freq')
    
    # Mark dominant frequencies
    analysis = signature['analysis']
    for band_name in ['high', 'medium', 'low']:
        dom_freqs = analysis[f'dominant_freq_{band_name}']
        if dom_freqs:
            for freq in dom_freqs[:2]:  # Show top 2
                if 0 <= freq <= freq_bins[-1]:
                    freq_idx = np.argmin(np.abs(freq_bins - freq))
                    ax2.plot(freq, magnitude_positive[freq_idx], 'o', 
                           color=COLORS[f'{band_name}_freq'], markersize=8, markeredgecolor='black')
    
    ax2.set_xlabel('Frequency (cycles/min)', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.set_title('FFT Magnitude Spectrum', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(0.5, freq_bins[-1]))  # Focus on meaningful frequencies
    
    # ==============================================================================
    # 3. FFT Phase Spectrum
    # ==============================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    phase_positive = fft_phase[:len(freq_bins)]
    
    # Plot phase spectrum
    ax3.plot(freq_bins, phase_positive, color=COLORS['phase'], linewidth=1, alpha=0.7, marker='.', markersize=2)
    
    # Highlight frequency bands
    for band_name, (min_freq, max_freq) in FREQ_BANDS.items():
        band_mask = (freq_bins >= min_freq) & (freq_bins <= max_freq)
        if np.any(band_mask):
            color = COLORS[f'{band_name}_freq']
            ax3.scatter(freq_bins[band_mask], phase_positive[band_mask], 
                       c=color, alpha=0.6, s=20, label=f'{band_name.capitalize()}')
    
    ax3.set_xlabel('Frequency (cycles/min)', fontsize=12)
    ax3.set_ylabel('Phase (radians)', fontsize=12)
    ax3.set_title('FFT Phase Spectrum', fontsize=14, fontweight='bold')
    ax3.set_ylim(-np.pi, np.pi)
    ax3.set_xlim(0, min(0.5, freq_bins[-1]))
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ==============================================================================
    # 4. Spectral Energy Distribution
    # ==============================================================================
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Energy by frequency band
    energies = [analysis['energy_high'], analysis['energy_medium'], analysis['energy_low']]
    band_names = ['High\n(1-5min)', 'Medium\n(15-30min)', 'Low\n(60-90min)']
    colors = [COLORS['high_freq'], COLORS['medium_freq'], COLORS['low_freq']]
    
    bars = ax4.bar(band_names, energies, color=colors, alpha=0.7, edgecolor='black')
    
    # Add percentage labels
    total_band_energy = sum(energies)
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        percentage = (energy / total_band_energy * 100) if total_band_energy > 0 else 0
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_ylabel('Spectral Energy', fontsize=12)
    ax4.set_title('Energy Distribution by Band', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ==============================================================================
    # 5. Dominant Frequencies Analysis
    # ==============================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Create frequency analysis table
    dom_freq_data = []
    for band_name in ['high', 'medium', 'low']:
        freqs = analysis[f'dominant_freq_{band_name}']
        energy = analysis[f'energy_{band_name}']
        
        if freqs:
            for i, freq in enumerate(freqs[:3]):  # Top 3
                period_min = 1/freq if freq > 0 else np.inf
                dom_freq_data.append([
                    band_name.capitalize(),
                    f'{freq:.4f}',
                    f'{period_min:.1f}' if period_min != np.inf else 'inf',
                    f'{energy:.2e}'
                ])
    
    if dom_freq_data:
        table_data = pd.DataFrame(dom_freq_data, 
                                columns=['Band', 'Freq (cyc/min)', 'Period (min)', 'Energy'])
        
        ax5.axis('tight')
        ax5.axis('off')
        
        table = ax5.table(cellText=table_data.values,
                         colLabels=table_data.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color code by band
        colors_map = {'High': COLORS['high_freq'], 'Medium': COLORS['medium_freq'], 'Low': COLORS['low_freq']}
        for i, row in enumerate(table_data.values):
            band = row[0]
            table[(i+1, 0)].set_facecolor(colors_map.get(band, 'white'))
            table[(i+1, 0)].set_alpha(0.3)
    
    ax5.set_title('Dominant Frequencies', fontsize=14, fontweight='bold')
    
    # ==============================================================================
    # 6. Spectral Characteristics
    # ==============================================================================
    ax6 = fig.add_subplot(gs[2, 1])
    
    spectral_stats = characteristics['spectral_stats']
    
    # Create radar chart for spectral characteristics
    metrics = ['Peak Freq', 'Centroid', 'Rolloff', 'Flatness']
    values = [
        spectral_stats['peak_frequency'] / 0.5,  # Normalize to [0,1]
        spectral_stats['spectral_centroid'] / 0.5,
        spectral_stats['spectral_rolloff'] / 0.5,
        spectral_stats['spectral_flatness']  # Already [0,1]
    ]
    
    # Ensure values are in [0,1] range
    values = [min(1.0, max(0.0, v)) for v in values]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    values_plot = values + [values[0]]  # Close the polygon
    angles_plot = np.concatenate([angles, [angles[0]]])
    
    ax6.plot(angles_plot, values_plot, 'o-', linewidth=2, color=COLORS['magnitude'])
    ax6.fill(angles_plot, values_plot, alpha=0.25, color=COLORS['magnitude'])
    ax6.set_xticks(angles)
    ax6.set_xticklabels(metrics)
    ax6.set_ylim(0, 1)
    ax6.set_title('Spectral Characteristics', fontsize=14, fontweight='bold')
    ax6.grid(True)
    
    # Add values as text
    for angle, value, metric in zip(angles, values, metrics):
        actual_value = spectral_stats[{
            'Peak Freq': 'peak_frequency',
            'Centroid': 'spectral_centroid', 
            'Rolloff': 'spectral_rolloff',
            'Flatness': 'spectral_flatness'
        }[metric]]
        
        ax6.text(angle, value + 0.1, f'{actual_value:.3f}', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # ==============================================================================
    # 7. Pattern Classification
    # ==============================================================================
    ax7 = fig.add_subplot(gs[2, 2])
    
    # Classify the pattern based on characteristics
    pattern_features = classify_pattern(characteristics, signature)
    
    # Create classification visualization
    feature_names = list(pattern_features.keys())
    feature_values = list(pattern_features.values())
    
    colors_features = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))
    
    wedges, texts, autotexts = ax7.pie(feature_values, labels=feature_names, autopct='%1.1f%%',
                                      colors=colors_features, startangle=90)
    
    ax7.set_title('Pattern Classification', fontsize=14, fontweight='bold')
    
    # ==============================================================================
    # 8. Summary Statistics
    # ==============================================================================
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    # Create comprehensive summary
    summary_text = create_summary_text(signature, characteristics)
    
    ax8.text(0.02, 0.98, summary_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Save plot
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {save_path}")
    
    plt.show()

def classify_pattern(characteristics: Dict, signature: Dict) -> Dict:
    """Classify the pattern based on its characteristics"""
    context_stats = characteristics['context_stats']
    movement_stats = characteristics['movement_stats']
    spectral_stats = characteristics['spectral_stats']
    
    features = {}
    
    # Trend classification
    trend = abs(context_stats['trend'])
    if trend > 1e-6:
        features['Trending'] = trend * 1e6
    else:
        features['Sideways'] = 1.0
    
    # Volatility classification
    volatility = movement_stats['context_volatility']
    if volatility > 0.01:
        features['High Vol'] = volatility * 100
    elif volatility > 0.005:
        features['Med Vol'] = volatility * 100
    else:
        features['Low Vol'] = volatility * 100
    
    # Frequency dominance
    analysis = signature['analysis']
    total_energy = analysis['energy_high'] + analysis['energy_medium'] + analysis['energy_low']
    
    if total_energy > 0:
        if analysis['energy_high'] / total_energy > 0.5:
            features['High Freq Dom'] = analysis['energy_high'] / total_energy
        elif analysis['energy_medium'] / total_energy > 0.4:
            features['Med Freq Dom'] = analysis['energy_medium'] / total_energy
        else:
            features['Low Freq Dom'] = analysis['energy_low'] / total_energy
    
    # Spectral characteristics
    if spectral_stats['spectral_flatness'] > 0.5:
        features['Noisy'] = spectral_stats['spectral_flatness']
    else:
        features['Tonal'] = 1 - spectral_stats['spectral_flatness']
    
    return features

def create_summary_text(signature: Dict, characteristics: Dict) -> str:
    """Create comprehensive summary text"""
    metadata = signature['metadata']
    context_stats = characteristics['context_stats']
    outcome_stats = characteristics['outcome_stats']
    movement_stats = characteristics['movement_stats']
    spectral_stats = characteristics['spectral_stats']
    analysis = signature['analysis']
    
    summary = f"SIGNATURE ANALYSIS SUMMARY\n"
    summary += f"{'='*50}\n\n"
    
    summary += f"METADATA:\n"
    summary += f"  ID: {metadata['sequence_id']}\n"
    summary += f"  Source: {metadata['source_file']}\n"
    summary += f"  Priority Score: {metadata['priority_score']:.3f}\n"
    summary += f"  Time Range: {metadata['timestamp_start']} → {metadata['timestamp_end']}\n\n"
    
    summary += f"TIME SERIES CHARACTERISTICS:\n"
    summary += f"  Context  - Mean: {context_stats['mean']:.6f}, Std: {context_stats['std']:.6f}, Trend: {context_stats['trend']:.8f}\n"
    summary += f"  Outcome  - Mean: {outcome_stats['mean']:.6f}, Std: {outcome_stats['std']:.6f}, Trend: {outcome_stats['trend']:.8f}\n"
    summary += f"  Movement - Context: {movement_stats['context_total_return']:.3f}%, Outcome: {movement_stats['outcome_total_return']:.3f}%\n"
    summary += f"  Volatility - Context: {movement_stats['context_volatility']:.6f}, Outcome: {movement_stats['outcome_volatility']:.6f}\n\n"
    
    summary += f"SPECTRAL ANALYSIS:\n"
    summary += f"  Peak Frequency: {spectral_stats['peak_frequency']:.4f} cyc/min ({1/spectral_stats['peak_frequency']:.1f} min period)\n"
    summary += f"  Spectral Centroid: {spectral_stats['spectral_centroid']:.4f} cyc/min\n"
    summary += f"  Spectral Rolloff: {spectral_stats['spectral_rolloff']:.4f} cyc/min\n"
    summary += f"  Spectral Flatness: {spectral_stats['spectral_flatness']:.4f} (0=tonal, 1=noisy)\n\n"
    
    summary += f"FREQUENCY BAND ENERGY:\n"
    total_energy = analysis['energy_high'] + analysis['energy_medium'] + analysis['energy_low']
    summary += f"  High (1-5min):    {analysis['energy_high']:.2e} ({analysis['energy_high']/total_energy*100:.1f}%)\n"
    summary += f"  Medium (15-30min): {analysis['energy_medium']:.2e} ({analysis['energy_medium']/total_energy*100:.1f}%)\n"
    summary += f"  Low (60-90min):   {analysis['energy_low']:.2e} ({analysis['energy_low']/total_energy*100:.1f}%)\n"
    summary += f"  Total Energy:     {analysis['total_energy']:.2e}\n"
    
    return summary

# ==============================================================================
# Main Functions
# ==============================================================================

def visualize_random_signatures(num_signatures: int = 3):
    """Visualize random signatures from the database"""
    print("🔮 FFT Signature Visualizer")
    print("=" * 50)
    
    # Load database metadata
    try:
        metadata = load_database_metadata()
        print(f"📊 Database: {metadata['database_info']['total_signatures']:,} signatures")
        print(f"📅 Generated: {metadata['generation_info']['processing_date']}")
        print()
    except Exception as e:
        print(f"⚠️ Could not load metadata: {e}")
    
    # Get random signature files
    try:
        signature_files = get_random_signature_files(num_signatures)
        print(f"🎲 Selected {len(signature_files)} random signatures for analysis")
    except Exception as e:
        print(f"❌ Error selecting signatures: {e}")
        return
    
    # Process each signature
    for i, signature_file in enumerate(signature_files, 1):
        print(f"\n{'='*60}")
        print(f"🔍 Analyzing signature {i}/{len(signature_files)}: {signature_file.name}")
        print(f"{'='*60}")
        
        try:
            # Load signature
            signature = load_signature(signature_file)
            print(f"✅ Loaded signature: {signature['metadata']['sequence_id']}")
            
            # Analyze characteristics
            characteristics = analyze_signature_characteristics(signature)
            print(f"📈 Analyzed characteristics")
            
            # Create visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"fft_signature_{signature['metadata']['sequence_id']}_{timestamp}.png"
            save_path = PLOTS_DIR / plot_filename
            
            print(f"🎨 Creating comprehensive visualization...")
            create_comprehensive_visualization(signature, characteristics, save_path)
            
        except Exception as e:
            print(f"❌ Error processing {signature_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Visualization complete! Plots saved to: {PLOTS_DIR}")

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize FFT signatures from the database')
    parser.add_argument('--num', type=int, default=3,
                      help='Number of random signatures to visualize (default: 3)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎯 Random seed set to: {args.seed}")
    
    try:
        visualize_random_signatures(args.num)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Visualization interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
