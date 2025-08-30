#!/usr/bin/env python3
"""
OpenAI Fine-Tuning Data Generator for Time Series Prediction

This script processes existing sequence data into OpenAI's JSONL format for fine-tuning.
Creates training data with historical context and future predictions, following the 
structure used in backtest_unified.py.

Features:
1. Processes existing CSV sequences from datasets/sequences/
2. Creates context-prediction pairs for time series forecasting
3. Generates 500, 5000, and full dataset variants
4. Exports to OpenAI-compliant JSONL format
5. Includes comprehensive data validation and statistics
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import time
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

# Set random seed for reproducibility [[memory:7111902]]
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Data processing parameters - adjusted for sequence file lengths
CONTEXT_LENGTH = 400           # Historical context in minutes (~6.7 hours) 
HORIZON_LENGTH = 96            # Prediction horizon in minutes (~1.6 hours)

# Dataset sizes to generate
DATASET_SIZES = {
    'small': 500,      # Small test dataset
    'medium': 5000,    # Medium training dataset  
    'full': None       # Full dataset (all available data)
}

# Performance optimization settings
BATCH_SIZE = 50              # Number of files to process in parallel
MAX_WORKERS = min(cpu_count(), 8)  # Number of worker threads
SAMPLES_PER_FILE = 5         # Reduce samples per file for faster processing
MAX_BATCHES_FOR_TESTING = 20 # Set to None for full processing, or number for testing

# Directory paths
SCRIPT_DIR = Path(__file__).parent
SEQUENCES_DIR = SCRIPT_DIR / "datasets" / "sequences"
OUTPUT_DIR = SCRIPT_DIR / "openai"
OUTPUT_DIR.mkdir(exist_ok=True)

# ==============================================================================
# Data Processing Functions
# ==============================================================================

def load_sequence_file(csv_file: Path) -> Optional[pd.DataFrame]:
    """
    Load a single sequence CSV file and validate its structure.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        DataFrame with validated sequence data or None if invalid
    """
    try:
        # Load the CSV file - following backtest_unified.py pattern
        df = pd.read_csv(csv_file)
        
        # Validate required columns
        required_cols = ['timestamp_pt', 'close']
        if not all(col in df.columns for col in required_cols):
            print(f"⚠️  Missing required columns in {csv_file.name}")
            return None
        
        # Ensure we have enough data for context + horizon
        min_required = CONTEXT_LENGTH + HORIZON_LENGTH + 1
        if len(df) < min_required:
            # Debug: print why files are being rejected
            if len(df) >= 500:  # Only print for reasonably sized files
                print(f"   ⚠️  {csv_file.name}: {len(df)} rows < {min_required} required")
            return None
            
        # Sort by timestamp to ensure proper order
        df['timestamp_pt'] = pd.to_datetime(df['timestamp_pt'])
        df = df.sort_values('timestamp_pt').reset_index(drop=True)
        
        # Remove any rows with missing close prices
        df = df.dropna(subset=['close'])
        
        return df
        
    except Exception as e:
        print(f"⚠️  Error loading {csv_file.name}: {e}")
        return None

def extract_training_samples(df: pd.DataFrame, file_name: str, 
                           max_samples_per_file: int = SAMPLES_PER_FILE) -> List[Dict]:
    """
    Extract training samples from a sequence DataFrame.
    Each sample contains context data and corresponding future predictions.
    
    Args:
        df: DataFrame with sequence data
        file_name: Name of the source file (for tracking)
        max_samples_per_file: Maximum samples to extract per file
        
    Returns:
        List of training samples in OpenAI format
    """
    training_samples = []
    
    # Calculate valid starting positions
    min_start = CONTEXT_LENGTH
    max_start = len(df) - HORIZON_LENGTH
    
    if max_start <= min_start:
        return training_samples
    
    # Generate random starting positions for samples
    available_range = max_start - min_start
    # Use smaller step size for shorter sequences, ensure we get at least 1 sample if range allows
    step_size = max(1, available_range // max_samples_per_file) if available_range >= max_samples_per_file else 1
    num_samples = min(max_samples_per_file, available_range // step_size)
    num_samples = max(1, num_samples) if available_range > 0 else 0  # Ensure at least 1 sample
    
    if num_samples <= 0:
        return training_samples
        
    start_positions = sorted(random.sample(range(min_start, max_start), num_samples))
    
    for start_idx in start_positions:
        try:
            # Extract context data (historical prices)
            context_start = start_idx - CONTEXT_LENGTH
            context_end = start_idx
            context_data = df['close'].iloc[context_start:context_end].values
            
            # Extract future data (prediction target)
            horizon_start = start_idx
            horizon_end = start_idx + HORIZON_LENGTH
            future_data = df['close'].iloc[horizon_start:horizon_end].values
            
            # Get timestamp information
            timestamp = df.iloc[start_idx]['timestamp_pt'].strftime("%Y-%m-%d %H:%M:%S")
            
            # Create context string (normalized prices for better training)
            # Normalize relative to the last context value to focus on relative changes
            last_price = context_data[-1]
            normalized_context = [round(x / last_price, 7) for x in context_data]
            normalized_future = [round(x / last_price, 7) for x in future_data]
            
            # Format context data as a structured prompt
            context_prompt = f"""Analyze the following time series data and predict the next {HORIZON_LENGTH} values.

Historical Context ({CONTEXT_LENGTH} data points):
Timestamp: {timestamp}
Last Price: {last_price:.7f}
Normalized Historical Values: {normalized_context}

Please provide a prediction for the next {HORIZON_LENGTH} normalized values, maintaining the same format and precision."""

            # Format the expected response
            response = f"Predicted Values: {normalized_future}"
            
            # Create OpenAI training format
            training_sample = {
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert time series forecasting model specialized in financial market prediction. You analyze historical price data and provide accurate predictions for future price movements. Focus on patterns, trends, and relative price changes."
                    },
                    {
                        "role": "user", 
                        "content": context_prompt
                    },
                    {
                        "role": "assistant", 
                        "content": response
                    }
                ],
                # Metadata for tracking (not used by OpenAI but useful for analysis)
                "metadata": {
                    "source_file": file_name,
                    "timestamp": timestamp,
                    "context_length": CONTEXT_LENGTH,
                    "horizon_length": HORIZON_LENGTH,
                    "last_price": round(last_price, 7),
                    "start_idx": start_idx
                }
            }
            
            training_samples.append(training_sample)
            
        except Exception as e:
            print(f"⚠️  Error extracting sample from {file_name} at index {start_idx}: {e}")
            continue
    
    return training_samples

def create_alternative_format_samples(df: pd.DataFrame, file_name: str, 
                                   max_samples_per_file: int = 2) -> List[Dict]:
    """
    Create alternative format samples focusing on directional predictions.
    This format asks for up/down predictions rather than exact values.
    
    Args:
        df: DataFrame with sequence data
        file_name: Name of the source file
        max_samples_per_file: Maximum samples to create
        
    Returns:
        List of directional prediction samples
    """
    training_samples = []
    
    min_start = CONTEXT_LENGTH
    max_start = len(df) - HORIZON_LENGTH
    
    if max_start <= min_start:
        return training_samples
    
    num_samples = min(max_samples_per_file, (max_start - min_start) // 100)
    if num_samples <= 0:
        return training_samples
        
    start_positions = sorted(random.sample(range(min_start, max_start), num_samples))
    
    for start_idx in start_positions:
        try:
            # Extract context and future data
            context_data = df['close'].iloc[start_idx - CONTEXT_LENGTH:start_idx].values
            future_data = df['close'].iloc[start_idx:start_idx + HORIZON_LENGTH].values
            
            # Calculate directional movements
            current_price = context_data[-1]
            
            # Calculate returns for context (for pattern analysis)
            context_returns = [round(x, 7) for x in (np.diff(context_data) / context_data[:-1] * 100)]  # Percentage returns
            
            # Calculate future directional movements
            future_directions = []
            for i in range(min(10, len(future_data) - 1)):  # First 10 future points
                direction = "UP" if future_data[i + 1] > future_data[i] else "DOWN"
                change_pct = round((future_data[i + 1] - future_data[i]) / future_data[i] * 100, 7)
                future_directions.append(f"{direction} ({change_pct:.7f}%)")
            
            timestamp = df.iloc[start_idx]['timestamp_pt'].strftime("%Y-%m-%d %H:%M:%S")
            
            # Create directional prediction prompt
            context_prompt = f"""Analyze the price movement pattern and predict the directional changes.

Timestamp: {timestamp}
Current Price: {current_price:.7f}
Recent Returns (%): {context_returns[-20:]}  

Based on this pattern, predict the directional movement (UP/DOWN) and approximate percentage change for the next 10 time periods."""

            response = f"Directional Predictions: {future_directions}"
            
            training_sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial market analyst specialized in predicting short-term price directions. You analyze recent price movements and identify patterns to predict whether prices will move UP or DOWN, along with estimated percentage changes."
                    },
                    {
                        "role": "user",
                        "content": context_prompt
                    },
                    {
                        "role": "assistant", 
                        "content": response
                    }
                ],
                "metadata": {
                    "source_file": file_name,
                    "timestamp": timestamp,
                    "format_type": "directional",
                    "current_price": round(current_price, 7)
                }
            }
            
            training_samples.append(training_sample)
            
        except Exception as e:
            continue
    
    return training_samples

def process_single_file(csv_file: Path) -> List[Dict]:
    """
    Process a single CSV file and extract training samples.
    
    Args:
        csv_file: Path to the CSV file to process
        
    Returns:
        List of training samples from this file
    """
    # Load and validate the file
    df = load_sequence_file(csv_file)
    if df is None:
        return []
    
    # Extract standard training samples
    samples = extract_training_samples(df, csv_file.name)
    
    # Extract alternative format samples (directional predictions)  
    alt_samples = create_alternative_format_samples(df, csv_file.name)
    
    # Combine samples
    return samples + alt_samples

def process_batch_of_files(csv_files_batch: List[Path]) -> List[Dict]:
    """
    Process a batch of files in parallel using ThreadPoolExecutor.
    
    Args:
        csv_files_batch: List of CSV files to process
        
    Returns:
        Combined list of training samples from all files in batch
    """
    batch_samples = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all files in the batch for processing
        future_to_file = {
            executor.submit(process_single_file, csv_file): csv_file 
            for csv_file in csv_files_batch
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            csv_file = future_to_file[future]
            try:
                file_samples = future.result()
                if file_samples:
                    batch_samples.extend(file_samples)
            except Exception as e:
                print(f"⚠️  Error processing {csv_file.name}: {e}")
    
    return batch_samples

def process_all_sequences() -> List[Dict]:
    """
    Process all sequence files in batches using parallel processing.
    Outputs smaller datasets as soon as thresholds are reached.
    
    Returns:
        List of all training samples from all files
    """
    print("🔍 Scanning for sequence files...")
    
    # Find all CSV files in sequences directory
    csv_files = list(SEQUENCES_DIR.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {SEQUENCES_DIR}")
    
    print(f"📁 Found {len(csv_files)} sequence files")
    print(f"⚡ Using {MAX_WORKERS} workers with batch size {BATCH_SIZE}")
    
    all_training_samples = []
    datasets_saved = set()  # Track which datasets we've already saved
    
    # Split files into batches for parallel processing
    file_batches = [csv_files[i:i + BATCH_SIZE] for i in range(0, len(csv_files), BATCH_SIZE)]
    total_batches = len(file_batches)
    
    # Limit batches for testing if specified
    if MAX_BATCHES_FOR_TESTING is not None:
        file_batches = file_batches[:MAX_BATCHES_FOR_TESTING]
        total_batches = len(file_batches)
        print(f"🧪 Testing mode: Processing only {total_batches} batches...")
    else:
        print(f"📦 Processing {total_batches} batches...")
    
    # Process batches with progress tracking
    start_time = time.time()
    for batch_idx, batch_files in enumerate(file_batches, 1):
        batch_start_time = time.time()
        
        # Process the batch in parallel
        print(f"\n🔄 Processing batch {batch_idx}/{total_batches} ({len(batch_files)} files)...")
        batch_samples = process_batch_of_files(batch_files)
        
        if batch_samples:
            all_training_samples.extend(batch_samples)
            
        batch_time = time.time() - batch_start_time
        current_count = len(all_training_samples)
        
        print(f"   ✅ Batch {batch_idx} complete: +{len(batch_samples)} samples in {batch_time:.1f}s")
        print(f"   📊 Total samples: {current_count:,}")
        
        # Check if we should save intermediate datasets
        # Save 500 dataset as soon as we have enough samples
        if current_count >= 500 and "small" not in datasets_saved:
            print(f"\n🚀 Reached 500 samples! Saving small dataset...")
            # Shuffle for better distribution
            shuffled_samples = all_training_samples.copy()
            random.shuffle(shuffled_samples)
            dataset_500 = shuffled_samples[:500]
            
            filename = f"training_data_small_{len(dataset_500)}.jsonl"
            output_path = save_dataset(dataset_500, filename)
            
            if validate_jsonl_format(output_path):
                print(f"✅ Small dataset ready: {output_path.name}")
                datasets_saved.add("small")
            else:
                print(f"⚠️  Validation failed for small dataset")
        
        # Save 5000 dataset as soon as we have enough samples
        if current_count >= 5000 and "medium" not in datasets_saved:
            print(f"\n🚀 Reached 5000 samples! Saving medium dataset...")
            # Shuffle for better distribution
            shuffled_samples = all_training_samples.copy()
            random.shuffle(shuffled_samples)
            dataset_5000 = shuffled_samples[:5000]
            
            filename = f"training_data_medium_{len(dataset_5000)}.jsonl"
            output_path = save_dataset(dataset_5000, filename)
            
            if validate_jsonl_format(output_path):
                print(f"✅ Medium dataset ready: {output_path.name}")
                datasets_saved.add("medium")
            else:
                print(f"⚠️  Validation failed for medium dataset")
        
        # Show estimated time remaining
        elapsed_time = time.time() - start_time
        if batch_idx > 1:  # Need at least 2 batches for estimation
            avg_batch_time = elapsed_time / batch_idx
            remaining_batches = total_batches - batch_idx
            eta_seconds = remaining_batches * avg_batch_time
            eta_minutes = eta_seconds / 60
            print(f"   ⏱️  ETA: {eta_minutes:.1f} minutes remaining")
    
    total_time = time.time() - start_time
    print(f"\n📊 Processing Summary:")
    print(f"   Total processing time: {total_time:.1f} seconds")
    print(f"   Files processed: {len(csv_files)}")
    print(f"   Total samples extracted: {len(all_training_samples):,}")
    print(f"   Average samples per second: {len(all_training_samples) / total_time:.1f}")
    print(f"   Intermediate datasets saved: {list(datasets_saved)}")
    
    return all_training_samples

def save_dataset(samples: List[Dict], filename: str, include_metadata: bool = False):
    """
    Save training samples to JSONL format for OpenAI fine-tuning.
    
    Args:
        samples: List of training samples
        filename: Output filename
        include_metadata: Whether to include metadata in output
    """
    output_path = OUTPUT_DIR / filename
    
    print(f"💾 Saving {len(samples)} samples to {filename}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            # Create clean sample for OpenAI (remove metadata)
            clean_sample = {"messages": sample["messages"]}
            
            # Optionally include metadata for debugging
            if include_metadata and "metadata" in sample:
                clean_sample["metadata"] = sample["metadata"]
            
            json.dump(clean_sample, f)
            f.write('\n')
    
    print(f"✅ Successfully saved to: {output_path}")
    return output_path

def validate_jsonl_format(filepath: Path) -> bool:
    """
    Validate that the JSONL file is properly formatted for OpenAI.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        print(f"🔍 Validating {filepath.name}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        sample_count = 0
        for i, line in enumerate(lines[:10]):  # Check first 10 samples
            try:
                data = json.loads(line.strip())
                
                # Check required structure
                if "messages" not in data:
                    print(f"❌ Missing 'messages' key in line {i+1}")
                    return False
                
                messages = data["messages"]
                if not isinstance(messages, list) or len(messages) < 3:
                    print(f"❌ Invalid messages structure in line {i+1}")
                    return False
                
                # Check message roles
                roles = [msg.get("role") for msg in messages]
                expected_roles = ["system", "user", "assistant"]
                if roles != expected_roles:
                    print(f"❌ Invalid role sequence in line {i+1}: {roles}")
                    return False
                
                sample_count += 1
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error in line {i+1}: {e}")
                return False
        
        print(f"✅ Validation passed - {len(lines)} samples, format correct")
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def print_sample_statistics(samples: List[Dict]):
    """
    Print detailed statistics about the training samples.
    
    Args:
        samples: List of training samples
    """
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {len(samples):,}")
    
    # Count by format type
    standard_count = sum(1 for s in samples if s.get("metadata", {}).get("format_type") != "directional")
    directional_count = len(samples) - standard_count
    
    print(f"   Standard prediction samples: {standard_count:,}")
    print(f"   Directional prediction samples: {directional_count:,}")
    
    # Count by source files
    source_files = set()
    for sample in samples:
        if "metadata" in sample and "source_file" in sample["metadata"]:
            source_files.add(sample["metadata"]["source_file"])
    
    print(f"   Unique source files: {len(source_files)}")
    
    # Calculate average message lengths
    user_lengths = []
    assistant_lengths = []
    
    for sample in samples[:1000]:  # Sample first 1000 for efficiency
        messages = sample["messages"]
        for msg in messages:
            if msg["role"] == "user":
                user_lengths.append(len(msg["content"]))
            elif msg["role"] == "assistant":
                assistant_lengths.append(len(msg["content"]))
    
    if user_lengths:
        print(f"   Avg user message length: {np.mean(user_lengths):.0f} chars")
    if assistant_lengths:
        print(f"   Avg assistant message length: {np.mean(assistant_lengths):.0f} chars")

# ==============================================================================
# Main Processing Function
# ==============================================================================

def main():
    """Main function to process sequences and generate OpenAI training datasets."""
    
    print("🎯 OpenAI Fine-Tuning Data Generator for Time Series Prediction")
    print("=" * 70)
    print(f"📊 Configuration:")
    print(f"   Context Length: {CONTEXT_LENGTH} minutes")
    print(f"   Prediction Horizon: {HORIZON_LENGTH} minutes") 
    print(f"   Random Seed: {RANDOM_SEED}")
    print(f"   Output Directory: {OUTPUT_DIR}")
    print(f"⚡ Performance Settings:")
    print(f"   Batch Size: {BATCH_SIZE} files per batch")
    print(f"   Worker Threads: {MAX_WORKERS}")
    print(f"   Samples per File: {SAMPLES_PER_FILE}")
    print()
    
    try:
        # Process all sequences to extract training samples
        print("🚀 Starting sequence processing...")
        all_samples = process_all_sequences()
        
        if not all_samples:
            print("❌ No training samples extracted. Check your sequence files.")
            return
        
        # Shuffle all samples for better training distribution
        random.shuffle(all_samples)
        
        # Print statistics
        print_sample_statistics(all_samples)
        
        # Generate datasets of different sizes
        datasets_created = []
        
        # Check which datasets were already saved during processing
        existing_files = list(OUTPUT_DIR.glob("training_data_*.jsonl"))
        already_saved = set()
        
        for file in existing_files:
            if "small" in file.name:
                already_saved.add("small")
                datasets_created.append(("small", file, 500))
            elif "medium" in file.name:
                already_saved.add("medium") 
                datasets_created.append(("medium", file, 5000))
        
        # Only create datasets that weren't already saved during processing
        for size_name, size_limit in DATASET_SIZES.items():
            if size_name in already_saved:
                print(f"✅ {size_name} dataset already saved during processing")
                continue
                
            print(f"\n📦 Creating {size_name} dataset...")
            
            if size_limit is None:
                # Full dataset
                dataset_samples = all_samples
                filename = f"training_data_full_{len(dataset_samples)}.jsonl"
            else:
                # Limited size dataset
                dataset_samples = all_samples[:size_limit] if len(all_samples) >= size_limit else all_samples
                filename = f"training_data_{size_name}_{len(dataset_samples)}.jsonl"
            
            # Save the dataset
            output_path = save_dataset(dataset_samples, filename)
            
            # Validate the output
            if validate_jsonl_format(output_path):
                datasets_created.append((size_name, output_path, len(dataset_samples)))
            else:
                print(f"⚠️  Validation failed for {filename}")
        
        # Final summary
        print(f"\n🎉 Dataset Generation Complete!")
        print(f"   Created {len(datasets_created)} datasets:")
        
        for size_name, path, count in datasets_created:
            print(f"   • {size_name}: {count:,} samples → {path.name}")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Review the generated JSONL files in: {OUTPUT_DIR}")
        print(f"   2. Upload to OpenAI for fine-tuning using their API")
        print(f"   3. Monitor training progress and adjust hyperparameters as needed")
        
        # Save a metadata summary
        metadata_summary = {
            "generation_timestamp": datetime.now().isoformat(),
            "total_samples_processed": len(all_samples),
            "context_length": CONTEXT_LENGTH,
            "horizon_length": HORIZON_LENGTH,
            "random_seed": RANDOM_SEED,
            "datasets_created": [
                {
                    "name": size_name,
                    "filename": path.name,
                    "sample_count": count
                }
                for size_name, path, count in datasets_created
            ]
        }
        
        metadata_path = OUTPUT_DIR / "generation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_summary, f, indent=2)
        
        print(f"   📋 Generation metadata saved to: {metadata_path.name}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()
