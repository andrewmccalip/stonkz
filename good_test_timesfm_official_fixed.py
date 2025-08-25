#!/usr/bin/env python3
"""
Fixed version of the official TimesFM test that actually creates the plot.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import torch
import timesfm

def test_timesfm_pytorch():
    """Test official TimesFM with PyTorch backend."""
    print("\n🚀 Testing Official TimesFM (PyTorch Backend)")
    print("=" * 50)
    
    # Detect if GPU is available
    if torch.cuda.is_available():
        backend = "gpu"
        print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        backend = "cpu"
        print("💻 No GPU detected, using CPU")
    
    # Create model - it will automatically use PyTorch
    print(f"\n📊 Loading TimesFM model on {backend.upper()}...")
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=backend,  # Use GPU if available
            per_core_batch_size=32,
            horizon_len=128,
            num_layers=20,
            use_positional_embedding=True,
            context_len=512,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )
    
    print("✅ Model loaded successfully!")
    print(f"📝 Backend: PyTorch on {backend.upper()}")
    
    # Generate simple test data
    np.random.seed(42)
    
    # Use shorter sequences
    context_len = 120
    horizon_len = 24
    
    # Create a simple time series
    t = np.arange(context_len + horizon_len)
    signal = 100 + 10 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 1, len(t))
    
    # Split into context and target
    context = signal[:context_len]
    target = signal[context_len:context_len + horizon_len]
    
    # Make prediction
    print("\n🔮 Making forecast...")
    print(f"  Context length: {len(context)}")
    print(f"  Horizon length: {horizon_len}")
    
    inputs = [context.tolist()]
    freq = [0]
    
    forecast, _ = model.forecast(inputs, freq)
    forecast = forecast[0][:horizon_len]
    print("✅ Forecast completed!")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot context
    context_x = np.arange(context_len)
    plt.plot(context_x, context, 'b-', label='Context', alpha=0.7)
    
    # Plot target and forecast
    forecast_x = np.arange(context_len, context_len + horizon_len)
    plt.plot(forecast_x, target, 'g-', label='Ground Truth', linewidth=2)
    plt.plot(forecast_x, forecast, 'r--', label='Forecast', linewidth=2)
    
    # Add vertical line at prediction start
    plt.axvline(x=context_len, color='black', linestyle=':', alpha=0.5)
    
    # Formatting
    plt.legend()
    plt.title('TimesFM Official Package - PyTorch Backend')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # Calculate MSE and add to plot
    mse = np.mean((forecast - target) ** 2)
    plt.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save plot
    print("\n📸 Saving plot...")
    
    # Create directory with absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, 'stock_plots')
    os.makedirs(plot_dir, exist_ok=True)
    print(f"  Created directory: {plot_dir}")
    
    # Save with absolute path
    output_path = os.path.join(plot_dir, 'timesfm_official_pytorch.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    # Verify the file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"\n✅ Plot saved successfully!")
        print(f"  Path: {output_path}")
        print(f"  Size: {file_size:,} bytes")
    else:
        print(f"\n❌ Failed to save plot to {output_path}")
    
    # Also show the plot
    #plt.show()
    
    print(f"\n📊 MSE: {mse:.4f}")
    print("\n✨ Success! This is the setup you should use for finetuning.")
    
    return model

if __name__ == "__main__":
    test_timesfm_pytorch()
