"""
Test TimesFM's quantile prediction capabilities
Shows how to get prediction ranges (confidence intervals)
"""

import numpy as np
import torch
import timesfm
import matplotlib.pyplot as plt

# Initialize model using the correct API
TIMESFM_BACKEND = "gpu" if torch.cuda.is_available() else "cpu"
TIMESFM_MODEL_REPO = "google/timesfm-1.0-200m-pytorch"

print("🔍 Testing TimesFM Quantile Predictions")
print("=" * 50)

# Test 1: Without quantiles (point forecast only)
print("\n1️⃣ Point Forecast Only:")
tfm_point = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend=TIMESFM_BACKEND,
        per_core_batch_size=32,
        horizon_len=64,
        context_len=448,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id=TIMESFM_MODEL_REPO
    ),
)

# Create sample data - normalized stock prices
np.random.seed(42)
context = 1.0 + np.cumsum(np.random.randn(448) * 0.001)  # Random walk around 1.0
inputs = [context.tolist()]
freq = [0]  # High frequency

# Get point forecast
point_forecast, _ = tfm_point.forecast(inputs, freq)
print(f"   Shape: {point_forecast.shape}")
print(f"   First 5 values: {point_forecast[0][:5]}")

# Test 2: With quantiles (prediction intervals)
print("\n2️⃣ Quantile Forecast (with uncertainty):")
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]  # 10%, 25%, 50%, 75%, 90% percentiles

tfm_quantile = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend=TIMESFM_BACKEND,
        per_core_batch_size=32,
        horizon_len=64,
        context_len=448,
        quantiles=quantiles,  # Enable quantile predictions
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id=TIMESFM_MODEL_REPO
    ),
)

# Get forecast with quantiles
forecast_output = tfm_quantile.forecast(inputs, freq)
print(f"   Forecast output type: {type(forecast_output)}")
print(f"   Number of outputs: {len(forecast_output)}")

if len(forecast_output) == 2:
    point_forecast_q, quantile_forecast = forecast_output
    print(f"   Point forecast shape: {point_forecast_q.shape}")
    
    if quantile_forecast is not None:
        print(f"   Quantile forecast shape: {quantile_forecast.shape}")
        print(f"   ✅ Quantile predictions available!")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Time steps
        context_time = np.arange(len(context))
        forecast_time = np.arange(len(context), len(context) + 64)
        
        # Plot context
        ax.plot(context_time, context, 'b-', label='Historical Data', alpha=0.8)
        
        # Plot point forecast
        ax.plot(forecast_time, point_forecast_q[0], 'r-', label='Point Forecast', linewidth=2)
        
        # Plot prediction intervals if available
        if quantile_forecast.ndim >= 2:
            # Extract quantiles for first series
            q_10 = quantile_forecast[0, :, 0] if quantile_forecast.ndim == 3 else quantile_forecast[:, 0]
            q_25 = quantile_forecast[0, :, 1] if quantile_forecast.ndim == 3 else quantile_forecast[:, 1]
            q_50 = quantile_forecast[0, :, 2] if quantile_forecast.ndim == 3 else quantile_forecast[:, 2]
            q_75 = quantile_forecast[0, :, 3] if quantile_forecast.ndim == 3 else quantile_forecast[:, 3]
            q_90 = quantile_forecast[0, :, 4] if quantile_forecast.ndim == 3 else quantile_forecast[:, 4]
            
            # Plot confidence intervals
            ax.fill_between(forecast_time, q_10, q_90, alpha=0.2, color='red', label='80% CI (10%-90%)')
            ax.fill_between(forecast_time, q_25, q_75, alpha=0.3, color='red', label='50% CI (25%-75%)')
            ax.plot(forecast_time, q_50, 'r--', alpha=0.5, label='Median (50%)')
            
            print(f"\n   📊 Prediction Range at t+10:")
            print(f"      10% quantile: {q_10[10]:.4f}")
            print(f"      25% quantile: {q_25[10]:.4f}")
            print(f"      50% quantile: {q_50[10]:.4f} (median)")
            print(f"      75% quantile: {q_75[10]:.4f}")
            print(f"      90% quantile: {q_90[10]:.4f}")
            print(f"      => 80% confidence interval: [{q_10[10]:.4f}, {q_90[10]:.4f}]")
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Normalized Price')
        ax.set_title('TimesFM Forecast with Prediction Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('timesfm_quantile_forecast.png', dpi=150)
        print("\n   ✅ Plot saved to timesfm_quantile_forecast.png")
    else:
        print("   ⚠️ Quantile forecast is None - may not be supported in this version")
else:
    print("   ⚠️ Unexpected output format")

print("\n3️⃣ Implementation in Training Pipeline:")
print("""
To use quantile predictions in your pipeline:

1. Initialize with quantiles:
   ```python
   tfm = timesfm.TimesFm(
       hparams=timesfm.TimesFmHparams(
           quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
           # other params...
       ),
       checkpoint=...
   )
   ```

2. Get predictions:
   ```python
   point_forecast, quantile_forecast = tfm.forecast(inputs, freq)
   ```

3. Use for:
   - Uncertainty-aware loss: weight predictions by confidence
   - Risk assessment: use lower quantiles for conservative predictions
   - Trading signals: buy when price < 10th percentile prediction
   """)
