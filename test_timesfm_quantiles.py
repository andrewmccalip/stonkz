"""
Test TimesFM's quantile prediction capabilities
"""

import numpy as np
import torch
import timesfm

# Initialize model using the correct API
TIMESFM_BACKEND = "gpu" if torch.cuda.is_available() else "cpu"
TIMESFM_MODEL_REPO = "google/timesfm-1.0-200m-pytorch"

# Initialize with quantiles
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]  # 10%, 25%, 50%, 75%, 90% percentiles

tfm = timesfm.TimesFm(
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

# Create sample data
context = np.random.randn(448) * 0.01 + 1.0  # Normalized prices around 1.0
inputs = [context.tolist()]
freq = [0]  # High frequency

print("Testing TimesFM quantile predictions...")
print("=" * 50)

# Try different methods
try:
    # Method 1: Standard forecast
    point_forecast, _ = tfm.forecast(inputs, freq)
    print(f"✅ Point forecast shape: {point_forecast.shape}")
    print(f"   Values: {point_forecast[0][:5]}")
except Exception as e:
    print(f"❌ Standard forecast error: {e}")

try:
    # Method 2: Forecast with quantiles
    # Check if quantiles parameter exists
    point_forecast, quantile_forecast = tfm.forecast(
        inputs, 
        freq,
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    print(f"\n✅ Quantile forecast available!")
    print(f"   Point forecast shape: {point_forecast.shape}")
    print(f"   Quantile forecast shape: {quantile_forecast.shape}")
    print(f"   Quantiles: 10%, 25%, 50%, 75%, 90%")
except Exception as e:
    print(f"\n❌ Quantile forecast error: {e}")

# Method 3: Check model attributes
print("\n📊 Model attributes:")
for attr in dir(tfm):
    if 'quantile' in attr.lower() or 'forecast' in attr:
        print(f"   - {attr}")

# Method 4: Inspect forecast method
import inspect
print("\n📝 Forecast method signature:")
sig = inspect.signature(tfm.forecast)
print(f"   {sig}")
