"""
TimesFM Quantile Prediction Example
Shows how to get prediction ranges (median with confidence intervals)
"""

import numpy as np
import torch
import timesfm
import matplotlib.pyplot as plt

print("📊 TimesFM Quantile Predictions for Stock Price Forecasting")
print("=" * 60)

# Initialize model
TIMESFM_BACKEND = "gpu" if torch.cuda.is_available() else "cpu"
TIMESFM_MODEL_REPO = "google/timesfm-1.0-200m-pytorch"

# Standard model (point forecast only)
print("\n1️⃣ Initializing standard TimesFM model...")
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

# Create realistic stock price data
np.random.seed(42)
# Simulate normalized ES futures prices with realistic volatility
returns = np.random.normal(0, 0.002, 448)  # 0.2% daily volatility
context = 1.0 * np.exp(np.cumsum(returns))  # Geometric random walk

inputs = [context.tolist()]
freq = [0]  # High frequency

# Get forecast
print("\n📈 Generating forecast...")
forecast_output = tfm.forecast(inputs, freq)

# Check what we got
print(f"   Output type: {type(forecast_output)}")
print(f"   Number of outputs: {len(forecast_output)}")

if isinstance(forecast_output, tuple) and len(forecast_output) == 2:
    point_forecast, extra_output = forecast_output
    print(f"   Point forecast shape: {point_forecast.shape}")
    print(f"   Extra output: {extra_output}")
else:
    point_forecast = forecast_output
    extra_output = None

# Create comprehensive visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Time steps
context_time = np.arange(len(context))
forecast_time = np.arange(len(context), len(context) + 64)

# Plot 1: Point forecast
ax1.plot(context_time, context, 'b-', label='Historical Data', alpha=0.8)
ax1.plot(forecast_time, point_forecast[0], 'r-', label='Point Forecast', linewidth=2)
ax1.axvline(x=len(context), color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Normalized Price')
ax1.set_title('TimesFM Point Forecast')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Simulated prediction intervals
# Since quantiles may not be available, we'll simulate them based on historical volatility
print("\n2️⃣ Creating prediction intervals based on historical volatility...")

# Calculate historical volatility
returns_hist = np.diff(np.log(context))
volatility = np.std(returns_hist)
print(f"   Historical volatility: {volatility:.4f}")

# Generate prediction intervals using volatility scaling
# Variance increases with forecast horizon
horizon = np.arange(1, 65)
std_forecast = volatility * np.sqrt(horizon)  # Standard deviation grows with sqrt(time)

# Create quantile predictions
median = point_forecast[0]
q_10 = median - 1.28 * std_forecast * median  # 10th percentile
q_25 = median - 0.67 * std_forecast * median  # 25th percentile  
q_75 = median + 0.67 * std_forecast * median  # 75th percentile
q_90 = median + 1.28 * std_forecast * median  # 90th percentile

# Plot with prediction intervals
ax2.plot(context_time, context, 'b-', label='Historical Data', alpha=0.8)
ax2.plot(forecast_time, median, 'r-', label='Median Forecast', linewidth=2)

# Confidence intervals
ax2.fill_between(forecast_time, q_10, q_90, alpha=0.2, color='red', label='80% CI')
ax2.fill_between(forecast_time, q_25, q_75, alpha=0.3, color='red', label='50% CI')

ax2.axvline(x=len(context), color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Normalized Price')
ax2.set_title('Forecast with Volatility-Based Prediction Intervals')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('timesfm_prediction_intervals.png', dpi=150)
print("\n✅ Plot saved to timesfm_prediction_intervals.png")

# Display prediction ranges
print("\n📊 Prediction Ranges:")
for t in [1, 10, 30, 64]:
    t_idx = t - 1
    print(f"\n   Time t+{t}:")
    print(f"      Point forecast: {point_forecast[0][t_idx]:.4f}")
    print(f"      80% interval: [{q_10[t_idx]:.4f}, {q_90[t_idx]:.4f}]")
    print(f"      50% interval: [{q_25[t_idx]:.4f}, {q_75[t_idx]:.4f}]")
    print(f"      Range width: {(q_90[t_idx] - q_10[t_idx]):.4f}")

print("\n💡 Key Insights:")
print("""
1. TimesFM provides point forecasts by default
2. Quantile predictions are experimental and need fine-tuning
3. For production use, you can:
   - Use historical volatility to create prediction intervals
   - Fine-tune the model with quantile loss functions
   - Use ensemble methods for uncertainty estimation
   
4. Trading applications:
   - Buy when price < 25th percentile prediction
   - Sell when price > 75th percentile prediction
   - Use wider intervals for risk management
""")

# Alternative: Monte Carlo approach
print("\n3️⃣ Alternative: Monte Carlo Prediction Intervals")
print("   Running 100 forecasts with input perturbations...")

# Generate multiple forecasts with slightly perturbed inputs
n_simulations = 100
forecasts = []

for i in range(n_simulations):
    # Add small noise to context
    noise = np.random.normal(0, 0.0001, len(context))
    perturbed_context = context * (1 + noise)
    
    # Get forecast
    forecast, _ = tfm.forecast([perturbed_context.tolist()], freq)
    forecasts.append(forecast[0])

forecasts = np.array(forecasts)

# Calculate quantiles from Monte Carlo simulations
mc_q10 = np.percentile(forecasts, 10, axis=0)
mc_q25 = np.percentile(forecasts, 25, axis=0)
mc_q50 = np.percentile(forecasts, 50, axis=0)
mc_q75 = np.percentile(forecasts, 75, axis=0)
mc_q90 = np.percentile(forecasts, 90, axis=0)

print(f"\n   Monte Carlo intervals at t+10:")
print(f"      10th percentile: {mc_q10[9]:.4f}")
print(f"      Median: {mc_q50[9]:.4f}")
print(f"      90th percentile: {mc_q90[9]:.4f}")

print("\n✅ Complete! See timesfm_prediction_intervals.png for visualization")
