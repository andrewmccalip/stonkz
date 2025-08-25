# Configuration Alignment with good_timefm_stock_working.py

## Summary
Updated `pytorch_timesfm_finetune.py` to match the time window and configuration parameters from the working example script.

## Key Parameter Alignments

### Time Windows
| Parameter | Old Value | New Value | Working Example | Notes |
|-----------|-----------|-----------|-----------------|-------|
| CONTEXT_LENGTH | 512 | **448** | 448 | ~7.5 hours of minute data |
| HORIZON_LENGTH | 128 | **64** | 64 | ~1 hour prediction window |
| INPUT_PATCH_LEN | 32 | 32 | N/A | 448/32 = 14 patches |

### Data Processing
- **Normalized Prices**: ✅ Both scripts use normalized prices
- **Data Source**: Same ES futures data file
- **Frequency**: Minute-level data (1m bars)

### Model Configuration
- **Official TimesFM Model**: 
  - Backend: Auto-detects GPU/CPU (same as working example)
  - Model: `google/timesfm-1.0-200m-pytorch`
  - Context capability: 512 (but we use 448 to match)
  - Horizon capability: 128 (but we use first 64 to match)

## Benefits of Alignment

1. **Direct Comparability**: Results from fine-tuning can be directly compared with the working example's predictions

2. **Consistent Time Horizons**: Both scripts now work with:
   - 7.5 hours of historical context
   - 1 hour prediction horizon

3. **Normalized Data**: Both use price normalization for better model performance

4. **Same Model Architecture**: Using the official TimesFM as baseline ensures we're comparing apples to apples

## Verification Output

When you run the fine-tuning script, you'll see:
```
📈 Prediction Configuration (matching good_timefm_stock_working.py):
   Context window: 448 minutes (~7.5 hours)
   Prediction horizon: 64 minutes (~1.1 hours)
   Input patch length: 32 minutes
   Number of patches: 14
   Using normalized prices: True (like working example)
```

This confirms the configuration matches the working example.
