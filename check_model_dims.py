#!/usr/bin/env python3
"""
Check model dimensions and configuration to debug the 12-step zero issue.
"""
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Try to import the model
try:
    from pytorch_timesfm_model import TimesFMModel
    
    print("🔍 Checking TimesFM Model Configuration")
    print("=" * 60)
    
    # Initialize model with same config as training
    CONTEXT_LENGTH = 448
    HORIZON_LENGTH = 64
    INPUT_PATCH_LEN = 32
    
    model = TimesFMModel(
        context_len=CONTEXT_LENGTH,
        horizon_len=HORIZON_LENGTH,
        input_patch_len=INPUT_PATCH_LEN
    )
    
    print(f"\n📊 Model Configuration:")
    print(f"   Context Length: {CONTEXT_LENGTH}")
    print(f"   Horizon Length: {HORIZON_LENGTH}")
    print(f"   Patch Length: {INPUT_PATCH_LEN}")
    print(f"   Context Patches: {CONTEXT_LENGTH // INPUT_PATCH_LEN}")
    
    # Check model attributes
    print(f"\n🔍 Model Attributes:")
    for attr in ['context_len', 'horizon_len', 'input_patch_len', 'output_len', 
                 'max_len', 'num_layers', 'd_model', 'num_heads']:
        if hasattr(model, attr):
            print(f"   {attr}: {getattr(model, attr)}")
    
    # Check output projection
    if hasattr(model, 'output_projection'):
        proj = model.output_projection
        print(f"\n📊 Output Projection:")
        print(f"   Type: {type(proj)}")
        if hasattr(proj, 'out_features'):
            print(f"   Output features: {proj.out_features}")
    
    # Test forward pass with dummy data
    print(f"\n🧪 Testing Forward Pass:")
    batch_size = 1
    num_patches = CONTEXT_LENGTH // INPUT_PATCH_LEN
    dummy_input = torch.randn(batch_size, num_patches, 64)  # 64 features per patch
    dummy_freq = torch.zeros(batch_size, dtype=torch.long)
    
    with torch.no_grad():
        output = model(dummy_input, dummy_freq)
    
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected output shape: [{batch_size}, {HORIZON_LENGTH}]")
    
    if output.shape[1] != HORIZON_LENGTH:
        print(f"   ⚠️ WARNING: Output dimension mismatch!")
        print(f"   Expected {HORIZON_LENGTH}, got {output.shape[1]}")
    
    # Check output values
    output_np = output[0].numpy()
    print(f"\n📊 Output Analysis:")
    print(f"   First 15 values: {output_np[:15]}")
    print(f"   Values 10-15: {output_np[10:15] if len(output_np) > 14 else 'N/A'}")
    print(f"   Last 10 values: {output_np[-10:] if len(output_np) >= 10 else output_np}")
    
    # Check for zeros
    near_zeros = sum(1 for x in output_np if abs(x) < 1e-6)
    if near_zeros > 0:
        print(f"   ⚠️ Found {near_zeros} near-zero values!")
        first_zero = next((i for i, x in enumerate(output_np) if abs(x) < 1e-6), -1)
        if first_zero >= 0:
            print(f"   First zero at index: {first_zero}")
    
except ImportError as e:
    print(f"❌ Could not import TimesFMModel: {e}")
    print("\n💡 The model file might be missing or have import errors.")
    print("   This is likely why predictions drop to zero - the model isn't properly initialized!")

print("\n✅ Model dimension check complete!")
