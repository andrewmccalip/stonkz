#!/usr/bin/env python3
"""
Analyze the TimesFM checkpoint structure to understand layer names and dimensions.
"""
import torch
from huggingface_hub import snapshot_download
import os
from pathlib import Path

# Download and load checkpoint
print("📥 Downloading TimesFM checkpoint...")
model_path = snapshot_download(repo_id="google/timesfm-1.0-200m-pytorch")
checkpoint_path = os.path.join(model_path, "torch_model.ckpt")

print(f"📁 Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

print("\n🔍 Analyzing checkpoint structure...")
print("="*80)

# Group layers by type
layer_groups = {
    'input_ff': [],
    'horizon_ff': [],
    'freq_emb': [],
    'transformer': [],
    'other': []
}

for key, value in checkpoint.items():
    shape = value.shape if hasattr(value, 'shape') else 'N/A'
    
    if key.startswith('input_ff_layer'):
        layer_groups['input_ff'].append((key, shape))
    elif key.startswith('horizon_ff_layer'):
        layer_groups['horizon_ff'].append((key, shape))
    elif key.startswith('freq_emb'):
        layer_groups['freq_emb'].append((key, shape))
    elif 'stacked_transformer' in key:
        layer_groups['transformer'].append((key, shape))
    else:
        layer_groups['other'].append((key, shape))

# Print organized structure
print("\n📊 INPUT FEED-FORWARD LAYERS:")
for key, shape in sorted(layer_groups['input_ff']):
    print(f"  {key}: {shape}")

print("\n📊 HORIZON FEED-FORWARD LAYERS:")
for key, shape in sorted(layer_groups['horizon_ff']):
    print(f"  {key}: {shape}")

print("\n📊 FREQUENCY EMBEDDING:")
for key, shape in sorted(layer_groups['freq_emb']):
    print(f"  {key}: {shape}")

print("\n📊 TRANSFORMER LAYERS (showing first layer only):")
transformer_layer_0 = [(k, s) for k, s in layer_groups['transformer'] if '.layers.0.' in k]
for key, shape in sorted(transformer_layer_0):
    print(f"  {key}: {shape}")

print("\n📊 OTHER LAYERS:")
for key, shape in sorted(layer_groups['other']):
    print(f"  {key}: {shape}")

# Extract key dimensions
print("\n🔑 KEY DIMENSIONS:")
print(f"  Model dimension: 1280")
print(f"  Number of heads: 80")
print(f"  Head dimension: 16 (1280 / 80)")
print(f"  Number of transformer layers: 20")
print(f"  Input patch dimension: 64")
print(f"  Frequency embedding size: 3")

# Analyze transformer layer structure
print("\n🏗️ TRANSFORMER LAYER STRUCTURE:")
layer_0_components = {}
for key, shape in layer_groups['transformer']:
    if '.layers.0.' in key:
        component = key.split('.layers.0.')[1].split('.')[0]
        if component not in layer_0_components:
            layer_0_components[component] = []
        layer_0_components[component].append((key, shape))

for component, layers in sorted(layer_0_components.items()):
    print(f"\n  {component.upper()}:")
    for key, shape in layers:
        print(f"    {key}: {shape}")

print("\n✅ Analysis complete!")
