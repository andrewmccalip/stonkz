"""
PyTorch TimesFM Model with Scale Normalization
Handles scale mismatch between predictions and targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledTimesFMModel(nn.Module):
    """
    TimesFM model with improved scale handling for stock price prediction.
    """
    
    def __init__(self, context_len=448, horizon_len=64, input_patch_len=32):
        super().__init__()
        
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.input_patch_len = input_patch_len
        self.num_patches = context_len // input_patch_len
        
        # Architecture parameters
        hidden_dim = 256
        
        # Patch embedding
        self.patch_embedding = nn.Linear(64, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * self.num_patches, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.horizon_len)
        )
        
        # Scale normalization layers
        self.output_scale = nn.Parameter(torch.ones(1) * 0.01)  # Start with small scale
        self.output_bias = nn.Parameter(torch.ones(1))  # Start near 1.0 for normalized prices
        
        # Adaptive scale learning
        self.scale_predictor = nn.Sequential(
            nn.Linear(hidden_dim * self.num_patches, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Predict scale and bias
        )
        
    def forward(self, x: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with scale normalization.
        
        Args:
            x: [batch_size, num_patches, patch_features]
            freq: [batch_size] (unused in this version)
            
        Returns:
            [batch_size, horizon_len]
        """
        batch_size = x.shape[0]
        
        # Embed patches
        x_embed = self.patch_embedding(x)  # [batch, num_patches, hidden]
        
        # Add positional encoding
        x_embed = x_embed + self.positional_encoding
        
        # Transformer
        x_transformed = self.transformer(x_embed)  # [batch, num_patches, hidden]
        
        # Flatten for projection
        x_flat = x_transformed.reshape(batch_size, -1)  # [batch, num_patches * hidden]
        
        # Generate raw predictions
        raw_output = self.output_projection(x_flat)  # [batch, horizon_len]
        
        # Method 1: Fixed scale normalization
        # Constrain output to reasonable range using tanh
        normalized_output = torch.tanh(raw_output) * self.output_scale + self.output_bias
        
        # Method 2: Adaptive scale based on input
        scale_params = self.scale_predictor(x_flat)  # [batch, 2]
        adaptive_scale = torch.sigmoid(scale_params[:, 0:1]) * 0.1  # [batch, 1], range 0-0.1
        adaptive_bias = scale_params[:, 1:2]  # [batch, 1]
        
        # Combine both methods
        output = normalized_output * (1.0 + adaptive_scale) + adaptive_bias * 0.01
        
        return output
    
    def load_pretrained_weights(self, checkpoint_path: str) -> bool:
        """Load pre-trained weights if available."""
        print("ℹ️ Using random initialization for scaled model")
        return True


class ResidualTimesFMModel(nn.Module):
    """
    TimesFM model that predicts residuals from the last context value.
    This naturally keeps predictions in the right scale.
    """
    
    def __init__(self, context_len=448, horizon_len=64, input_patch_len=32):
        super().__init__()
        
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.input_patch_len = input_patch_len
        self.num_patches = context_len // input_patch_len
        
        # Architecture parameters
        hidden_dim = 256
        
        # Patch embedding
        self.patch_embedding = nn.Linear(64, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Residual projection - predicts changes from last value
        self.residual_projection = nn.Sequential(
            nn.Linear(hidden_dim * self.num_patches, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.horizon_len)
        )
        
        # Small scale for residuals
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.001)
        
    def forward(self, x: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting residuals.
        
        Args:
            x: [batch_size, num_patches, patch_features]
            freq: [batch_size]
            
        Returns:
            [batch_size, horizon_len]
        """
        batch_size = x.shape[0]
        
        # Get last value from context for residual prediction
        # Assuming the last patch contains the most recent values
        last_patch = x[:, -1, :]  # [batch, 64]
        last_value = last_patch[:, -1:].mean(dim=1, keepdim=True)  # [batch, 1]
        
        # Embed patches
        x_embed = self.patch_embedding(x)  # [batch, num_patches, hidden]
        
        # Add positional encoding
        x_embed = x_embed + self.positional_encoding
        
        # Transformer
        x_transformed = self.transformer(x_embed)  # [batch, num_patches, hidden]
        
        # Flatten and project to residuals
        x_flat = x_transformed.reshape(batch_size, -1)
        residuals = self.residual_projection(x_flat)  # [batch, horizon_len]
        
        # Scale residuals and add to last value
        scaled_residuals = torch.tanh(residuals) * self.residual_scale
        
        # Broadcast last value and add residuals
        output = last_value + scaled_residuals
        
        return output
    
    def load_pretrained_weights(self, checkpoint_path: str) -> bool:
        """Load pre-trained weights if available."""
        print("ℹ️ Using random initialization for residual model")
        return True


# Export the model to use
TimesFMModel = ResidualTimesFMModel  # Use residual model by default
