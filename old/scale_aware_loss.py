"""
Scale-aware loss functions for time series prediction.
Helps address scale mismatch between predictions and targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaleAwareLoss(nn.Module):
    """
    Loss function that addresses scale mismatch in predictions.
    Combines multiple components to guide the model towards correct scale.
    """
    
    def __init__(self, 
                 mse_weight=0.3,
                 scale_weight=0.3, 
                 distribution_weight=0.2,
                 directional_weight=0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.scale_weight = scale_weight
        self.distribution_weight = distribution_weight
        self.directional_weight = directional_weight
        
    def forward(self, predictions, targets):
        """
        Calculate scale-aware loss.
        
        Args:
            predictions: [batch_size, horizon_len]
            targets: [batch_size, horizon_len]
        """
        batch_size = predictions.shape[0]
        
        # 1. Standard MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # 2. Scale matching loss
        # Penalize difference in mean and std
        pred_mean = predictions.mean(dim=1)
        target_mean = targets.mean(dim=1)
        mean_loss = F.mse_loss(pred_mean, target_mean)
        
        pred_std = predictions.std(dim=1)
        target_std = targets.std(dim=1)
        std_loss = F.mse_loss(pred_std, target_std)
        
        scale_loss = mean_loss + std_loss
        
        # 3. Distribution matching loss (using KL divergence approximation)
        # Normalize to compare distributions
        pred_normalized = (predictions - pred_mean.unsqueeze(1)) / (pred_std.unsqueeze(1) + 1e-6)
        target_normalized = (targets - target_mean.unsqueeze(1)) / (target_std.unsqueeze(1) + 1e-6)
        
        # Simple distribution matching using histogram-like approach
        distribution_loss = F.mse_loss(pred_normalized.sort(dim=1)[0], 
                                      target_normalized.sort(dim=1)[0])
        
        # 4. Directional accuracy loss
        pred_direction = (predictions[:, 1:] > predictions[:, :-1]).float()
        true_direction = (targets[:, 1:] > targets[:, :-1]).float()
        directional_loss = F.binary_cross_entropy(pred_direction, true_direction)
        
        # Combine all losses
        total_loss = (self.mse_weight * mse_loss + 
                     self.scale_weight * scale_loss +
                     self.distribution_weight * distribution_loss +
                     self.directional_weight * directional_loss)
        
        # Store components for logging
        self.last_components = {
            'mse': mse_loss.item(),
            'scale': scale_loss.item(),
            'distribution': distribution_loss.item(),
            'directional': directional_loss.item(),
            'pred_mean': pred_mean.mean().item(),
            'pred_std': pred_std.mean().item(),
            'target_mean': target_mean.mean().item(),
            'target_std': target_std.mean().item()
        }
        
        return total_loss


class RangePenaltyLoss(nn.Module):
    """
    Loss that penalizes predictions outside expected range.
    Useful when targets are normalized prices around 1.0.
    Default range is ±2% (0.98 to 1.02) for tighter constraints on stock price predictions.
    """
    
    def __init__(self, 
                 mse_weight=0.5,
                 range_weight=0.3,
                 directional_weight=0.2,
                 expected_center=1.0,
                 expected_range=0.02):
        super().__init__()
        self.mse_weight = mse_weight
        self.range_weight = range_weight
        self.directional_weight = directional_weight
        self.expected_center = expected_center
        self.expected_range = expected_range
        
    def forward(self, predictions, targets):
        """
        Calculate loss with range penalty.
        """
        # MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Range penalty - penalize predictions far from expected range
        lower_bound = self.expected_center - self.expected_range
        upper_bound = self.expected_center + self.expected_range
        
        # Soft penalty using smooth functions
        below_penalty = F.relu(lower_bound - predictions).mean()
        above_penalty = F.relu(predictions - upper_bound).mean()
        range_loss = below_penalty + above_penalty
        
        # Directional accuracy
        pred_direction = (predictions[:, 1:] > predictions[:, :-1]).float()
        true_direction = (targets[:, 1:] > targets[:, :-1]).float()
        directional_accuracy = (pred_direction == true_direction).float().mean()
        directional_loss = 1.0 - directional_accuracy
        
        # Combine losses
        total_loss = (self.mse_weight * mse_loss + 
                     self.range_weight * range_loss +
                     self.directional_weight * directional_loss)
        
        # Store metrics
        self.last_accuracy = directional_accuracy.item()
        self.last_range_violations = ((predictions < lower_bound) | 
                                     (predictions > upper_bound)).float().mean().item()
        
        return total_loss


class AdaptiveScaleLoss(nn.Module):
    """
    Loss that adapts to the scale of targets dynamically.
    """
    
    def __init__(self, base_loss='mse', scale_adaptation_rate=0.1):
        super().__init__()
        self.base_loss = base_loss
        self.scale_adaptation_rate = scale_adaptation_rate
        
        # Running statistics (will be updated during training)
        self.register_buffer('running_mean', torch.tensor(1.0))
        self.register_buffer('running_std', torch.tensor(0.01))
        self.register_buffer('n_updates', torch.tensor(0))
        
    def forward(self, predictions, targets):
        """
        Calculate adaptive loss.
        """
        # Update running statistics
        with torch.no_grad():
            target_mean = targets.mean()
            target_std = targets.std()
            
            if self.n_updates == 0:
                self.running_mean = target_mean
                self.running_std = target_std
            else:
                self.running_mean = (1 - self.scale_adaptation_rate) * self.running_mean + \
                                   self.scale_adaptation_rate * target_mean
                self.running_std = (1 - self.scale_adaptation_rate) * self.running_std + \
                                  self.scale_adaptation_rate * target_std
            
            self.n_updates += 1
        
        # Normalize predictions to match target scale
        pred_normalized = (predictions - predictions.mean()) / (predictions.std() + 1e-6)
        pred_rescaled = pred_normalized * self.running_std + self.running_mean
        
        # Calculate base loss on rescaled predictions
        if self.base_loss == 'mse':
            loss = F.mse_loss(pred_rescaled, targets)
        elif self.base_loss == 'mae':
            loss = F.l1_loss(pred_rescaled, targets)
        else:
            raise ValueError(f"Unknown base loss: {self.base_loss}")
        
        # Add small penalty for scale mismatch
        scale_penalty = 0.1 * (
            (predictions.mean() - self.running_mean).pow(2) +
            (predictions.std() - self.running_std).pow(2)
        )
        
        return loss + scale_penalty
