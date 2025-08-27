"""
Uncertainty-aware loss functions for time series prediction.
These losses encourage the model to predict ranges/distributions rather than just point estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuantileLoss(nn.Module):
    """
    Quantile loss for predicting multiple quantiles simultaneously.
    The model outputs multiple predictions for different quantiles.
    """
    
    def __init__(self, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        self.n_quantiles = len(quantiles)
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch_size, horizon_len, n_quantiles]
            targets: [batch_size, horizon_len]
        """
        # Expand targets to match quantile dimension
        targets = targets.unsqueeze(-1).expand(-1, -1, self.n_quantiles)
        
        # Calculate quantile loss (pinball loss)
        errors = targets - predictions
        quantiles = self.quantiles.view(1, 1, -1).to(predictions.device)
        
        loss = torch.where(
            errors >= 0,
            quantiles * errors,
            (quantiles - 1) * errors
        )
        
        return loss.mean()


class GaussianNLLLoss(nn.Module):
    """
    Negative log-likelihood loss assuming Gaussian distribution.
    Model predicts mean and variance for each time step.
    """
    
    def __init__(self, min_variance=1e-6):
        super().__init__()
        self.min_variance = min_variance
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch_size, horizon_len, 2] (mean and log_variance)
            targets: [batch_size, horizon_len]
        """
        mean = predictions[..., 0]
        log_var = predictions[..., 1]
        
        # Ensure variance is positive
        var = torch.exp(log_var) + self.min_variance
        
        # Gaussian NLL
        loss = 0.5 * (torch.log(2 * np.pi * var) + (targets - mean)**2 / var)
        
        return loss.mean()


class IntervalLoss(nn.Module):
    """
    Loss for predicting prediction intervals (lower and upper bounds).
    Penalizes intervals that don't contain the target and rewards narrow intervals.
    """
    
    def __init__(self, coverage_weight=0.8, width_weight=0.2, target_coverage=0.8):
        super().__init__()
        self.coverage_weight = coverage_weight
        self.width_weight = width_weight
        self.target_coverage = target_coverage
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch_size, horizon_len, 2] (lower and upper bounds)
            targets: [batch_size, horizon_len]
        """
        lower = predictions[..., 0]
        upper = predictions[..., 1]
        
        # Coverage loss: penalize when target is outside interval
        coverage_loss = torch.relu(lower - targets) + torch.relu(targets - upper)
        
        # Width loss: penalize wide intervals
        width = upper - lower
        width_loss = width.mean()
        
        # Coverage calibration: encourage target coverage rate
        in_interval = ((targets >= lower) & (targets <= upper)).float()
        coverage_rate = in_interval.mean()
        calibration_loss = (coverage_rate - self.target_coverage)**2
        
        total_loss = (self.coverage_weight * coverage_loss.mean() + 
                     self.width_weight * width_loss +
                     0.1 * calibration_loss)
        
        # Store metrics
        self.last_coverage = coverage_rate.item()
        self.last_width = width.mean().item()
        
        return total_loss


class UncertaintyAwareDirectionalLoss(nn.Module):
    """
    Combines directional accuracy with uncertainty estimation.
    Model predicts direction and confidence simultaneously.
    """
    
    def __init__(self, directional_weight=0.5, confidence_weight=0.3, mse_weight=0.2):
        super().__init__()
        self.directional_weight = directional_weight
        self.confidence_weight = confidence_weight
        self.mse_weight = mse_weight
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch_size, horizon_len, 3] (value, direction_logit, confidence)
            targets: [batch_size, horizon_len]
        """
        pred_values = predictions[..., 0]
        direction_logits = predictions[..., 1]
        confidence = torch.sigmoid(predictions[..., 2])
        
        # MSE loss
        mse_loss = F.mse_loss(pred_values, targets)
        
        # Directional loss
        true_direction = (targets[:, 1:] > targets[:, :-1]).float()
        pred_direction = torch.sigmoid(direction_logits[:, :-1])
        directional_loss = F.binary_cross_entropy(pred_direction, true_direction)
        
        # Confidence loss: high confidence when prediction is correct
        value_error = torch.abs(pred_values - targets)
        normalized_error = value_error / (targets.abs() + 1e-6)
        
        # Confidence should be high when error is low
        target_confidence = torch.exp(-normalized_error * 10)  # Maps error to confidence
        confidence_loss = F.mse_loss(confidence, target_confidence)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.directional_weight * directional_loss +
                     self.confidence_weight * confidence_loss)
        
        # Store metrics
        self.last_dir_accuracy = ((pred_direction > 0.5) == true_direction).float().mean().item()
        self.last_confidence = confidence.mean().item()
        
        return total_loss


class EnsembleLoss(nn.Module):
    """
    Loss for training ensemble predictions.
    Model outputs multiple predictions, encouraging diversity.
    """
    
    def __init__(self, n_models=5, diversity_weight=0.1):
        super().__init__()
        self.n_models = n_models
        self.diversity_weight = diversity_weight
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch_size, horizon_len, n_models]
            targets: [batch_size, horizon_len]
        """
        # Expand targets
        targets_expanded = targets.unsqueeze(-1).expand(-1, -1, self.n_models)
        
        # Individual model losses
        individual_losses = F.mse_loss(predictions, targets_expanded, reduction='none')
        mean_loss = individual_losses.mean()
        
        # Diversity loss: encourage different predictions
        pred_mean = predictions.mean(dim=-1, keepdim=True)
        diversity = (predictions - pred_mean).pow(2).mean()
        diversity_loss = -diversity  # Negative because we want to maximize diversity
        
        total_loss = mean_loss + self.diversity_weight * diversity_loss
        
        # Calculate ensemble prediction and its error
        ensemble_pred = predictions.mean(dim=-1)
        ensemble_error = F.mse_loss(ensemble_pred, targets)
        
        # Store metrics
        self.last_ensemble_error = ensemble_error.item()
        self.last_diversity = diversity.item()
        
        return total_loss


# Example model modifications for uncertainty

class UncertaintyTimesFMModel(nn.Module):
    """
    Modified TimesFM model that outputs uncertainty estimates.
    """
    
    def __init__(self, base_model, output_type='interval'):
        super().__init__()
        self.base_model = base_model
        self.output_type = output_type
        
        # Get base model output dimension
        hidden_dim = 256  # Adjust based on your model
        
        if output_type == 'interval':
            # Output lower and upper bounds
            self.uncertainty_head = nn.Linear(hidden_dim, 2)
        elif output_type == 'gaussian':
            # Output mean and log variance
            self.uncertainty_head = nn.Linear(hidden_dim, 2)
        elif output_type == 'quantile':
            # Output multiple quantiles
            self.uncertainty_head = nn.Linear(hidden_dim, 5)  # 5 quantiles
        elif output_type == 'ensemble':
            # Output multiple predictions
            self.uncertainty_head = nn.Linear(hidden_dim, 5)  # 5 models
            
    def forward(self, x, freq):
        # Get base features (before final projection)
        # This requires modifying the base model to return intermediate features
        features = self.base_model.get_features(x, freq)
        
        # Generate uncertainty outputs
        uncertainty = self.uncertainty_head(features)
        
        if self.output_type == 'interval':
            # Ensure upper > lower
            lower = uncertainty[..., 0]
            width = F.softplus(uncertainty[..., 1])
            upper = lower + width
            return torch.stack([lower, upper], dim=-1)
        else:
            return uncertainty
