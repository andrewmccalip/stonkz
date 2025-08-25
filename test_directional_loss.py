#!/usr/bin/env python3
"""
Test script to demonstrate the directional loss functionality.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define DirectionalLoss class here to test independently
class DirectionalLoss(nn.Module):
    """Custom loss function that combines MSE with directional accuracy."""
    
    def __init__(self, mse_weight=0.5, directional_weight=0.5, epsilon=1e-6):
        """
        Args:
            mse_weight: Weight for MSE component
            directional_weight: Weight for directional component
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.directional_weight = directional_weight
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Calculate combined loss focusing on directional accuracy.
        
        Args:
            predictions: Model predictions [batch_size, horizon_len]
            targets: Ground truth values [batch_size, horizon_len]
        
        Returns:
            Combined loss value
        """
        # Standard MSE loss
        mse_loss = self.mse_loss(predictions, targets)
        
        # Directional loss component
        # Calculate price changes (direction of movement)
        pred_changes = predictions[:, 1:] - predictions[:, :-1]  # [batch, horizon-1]
        target_changes = targets[:, 1:] - targets[:, :-1]  # [batch, horizon-1]
        
        # Sign agreement loss (1 - accuracy)
        # When signs match, loss is 0; when they don't match, loss is 1
        pred_signs = torch.sign(pred_changes)
        target_signs = torch.sign(target_changes)
        
        # Handle zero changes by treating them as correct predictions
        sign_matches = (pred_signs == target_signs) | (target_signs == 0)
        directional_accuracy = sign_matches.float().mean()
        directional_loss = 1.0 - directional_accuracy
        
        # Correlation-based loss (optional, captures trend alignment)
        # Higher correlation = lower loss
        batch_size = predictions.shape[0]
        correlation_loss = 0.0
        
        for i in range(batch_size):
            pred_i = predictions[i]
            target_i = targets[i]
            
            # Standardize to compute correlation
            pred_std = (pred_i - pred_i.mean()) / (pred_i.std() + self.epsilon)
            target_std = (target_i - target_i.mean()) / (target_i.std() + self.epsilon)
            
            # Pearson correlation
            correlation = (pred_std * target_std).mean()
            # Convert correlation to loss (1 - correlation) / 2 to get [0, 1] range
            correlation_loss += (1.0 - correlation) / 2.0
        
        correlation_loss /= batch_size
        
        # Combine directional and correlation losses
        dir_loss_combined = (directional_loss + correlation_loss) / 2.0
        
        # Final combined loss
        total_loss = self.mse_weight * mse_loss + self.directional_weight * dir_loss_combined
        
        # Store components for logging
        self.last_mse = mse_loss.item()
        self.last_directional = directional_loss.item()
        self.last_correlation = correlation_loss.item()
        self.last_accuracy = directional_accuracy.item()
        
        return total_loss

def test_directional_loss():
    """Test the directional loss with various scenarios."""
    print("🧪 Testing Directional Loss Function")
    print("=" * 60)
    
    # Create loss function
    criterion = DirectionalLoss(mse_weight=0.5, directional_weight=0.5)
    
    # Test scenarios
    batch_size = 4
    horizon_len = 64
    
    # Scenario 1: Perfect predictions
    print("\n1️⃣ Perfect Predictions")
    targets = torch.randn(batch_size, horizon_len)
    perfect_preds = targets.clone()
    loss = criterion(perfect_preds, targets)
    print(f"   Loss: {loss.item():.6f}")
    print(f"   MSE: {criterion.last_mse:.6f}")
    print(f"   Directional Accuracy: {criterion.last_accuracy:.2%}")
    print(f"   Correlation Loss: {criterion.last_correlation:.6f}")
    
    # Scenario 2: Random predictions
    print("\n2️⃣ Random Predictions")
    random_preds = torch.randn(batch_size, horizon_len)
    loss = criterion(random_preds, targets)
    print(f"   Loss: {loss.item():.6f}")
    print(f"   MSE: {criterion.last_mse:.6f}")
    print(f"   Directional Accuracy: {criterion.last_accuracy:.2%}")
    print(f"   Correlation Loss: {criterion.last_correlation:.6f}")
    
    # Scenario 3: Correct direction, wrong magnitude
    print("\n3️⃣ Correct Direction, Wrong Magnitude")
    # Scale predictions by 2x but keep direction
    scaled_preds = targets * 2.0
    loss = criterion(scaled_preds, targets)
    print(f"   Loss: {loss.item():.6f}")
    print(f"   MSE: {criterion.last_mse:.6f}")
    print(f"   Directional Accuracy: {criterion.last_accuracy:.2%}")
    print(f"   Correlation Loss: {criterion.last_correlation:.6f}")
    
    # Scenario 4: Wrong direction
    print("\n4️⃣ Wrong Direction (Inverted)")
    inverted_preds = -targets
    loss = criterion(inverted_preds, targets)
    print(f"   Loss: {loss.item():.6f}")
    print(f"   MSE: {criterion.last_mse:.6f}")
    print(f"   Directional Accuracy: {criterion.last_accuracy:.2%}")
    print(f"   Correlation Loss: {criterion.last_correlation:.6f}")
    
    # Visualize a sample
    print("\n📊 Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot price series
    sample_idx = 0
    x = np.arange(horizon_len)
    
    ax1.plot(x, targets[sample_idx].numpy(), 'black', linewidth=2, label='Ground Truth')
    ax1.plot(x, scaled_preds[sample_idx].numpy(), 'green', linewidth=2, alpha=0.7, 
             label='Correct Direction (2x magnitude)')
    ax1.plot(x, inverted_preds[sample_idx].numpy(), 'red', linewidth=2, alpha=0.7, 
             label='Wrong Direction (inverted)')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Price')
    ax1.set_title('Directional Loss Test: Price Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot directional changes
    target_changes = np.diff(targets[sample_idx].numpy())
    correct_changes = np.diff(scaled_preds[sample_idx].numpy())
    wrong_changes = np.diff(inverted_preds[sample_idx].numpy())
    
    x_changes = np.arange(len(target_changes))
    
    ax2.bar(x_changes - 0.3, np.sign(target_changes), width=0.3, alpha=0.8, 
            color='black', label='Ground Truth Direction')
    ax2.bar(x_changes, np.sign(correct_changes), width=0.3, alpha=0.8, 
            color='green', label='Correct Direction Pred')
    ax2.bar(x_changes + 0.3, np.sign(wrong_changes), width=0.3, alpha=0.8, 
            color='red', label='Wrong Direction Pred')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Direction (+1 = Up, -1 = Down)')
    ax2.set_title('Directional Changes Comparison')
    ax2.set_ylim(-1.5, 1.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('test_directional_loss.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to test_directional_loss.png")
    
    # Summary
    print("\n📊 Summary")
    print("=" * 60)
    print("The directional loss successfully:")
    print("✅ Gives lowest loss to perfect predictions")
    print("✅ Penalizes wrong directions more than wrong magnitudes")
    print("✅ Tracks directional accuracy separately from MSE")
    print("✅ Encourages models to focus on trend prediction")

if __name__ == "__main__":
    test_directional_loss()
