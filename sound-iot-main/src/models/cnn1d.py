"""1D CNN for MFCC sequence features."""
from __future__ import annotations

import torch
from torch import nn


class CNN1DClassifier(nn.Module):
    """1D CNN for MFCC sequences matching Keras architecture.
    
    Input shape: (batch, channels, n_mfcc, time) = (batch, 1, 128, time)
    Converts to 1D and applies Conv1D along time dimension.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        
        # Input: (batch, 1, 128, time) -> reshape to (batch, 128, time)
        # Apply Conv1D along time dimension with 128 input channels
        
        self.features = nn.Sequential(
            # First Conv1D: 128 input channels -> 128 filters (matches conv1d output: 128 filters)
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Halve time dimension
            
            # Second Conv1D: 128 -> 64 filters (matches conv1d_1 output: 64 filters)
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.Dropout(0.25),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Third Conv1D: 64 -> 32 filters (matches conv1d_2 output: 32 filters)
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.Dropout(0.25),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Fourth Conv1D: 32 -> 16 filters (matches conv1d_3 output: 16 filters)
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.Dropout(0.25),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling to get fixed size: flatten output should be 512 (16 * 32)
        # Based on Keras: flatten output is 512, so 16 filters * 32 time frames
        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)  # Output: (batch, 16, 32) = 512 when flattened
        
        # Classifier layers (matches dense_7 and dense_8)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (batch, 512)
            nn.Linear(16 * 32, 1024),  # 512 -> 1024 (matches dense_7)
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),  # 1024 -> 10 (matches dense_8)
        )
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: (batch, channels, n_mfcc, time) = (batch, 1, 128, time)
        # Reshape to (batch, n_mfcc, time) for Conv1D
        if sequence.dim() == 4:
            batch, channels, n_mfcc, time = sequence.shape
            x = sequence.squeeze(1)  # Remove channel dim: (batch, 128, time)
        elif sequence.dim() == 3:
            x = sequence  # Already (batch, 128, time)
        else:
            raise ValueError(f"Unexpected sequence shape: {sequence.shape}")
        
        x = self.features(x)  # (batch, 16, variable_time)
        x = self.adaptive_pool(x)  # (batch, 16, 32) - fixed size
        x = self.classifier(x)  # (batch, num_classes)
        return x

