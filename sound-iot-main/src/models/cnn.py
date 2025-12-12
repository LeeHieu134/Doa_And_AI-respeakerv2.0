"""2D CNN for MFCC-style sequence tensors matching Keras architecture."""
from __future__ import annotations

import torch
from torch import nn


class CNNClassifier(nn.Module):
    """2D CNN matching Keras architecture.
    
    Architecture matches:
    - Conv2D(16 filters) -> MaxPool2D
    - Conv2D(128 filters) -> MaxPool2D -> Dropout
    - Flatten -> Dense(1024) -> Dense(10)
    
    Input: (batch, 1, 128, time) - single channel, 128 MFCC coefficients
    """
    
    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # First Conv2D: 1 -> 16 filters
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 16, 64, time/2)
            
            # Second Conv2D: 16 -> 128 filters
            nn.Conv2d(16, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 128, 32, time/4)
            nn.Dropout(0.25),
        )
        
        # Adaptive pooling to get fixed size before flattening
        # Output shape from features: (batch, 128, variable_height, variable_width)
        # We'll use adaptive pooling to get (batch, 128, 2, 2) = 512 when flattened
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 2))  # (batch, 128, 4, 2) = 1024
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (batch, 1024)
            nn.Linear(128 * 4 * 2, 1024),  # 1024 -> 1024 (matches dense_9)
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),  # 1024 -> 10 (matches dense_10)
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: (batch, channels, mfcc, time) = (batch, 1, 128, time)
        # If 3D tensor (batch, mfcc, time), add channel dimension
        if sequence.dim() == 3:
            sequence = sequence.unsqueeze(1)  # Add channel dim: (batch, 1, 128, time)
        
        x = self.features(sequence)  # (batch, 128, variable, variable)
        x = self.adaptive_pool(x)  # (batch, 128, 4, 2) - fixed size
        x = self.classifier(x)  # (batch, num_classes)
        return x







