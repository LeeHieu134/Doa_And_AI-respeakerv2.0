"""Fully-connected ANN matching debug.ipynb architecture."""
from __future__ import annotations

import torch
from torch import nn


class DNNClassifier(nn.Module):
    """ANN (Artificial Neural Network) matching debug.ipynb architecture.
    
    Architecture matches debug.ipynb ANN_Model:
    - Input: 128 MFCC features
    - Linear(128, 1000) -> ReLU
    - Linear(1000, 750) -> ReLU
    - Linear(750, 500) -> ReLU
    - Linear(500, 250) -> ReLU
    - Linear(250, 100) -> ReLU
    - Linear(100, 50) -> ReLU
    - Linear(50, num_classes)
    
    No BatchNorm, no Dropout - simple Linear + ReLU layers only.
    """

    def __init__(
        self,
        input_dim: int = 128,  # MFCC features from extractor
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        
        # Match exact architecture from debug.ipynb
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 750),
            nn.ReLU(),
            nn.Linear(750, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes)
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        # Flatten input to (batch_size, input_dim)
        x = stats.view(stats.size(0), -1)
        return self.model(x)








