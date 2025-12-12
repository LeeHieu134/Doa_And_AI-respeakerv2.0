"""LSTM-based classifier for sequential MFCC features."""
from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * direction_factor, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence shape: (batch, channels, n_mfcc, time) = (batch, 1, 128, time)
        # LSTM expects (batch, time, features)
        if sequence.dim() == 4:
            batch, channels, n_mfcc, time = sequence.shape
            # Remove channel dim and transpose: (batch, 1, 128, time) -> (batch, 128, time) -> (batch, time, 128)
            sequence = sequence.squeeze(1).transpose(1, 2)  # (batch, time, 128)
        elif sequence.dim() == 3:
            # Already (batch, n_mfcc, time), transpose to (batch, time, n_mfcc)
            sequence = sequence.transpose(1, 2)  # (batch, time, 128)
        else:
            raise ValueError(f"Unexpected sequence shape: {sequence.shape}")
        
        outputs, _ = self.lstm(sequence)  # (batch, time, hidden_size * direction_factor)
        last_output = outputs[:, -1, :]  # (batch, hidden_size * direction_factor)
        return self.classifier(last_output)







