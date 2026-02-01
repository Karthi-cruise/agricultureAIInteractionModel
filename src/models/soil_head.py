"""
Soil Nutrient Head

Predicts soil nutrient status (NPK, organic carbon) from shared encoder.
"""

import torch
import torch.nn as nn


class SoilHead(nn.Module):
    """Task-specific head for soil nutrient prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        output_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
