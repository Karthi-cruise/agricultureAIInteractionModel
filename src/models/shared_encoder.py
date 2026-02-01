"""
Shared Encoder Module

Temporal encoder (LSTM/CNN/Transformer) that learns shared representations
from aligned satellite, weather, soil, and water features.
"""

import torch
import torch.nn as nn
from typing import Literal


class SharedEncoder(nn.Module):
    """Shared temporal encoder for multi-task learning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        encoder_type: Literal["lstm", "cnn", "transformer"] = "lstm",
        bidirectional: bool = False,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        if encoder_type == "lstm":
            self.encoder = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.output_dim = hidden_dim * self.num_directions
        elif encoder_type == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.output_dim = hidden_dim
        elif encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.proj = nn.Linear(input_dim, hidden_dim)
            self.output_dim = hidden_dim
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, output_dim) - shared representation
        """
        if self.encoder_type == "lstm":
            _, (h_n, _) = self.encoder(x)
            h = h_n[-1]
            if self.num_directions == 2:
                h = torch.cat([h[-2], h[-1]], dim=-1)
            return h
        elif self.encoder_type == "cnn":
            x = x.transpose(1, 2)
            h = self.encoder(x).squeeze(-1)
            return h
        else:
            h = self.encoder(x)
            h = h.mean(dim=1)
            return self.proj(h)
