"""
Multi-Task Model

Combines shared encoder with task-specific heads for crop, soil, water.
"""

import torch
import torch.nn as nn
from .shared_encoder import SharedEncoder
from .crop_head import CropHead
from .soil_head import SoilHead
from .water_head import WaterHead


class CropWaterSoilMTL(nn.Module):
    """Full multi-task model: Shared Encoder + Crop / Soil / Water heads."""

    def __init__(
        self,
        input_dim: int,
        encoder_config: dict,
        crop_config: dict,
        soil_config: dict,
        water_config: dict,
    ):
        super().__init__()
        self.encoder = SharedEncoder(input_dim=input_dim, **encoder_config)
        enc_out = self.encoder.output_dim

        self.crop_head = CropHead(input_dim=enc_out, **crop_config)
        self.soil_head = SoilHead(input_dim=enc_out, **soil_config)
        self.water_head = WaterHead(input_dim=enc_out, **water_config)

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            dict with keys: crop, soil, water
        """
        h = self.encoder(x)
        return {
            "crop": self.crop_head(h),
            "soil": self.soil_head(h),
            "water": self.water_head(h),
        }
