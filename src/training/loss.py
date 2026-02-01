"""
Multi-Task Loss Module

L = α·CropLoss + β·SoilLoss + γ·WaterLoss + ConstraintPenalty
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from .constraints import ConstraintValidator


class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss with optional constraint penalty."""

    def __init__(
        self,
        weights: Dict[str, float] = None,
        constraint_weight: float = 0.1,
        use_constraints: bool = True,
    ):
        super().__init__()
        self.weights = weights or {"crop": 0.4, "soil": 0.3, "water": 0.3}
        self.constraint_weight = constraint_weight
        self.use_constraints = use_constraints
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        aux: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple:
        """
        Args:
            pred: {crop, soil, water} predictions
            target: {crop, soil, water} targets
            aux: optional aux inputs for constraint computation (rainfall, irrigation, etc.)
        Returns:
            (total_loss, loss_dict)
        """
        losses = {}
        total = 0.0

        for task, w in self.weights.items():
            if task in pred and task in target:
                tgt = target[task]
                if tgt.dim() == 1:
                    tgt = tgt.unsqueeze(-1)
                loss = self.mse(pred[task], tgt)
                losses[task] = loss.item()
                total = total + w * loss

        if self.use_constraints and aux is not None:
            penalty = ConstraintValidator.compute_penalty(pred, aux)
            losses["constraint"] = penalty.item()
            total = total + self.constraint_weight * penalty

        return total, losses
