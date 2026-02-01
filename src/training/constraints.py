"""
Domain Constraint Module

Enforces agricultural and hydrological constraints:
1. Low rainfall + high irrigation → groundwater cannot increase
2. Poor soil nutrients → crop health cannot spike
3. Water demand obeys evapotranspiration limits
"""

import torch
from typing import Dict, Optional


class ConstraintValidator:
    """Compute constraint violation penalties for multi-task predictions."""

    @staticmethod
    def groundwater_rainfall_constraint(
        gw_pred: torch.Tensor,
        rainfall: torch.Tensor,
        irrigation: torch.Tensor,
    ) -> torch.Tensor:
        """
        If rainfall is low and irrigation is high, groundwater should decrease.
        Penalize: max(0, gw_increase) when rainfall_down and irrigation_up.
        """
        if rainfall is None or irrigation is None:
            return torch.tensor(0.0, device=gw_pred.device)
        rain_low = (rainfall < rainfall.median()).float()
        irr_high = (irrigation > irrigation.median()).float()
        stress_indicator = rain_low * irr_high
        return (stress_indicator * torch.relu(gw_pred - 0.5)).mean()

    @staticmethod
    def soil_crop_constraint(crop_pred: torch.Tensor, soil_pred: torch.Tensor) -> torch.Tensor:
        """
        Crop health cannot exceed what soil nutrients can support.
        soil_pred: (N, 4) for NPK+OC - use mean as soil quality proxy
        """
        soil_quality = soil_pred.mean(dim=-1, keepdim=True)
        violation = torch.relu(crop_pred - soil_quality - 0.1)
        return violation.mean()

    @staticmethod
    def et_limit_constraint(
        water_pred: torch.Tensor,
        et: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Water stress should correlate with ET (simplified: penalize extreme predictions)."""
        return torch.relu(water_pred - 1.0).mean() + torch.relu(-water_pred).mean()

    @classmethod
    def compute_penalty(
        cls,
        pred: Dict[str, torch.Tensor],
        aux: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Compute total constraint penalty.
        aux may contain: rainfall, irrigation, evapotranspiration
        """
        weights = weights or {
            "groundwater_rainfall": 0.5,
            "soil_crop": 0.3,
            "et_limit": 0.2,
        }
        penalty = torch.tensor(0.0, device=pred["crop"].device)

        if "water" in pred and "rainfall" in aux and "irrigation" in aux:
            p = cls.groundwater_rainfall_constraint(
                pred["water"],
                aux["rainfall"],
                aux["irrigation"],
            )
            penalty = penalty + weights["groundwater_rainfall"] * p

        if "crop" in pred and "soil" in pred:
            p = cls.soil_crop_constraint(pred["crop"], pred["soil"])
            penalty = penalty + weights["soil_crop"] * p

        if "water" in pred:
            et = aux.get("evapotranspiration")
            p = cls.et_limit_constraint(pred["water"], et)
            penalty = penalty + weights["et_limit"] * p

        return penalty
