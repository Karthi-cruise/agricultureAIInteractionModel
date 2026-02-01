"""
Evaluation Metrics Module

Computes RMSE, MAE per task and constraint violation rate.
"""

import numpy as np
from typing import Dict, Optional


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_metrics(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    task_metrics: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per task.

    Args:
        predictions: {task: pred_array}
        targets: {task: target_array}
        task_metrics: {task: "rmse"|"mae"} - default RMSE for all

    Returns:
        {task: {"rmse": ..., "mae": ...}}
    """
    task_metrics = task_metrics or {}
    results = {}

    for task in predictions:
        if task not in targets:
            continue
        y_true = np.asarray(targets[task]).flatten()
        y_pred = np.asarray(predictions[task]).flatten()
        if len(y_true) != len(y_pred):
            min_len = min(len(y_true), len(y_pred))
            y_true, y_pred = y_true[:min_len], y_pred[:min_len]

        metric_type = task_metrics.get(task, "rmse")
        results[task] = {
            "rmse": compute_rmse(y_true, y_pred),
            "mae": compute_mae(y_true, y_pred),
        }

    return results


def constraint_violation_rate(
    pred_crop: np.ndarray,
    pred_soil: np.ndarray,
    pred_water: np.ndarray,
    rainfall: np.ndarray,
    irrigation: np.ndarray,
) -> float:
    """
    Approximate constraint violation rate.
    Violation: groundwater increases when rainfall low and irrigation high.
    """
    rain_low = rainfall < np.median(rainfall)
    irr_high = irrigation > np.median(irrigation)
    stress_condition = rain_low & irr_high
    # High water pred when stress condition suggests violation (simplified)
    violations = stress_condition & (pred_water > 0.6)
    return float(np.mean(violations)) if len(violations) > 0 else 0.0
