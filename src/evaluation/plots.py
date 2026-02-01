"""
Visualization Module

Generates training curves, prediction vs actual plots, and seasonal trends.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot train/val loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#2ecc71", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", color="#3498db", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_predictions_vs_actual(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> None:
    """Scatter plots: predicted vs actual for each task."""
    tasks = [k for k in predictions if k in targets]
    n_tasks = len(tasks)
    if n_tasks == 0:
        return
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    for i, task in enumerate(tasks):
        y_true = np.asarray(targets[task]).flatten()
        y_pred = np.asarray(predictions[task]).flatten()
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]
        axes[i].scatter(y_true, y_pred, alpha=0.5, c=colors[i % 3], edgecolors="white", s=30)
        max_val = max(y_true.max(), y_pred.max())
        axes[i].plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect")
        axes[i].set_xlabel(f"Actual ({task})")
        axes[i].set_ylabel(f"Predicted ({task})")
        axes[i].set_title(f"{task.capitalize()} Head")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_seasonal_trends(
    time_steps: np.ndarray,
    crop_actual: np.ndarray,
    crop_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot seasonal trend: crop health over time (actual vs predicted)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_steps, crop_actual, label="Actual", color="#2ecc71", alpha=0.8)
    ax.plot(time_steps, crop_pred, label="Predicted", color="#3498db", alpha=0.8, linestyle="--")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Crop Health (NDVI)")
    ax.set_title("Seasonal Trend: Crop Health")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_error_heatmap(
    errors: np.ndarray,
    regions: np.ndarray,
    time_bins: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Heatmap of prediction errors by region and time."""
    # Simplified: reshape errors if possible
    try:
        mat = errors.reshape(len(np.unique(regions)), -1)[:20, :20]
    except Exception:
        mat = errors[:400].reshape(20, 20)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto")
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Region")
    ax.set_title("Prediction Error Heatmap")
    plt.colorbar(im, ax=ax, label="Error")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
