#!/usr/bin/env python3
"""
Evaluation runner for Crop-Water-Soil Multi-Task Model.

Usage:
    python -m src.evaluation.evaluate --checkpoint results/checkpoints/best_model.pt
"""

import argparse
from pathlib import Path
import sys
import yaml
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.models.mtl_model import CropWaterSoilMTL
from src.evaluation.metrics import compute_metrics, constraint_violation_rate
from src.evaluation.plots import (
    plot_predictions_vs_actual,
    plot_training_history,
    plot_seasonal_trends,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Crop-Water-Soil MTL model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    output_dir = PROJECT_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})
    if not config:
        config_path = PROJECT_ROOT / args.config
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Prepare data (same pipeline as training)
    ingestion = DataIngestion(data_root=str(PROJECT_ROOT / args.data_root))
    data = ingestion.load_all(use_synthetic=True)
    seq_len = config.get("data", {}).get("sequence_length", 24)
    preprocessor = DataPreprocessor(sequence_length=seq_len)
    merged, _, _ = preprocessor.preprocess_pipeline(data)
    engineer = FeatureEngineer()
    merged = engineer.transform(merged)

    exclude = ["region_id", "time_step", "ndvi", "organic_carbon", "groundwater_level",
               "nitrogen", "phosphorus", "potassium"]
    feature_cols = [c for c in merged.select_dtypes(include=[np.number]).columns if c not in exclude][:64]
    soil_cols = ["nitrogen", "phosphorus", "potassium", "organic_carbon"]
    target_cols = {
        "crop": "ndvi",
        "soil": soil_cols if all(c in merged.columns for c in soil_cols) else "organic_carbon",
        "water": "groundwater_level",
    }
    X, targets_raw = preprocessor.create_sequences(merged, feature_cols, target_cols)

    targets = {}
    targets["crop"] = np.asarray(targets_raw["crop"], dtype=np.float32).reshape(-1, 1)
    targets["water"] = np.asarray(targets_raw["water"], dtype=np.float32).reshape(-1, 1)
    soil_t = targets_raw["soil"]
    targets["soil"] = (np.column_stack([soil_t] * 4) if soil_t.ndim == 1 else np.asarray(soil_t)).astype(np.float32)

    # Aux for constraint check
    n_samples = len(X)
    aux_rainfall = np.zeros(n_samples)
    aux_irrigation = np.zeros(n_samples)
    idx = 0
    for region in merged["region_id"].unique():
        subset = merged[merged["region_id"] == region].sort_values("time_step").reset_index(drop=True)
        if len(subset) < seq_len + 1:
            continue
        for j in range(len(subset) - seq_len):
            if idx >= n_samples:
                break
            row = subset.iloc[j + seq_len - 1]
            aux_rainfall[idx] = row.get("rainfall_mm", 0)
            aux_irrigation[idx] = row.get("irrigation_mm", 0)
            idx += 1

    # Build model and load weights
    input_dim = X.shape[2]
    enc_cfg = config.get("model", {}).get("shared_encoder", {})
    heads_cfg = config.get("model", {}).get("heads", {})
    model = CropWaterSoilMTL(
        input_dim=input_dim,
        encoder_config={
            "hidden_dim": enc_cfg.get("hidden_dim", 128),
            "num_layers": enc_cfg.get("num_layers", 2),
            "dropout": enc_cfg.get("dropout", 0.2),
            "encoder_type": enc_cfg.get("type", "lstm"),
            "bidirectional": enc_cfg.get("bidirectional", False),
        },
        crop_config=heads_cfg.get("crop", {"hidden_dims": [64, 32], "output_dim": 1}),
        soil_config=heads_cfg.get("soil", {"hidden_dims": [64, 32], "output_dim": 4}),
        water_config=heads_cfg.get("water", {"hidden_dims": [64, 32], "output_dim": 1}),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Predict
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        pred_dict = model(X_t)

    predictions = {k: v.numpy() for k, v in pred_dict.items()}

    # Metrics
    metrics = compute_metrics(predictions, targets)
    cvr = constraint_violation_rate(
        predictions["crop"].flatten(),
        predictions["soil"].mean(axis=1),
        predictions["water"].flatten(),
        aux_rainfall[:n_samples],
        aux_irrigation[:n_samples],
    )

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for task, m in metrics.items():
        print(f"\n{task.upper()}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}")
    print(f"\nConstraint Violation Rate: {cvr:.4f}")
    print("=" * 50)

    # Save metrics
    metrics_path = output_dir / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump({**{k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics.items()},
                   "constraint_violation_rate": float(cvr)}, f)
    print(f"\nMetrics saved to {metrics_path}")

    # Plots
    plot_predictions_vs_actual(predictions, targets, save_path=str(output_dir / "predictions_vs_actual.png"))
    print(f"Plot saved: {output_dir / 'predictions_vs_actual.png'}")

    history_path = ckpt_path.parent / "training_history.yaml"
    if history_path.exists():
        with open(history_path) as f:
            history = yaml.safe_load(f)
        plot_training_history(history, save_path=str(output_dir / "training_curves.png"))
        print(f"Plot saved: {output_dir / 'training_curves.png'}")

    n_show = min(200, len(predictions["crop"]))
    plot_seasonal_trends(
        np.arange(n_show),
        targets["crop"][:n_show].flatten(),
        predictions["crop"][:n_show].flatten(),
        save_path=str(output_dir / "seasonal_trend.png"),
    )
    print(f"Plot saved: {output_dir / 'seasonal_trend.png'}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
