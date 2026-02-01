#!/usr/bin/env python3
"""
Main training entry point for Crop-Water-Soil Multi-Task Model.

Usage:
    python -m src.training.train --config configs/model.yaml
    python src/training/train.py --config configs/model.yaml
"""

import argparse
from pathlib import Path
import sys
import yaml

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.models.mtl_model import CropWaterSoilMTL
from src.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Crop-Water-Soil MTL model")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Path to config YAML")
    parser.add_argument("--data-root", type=str, default="data", help="Data directory root")
    parser.add_argument("--output", type=str, default="results/checkpoints/best_model.pt", help="Output checkpoint path")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic data (default)")
    parser.add_argument("--no-synthetic", dest="synthetic", action="store_false", help="Use real data from data/raw")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(str(config_path))
    output_path = PROJECT_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    data_root = PROJECT_ROOT / args.data_root if not Path(args.data_root).is_absolute() else Path(args.data_root)

    print("=" * 60)
    print("Crop-Water-Soil Multi-Task Model Training")
    print("=" * 60)

    # 1. Data ingestion
    print("\n[1/5] Loading data...")
    ingestion = DataIngestion(data_root=str(data_root))
    data = ingestion.load_all(use_synthetic=args.synthetic)
    print(f"  Satellite: {len(data['satellite'])} rows | Soil: {len(data['soil'])} rows")
    print(f"  Groundwater: {len(data['groundwater'])} rows | Weather: {len(data['weather'])} rows")

    # 2. Preprocessing
    print("\n[2/5] Preprocessing & alignment...")
    seq_len = config.get("data", {}).get("sequence_length", 24)
    preprocessor = DataPreprocessor(sequence_length=seq_len)
    merged, _, target_mapping = preprocessor.preprocess_pipeline(data)

    # 3. Feature engineering
    print("\n[3/5] Feature engineering...")
    engineer = FeatureEngineer()
    merged = engineer.transform(merged)

    exclude = ["region_id", "time_step", "ndvi", "organic_carbon", "groundwater_level",
               "nitrogen", "phosphorus", "potassium"]
    feature_cols = [c for c in merged.select_dtypes(include=[np.number]).columns
                    if c not in exclude][:64]

    soil_cols = ["nitrogen", "phosphorus", "potassium", "organic_carbon"]
    target_cols_dict = {
        "crop": "ndvi",
        "soil": soil_cols if all(c in merged.columns for c in soil_cols) else "organic_carbon",
        "water": "groundwater_level",
    }

    X, targets_raw = preprocessor.create_sequences(merged, feature_cols, target_cols_dict)

    # Build targets: crop (N,1), soil (N,4), water (N,1)
    targets = {}
    targets["crop"] = np.asarray(targets_raw["crop"], dtype=np.float32).reshape(-1, 1)
    targets["water"] = np.asarray(targets_raw["water"], dtype=np.float32).reshape(-1, 1)

    soil_t = targets_raw["soil"]
    if soil_t.ndim == 1:
        targets["soil"] = np.column_stack([soil_t] * 4).astype(np.float32)
    else:
        targets["soil"] = np.asarray(soil_t, dtype=np.float32)

    # Aux data for constraints (last timestep of each sequence)
    n_samples = len(X)
    aux_rainfall = np.zeros(n_samples, dtype=np.float32)
    aux_irrigation = np.zeros(n_samples, dtype=np.float32)
    aux_et = np.zeros(n_samples, dtype=np.float32)
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
            aux_et[idx] = row.get("evapotranspiration", 0)
            idx += 1

    X_t = torch.tensor(X, dtype=torch.float32)
    targets_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()}
    aux_t = {
        "rainfall": torch.tensor(aux_rainfall[:n_samples], dtype=torch.float32).unsqueeze(-1),
        "irrigation": torch.tensor(aux_irrigation[:n_samples], dtype=torch.float32).unsqueeze(-1),
        "evapotranspiration": torch.tensor(aux_et[:n_samples], dtype=torch.float32).unsqueeze(-1),
    }

    input_dim = X_t.shape[2]
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

    # 4. Training
    print("\n[4/5] Training model...")
    trainer = Trainer(model=model, config=config)
    history = trainer.fit(X_t, targets_t, aux=aux_t)

    # 5. Save
    print("\n[5/5] Saving checkpoint...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(output_path))
    print(f"  Saved to {output_path}")

    history_path = output_path.parent / "training_history.yaml"
    with open(history_path, "w") as f:
        yaml.dump({k: [float(x) for x in v] for k, v in history.items()}, f)
    print(f"  History saved to {history_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
