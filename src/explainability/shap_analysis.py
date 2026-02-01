#!/usr/bin/env python3
"""
SHAP-based Explainability for Crop-Water-Soil Multi-Task Model.

Provides feature importance per task, temporal contribution, and factor attribution.

Usage:
    python -m src.explainability.shap_analysis --checkpoint results/checkpoints/best_model.pt
"""

import argparse
from pathlib import Path
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mtl_model import CropWaterSoilMTL


def _predict_crop(model, x):
    """Wrapper for crop head prediction."""
    with torch.no_grad():
        out = model(x)
    return out["crop"]


def _predict_soil(model, x):
    """Wrapper for soil head (mean of 4 outputs as proxy)."""
    with torch.no_grad():
        out = model(x)
    return out["soil"].mean(dim=1, keepdim=True)


def _predict_water(model, x):
    """Wrapper for water head prediction."""
    with torch.no_grad():
        out = model(x)
    return out["water"]


def run_shap_analysis(
    model,
    X: np.ndarray,
    feature_names: list,
    task: str = "crop",
    n_background: int = 50,
    n_explain: int = 100,
    output_dir: str = "results",
) -> None:
    """
    Run SHAP analysis for a given task.

    Uses GradientExplainer for efficient deep model interpretation.
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Subsample for efficiency
    bg_idx = np.random.choice(len(X), min(n_background, len(X)), replace=False)
    explain_idx = np.random.choice(len(X), min(n_explain, len(X)), replace=False)
    X_bg = torch.tensor(X[bg_idx], dtype=torch.float32, requires_grad=True)
    X_explain = torch.tensor(X[explain_idx], dtype=torch.float32)

    wrappers = {"crop": _predict_crop, "soil": _predict_soil, "water": _predict_water}
    pred_fn = lambda x: wrappers[task](model, x)

    try:
        explainer = shap.GradientExplainer(pred_fn, X_bg)
        shap_values = explainer.shap_values(X_explain)
    except Exception as e:
        print(f"GradientExplainer failed: {e}. Using permutation importance fallback.")
        _plot_permutation_importance(model, X, feature_names, task, output_path)
        return

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values.mean(axis=1)  # (n_samples, seq_len, n_features) -> avg over seq

    # Feature importance (mean |SHAP| per feature)
    n_features = X.shape[2]
    if shap_values.shape[-1] != n_features:
        shap_values = shap_values.reshape(-1, n_features)
    importance = np.abs(shap_values).mean(axis=0)
    top_k = min(15, len(feature_names))
    order = np.argsort(importance)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_k), importance[order][::-1], color="#3498db", alpha=0.8)
    ax.set_yticks(range(top_k))
    labels = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in order[::-1]]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance: {task.capitalize()} Task")
    plt.tight_layout()
    fig_path = output_path / f"shap_importance_{task}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved: {fig_path}")

    # Summary plot (beeswarm) - use last timestep features as proxy
    if shap_values.ndim >= 2 and len(feature_names) <= 50:
        try:
            shap.summary_plot(
                shap_values,
                X_explain.numpy().reshape(-1, n_features)[: len(shap_values)],
                feature_names=feature_names[:n_features],
                show=False,
                max_display=12,
            )
            plt.title(f"SHAP Summary: {task.capitalize()} Task")
            plt.tight_layout()
            plt.savefig(output_path / f"shap_summary_{task}.png", dpi=150)
            plt.close()
            print(f"Saved: {output_path / f'shap_summary_{task}.png'}")
        except Exception as e:
            print(f"Summary plot skipped: {e}")


def _plot_permutation_importance(model, X, feature_names, task, output_path):
    """Fallback: permutation-based feature importance."""
    model.eval()
    X_t = torch.tensor(X[:100], dtype=torch.float32)
    with torch.no_grad():
        pred = model(X_t)[task].numpy()
    baseline = np.mean((pred - pred) ** 2)
    importance = []
    for i in range(min(20, X.shape[2])):
        X_perm = X_t.clone()
        X_perm[:, :, i] = X_perm[:, :, i][:, np.random.permutation(X_perm.shape[1])]
        with torch.no_grad():
            pred_perm = model(X_perm)[task].numpy()
        mse = np.mean((pred - pred_perm) ** 2)
        importance.append(mse)
    importance = np.array(importance)
    order = np.argsort(importance)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(order)), importance[order][::-1], color="#e74c3c", alpha=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f"F{i}" for i in order[::-1]], fontsize=9)
    ax.set_xlabel("MSE increase (permutation)")
    ax.set_title(f"Permutation Importance: {task.capitalize()} Task")
    plt.tight_layout()
    plt.savefig(output_path / f"permutation_importance_{task}.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / f'permutation_importance_{task}.png'}")


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for Crop-Water-Soil model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    parser.add_argument("--task", type=str, default="all", choices=["crop", "soil", "water", "all"])
    parser.add_argument("--n-background", type=int, default=50)
    parser.add_argument("--n-explain", type=int, default=100)
    args = parser.parse_args()

    from src.data.ingestion import DataIngestion
    from src.data.preprocessing import DataPreprocessor
    from src.data.feature_engineering import FeatureEngineer

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})
    if not config:
        with open(PROJECT_ROOT / args.config) as f:
            config = yaml.safe_load(f)

    # Load data
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
    target_cols = {"crop": "ndvi", "soil": "organic_carbon", "water": "groundwater_level"}
    X, _ = preprocessor.create_sequences(merged, feature_cols, target_cols)

    enc_cfg = config.get("model", {}).get("shared_encoder", {})
    heads_cfg = config.get("model", {}).get("heads", {})
    model = CropWaterSoilMTL(
        input_dim=X.shape[2],
        encoder_config={
            "hidden_dim": enc_cfg.get("hidden_dim", 128),
            "num_layers": enc_cfg.get("num_layers", 2),
            "dropout": 0.0,
            "encoder_type": enc_cfg.get("type", "lstm"),
            "bidirectional": enc_cfg.get("bidirectional", False),
        },
        crop_config=heads_cfg.get("crop", {"hidden_dims": [64, 32], "output_dim": 1}),
        soil_config=heads_cfg.get("soil", {"hidden_dims": [64, 32], "output_dim": 4}),
        water_config=heads_cfg.get("water", {"hidden_dims": [64, 32], "output_dim": 1}),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tasks = ["crop", "soil", "water"] if args.task == "all" else [args.task]
    for task in tasks:
        print(f"\nAnalyzing task: {task}")
        run_shap_analysis(
            model,
            X,
            feature_names=feature_cols,
            task=task,
            n_background=args.n_background,
            n_explain=args.n_explain,
            output_dir=str(PROJECT_ROOT / args.output_dir),
        )
    print("\nExplainability analysis complete.")


if __name__ == "__main__":
    main()
