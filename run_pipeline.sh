#!/bin/bash
# End-to-end pipeline: train → evaluate → explainability

set -e
cd "$(dirname "$0")"

echo "=== Crop-Water-Soil MTL Pipeline ==="

echo ""
echo "[1/3] Training..."
python -m src.training.train --config configs/model.yaml

echo ""
echo "[2/3] Evaluation..."
python -m src.evaluation.evaluate --checkpoint results/checkpoints/best_model.pt

echo ""
echo "[3/3] Explainability (SHAP)..."
python -m src.explainability.shap_analysis --checkpoint results/checkpoints/best_model.pt --task all

echo ""
echo "=== Pipeline complete ==="
echo "Results: results/"
echo "Plots: results/*.png"
