#!/usr/bin/env python3
"""Generate docs/architecture.png for the README."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

# Colors
c_input = "#3498db"
c_encoder = "#2ecc71"
c_crop, c_soil, c_water = "#27ae60", "#d35400", "#2980b9"
c_constraint = "#e74c3c"

# Input boxes (top)
inputs = [
    (1.5, 6.5, "Satellite\n(NDVI/EVI)"),
    (4, 6.5, "Weather\n(Temp, Rain, ET)"),
    (6.5, 6.5, "Soil\n(NPK, OC)"),
    (9, 6.5, "Water\n(GW, Irrigation)"),
]
for x, y, label in inputs:
    rect = FancyBboxPatch((x, y), 1.8, 1.2, boxstyle="round,pad=0.05", facecolor=c_input, edgecolor="white")
    ax.add_patch(rect)
    ax.text(x + 0.9, y + 0.6, label, ha="center", va="center", fontsize=8, color="white")

# Shared encoder (center)
enc = FancyBboxPatch((3, 3.5), 6, 2.2, boxstyle="round,pad=0.1", facecolor=c_encoder, edgecolor="white", linewidth=2)
ax.add_patch(enc)
ax.text(6, 4.6, "Shared Temporal Encoder", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
ax.text(6, 3.9, "(LSTM / CNN / Transformer)", ha="center", va="center", fontsize=9, color="white")

# Arrows: inputs -> encoder
for x, y, _ in inputs:
    ax.annotate("", xy=(5, 4.8), xytext=(x + 0.9, y + 1.2), arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

# Task heads (bottom)
heads = [(2, 1.2, "Crop Health\nHead", c_crop), (5.5, 1.2, "Soil Nutrient\nHead", c_soil), (9, 1.2, "Groundwater\nStress Head", c_water)]
for x, y, label, col in heads:
    rect = FancyBboxPatch((x, y), 2.2, 1.2, boxstyle="round,pad=0.05", facecolor=col, edgecolor="white")
    ax.add_patch(rect)
    ax.text(x + 1.1, y + 0.6, label, ha="center", va="center", fontsize=9, color="white")

# Arrows: encoder -> heads
for x, y, _, _ in heads:
    ax.annotate("", xy=(x + 1.1, y + 1.2), xytext=(6, 3.5), arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

# Constraint layer
const = FancyBboxPatch((4, 0.2), 4, 0.6, boxstyle="round,pad=0.05", facecolor=c_constraint, edgecolor="white")
ax.add_patch(const)
ax.text(6, 0.5, "Constraint Validation Layer", ha="center", va="center", fontsize=10, color="white")
ax.annotate("", xy=(6, 0.8), xytext=(6, 2.2), arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

plt.tight_layout()
out_path = "docs/architecture.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved {out_path}")
