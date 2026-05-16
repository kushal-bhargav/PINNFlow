"""Stress heatmap plotting helpers."""
from __future__ import annotations

import numpy as np


def plot_stress_heatmap(ax, model, base_state: np.ndarray, diameters=None, pressures=None, shape_id: int = 1):
    diameters = np.linspace(114.0, 620.0, 40) if diameters is None else np.asarray(diameters)
    pressures = np.linspace(1.0, 20.0, 40) if pressures is None else np.asarray(pressures)
    grid = []
    for p in pressures:
        for d in diameters:
            row = np.asarray(base_state, dtype=float).copy()
            row[0] = d
            row[3] = p
            row[8] = shape_id
            grid.append(row)
    pred = model.predict(np.asarray(grid))[:, 0].reshape(len(pressures), len(diameters))
    im = ax.contourf(diameters, pressures, pred, levels=18, cmap="viridis")
    ax.set_xlabel("Diameter (mm)")
    ax.set_ylabel("Pressure (MPa)")
    ax.set_title("Stress heatmap")
    return im
