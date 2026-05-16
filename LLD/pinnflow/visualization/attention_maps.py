"""Curvature gating visualizations."""
from __future__ import annotations

import numpy as np

from pinnflow.geometry.features import ensure_geometry_state


def plot_curvature_attention(ax, model, base_state: np.ndarray, r_over_d=None, reynolds=None):
    r_over_d = np.linspace(1.0, 5.0, 40) if r_over_d is None else np.asarray(r_over_d)
    reynolds = np.logspace(4, 6, 40) if reynolds is None else np.asarray(reynolds)
    rows = []
    base = np.asarray(base_state, dtype=float).copy()
    for re in reynolds:
        for rd in r_over_d:
            row = base.copy()
            row[8] = 1
            row[9] = rd
            row[6] = re * 0.001 / (1000.0 * (max(row[0], 1.0) / 1000.0))
            rows.append(row)
    X = ensure_geometry_state(np.asarray(rows))
    gate = model.gating.forward(X)[:, 1].reshape(len(reynolds), len(r_over_d))
    im = ax.contourf(r_over_d, reynolds, gate, levels=18, cmap="magma", vmin=0, vmax=1)
    ax.set_yscale("log")
    ax.set_xlabel("R/D")
    ax.set_ylabel("Re")
    ax.set_title("Elbow gate weight")
    return im
