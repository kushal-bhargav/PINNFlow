"""Straight-pipe analytic residual helpers."""
from __future__ import annotations

import numpy as np


def hoop_stress(X_raw: np.ndarray) -> np.ndarray:
    X = np.asarray(X_raw, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    d = np.maximum(X[:, 0:1], 1.0)
    t = np.maximum(X[:, 1:2], 1.0)
    P = np.maximum(X[:, 3:4], 0.0)
    return P * d / (2.0 * t)


def darcy_pressure_drop(X_raw: np.ndarray) -> np.ndarray:
    X = np.asarray(X_raw, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    d = np.maximum(X[:, 0:1], 1.0)
    L = np.maximum(X[:, 2:3], 1.0)
    v = np.maximum(X[:, 6:7], 0.1)
    diameter_m = np.maximum(d / 1000.0, 1e-4)
    reynolds = 1000.0 * v * diameter_m / 0.001
    friction = np.where(reynolds > 4000.0, 0.316 * np.maximum(reynolds, 1.0) ** -0.25, 64.0 / np.maximum(reynolds, 1.0))
    return friction * (L / diameter_m) * 1000.0 * v**2 / 2.0 / 1000.0
