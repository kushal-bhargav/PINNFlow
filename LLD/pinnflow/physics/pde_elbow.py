"""Curvature-aware elbow residuals."""
from __future__ import annotations

import numpy as np

from pinnflow.geometry.features import ensure_geometry_state
from pinnflow.physics.pde_straight import darcy_pressure_drop, hoop_stress


def _prepared(X_raw: np.ndarray) -> np.ndarray:
    return ensure_geometry_state(X_raw)


def ito_friction_factor(X_raw: np.ndarray) -> np.ndarray:
    X = _prepared(X_raw)
    d = np.maximum(X[:, 0:1], 1.0)
    v = np.maximum(X[:, 6:7], 0.1)
    r_over_d = np.clip(X[:, 9:10], 0.3, 5.0)
    reynolds = 1000.0 * v * (d / 1000.0) / 0.001
    return 0.00431 * np.maximum(reynolds, 1.0) ** 0.05 * np.maximum(r_over_d, 0.3) ** -0.29


def elbow_pressure_drop(X_raw: np.ndarray) -> np.ndarray:
    X = _prepared(X_raw)
    d_m = np.maximum(X[:, 0:1] / 1000.0, 1e-4)
    v = np.maximum(X[:, 6:7], 0.1)
    r_over_d = np.clip(X[:, 9:10], 0.3, 5.0)
    theta = np.arctan2(X[:, 12:13], X[:, 13:14])
    theta = np.where(theta <= 0.0, np.pi / 2.0, theta)
    arc_length = np.maximum(r_over_d * d_m * theta, d_m)
    return ito_friction_factor(X) * (arc_length / d_m) * 1000.0 * v**2 / 2.0 / 1000.0


def dean_stress_factor(X_raw: np.ndarray) -> np.ndarray:
    X = _prepared(X_raw)
    dean = np.maximum(X[:, 15:16], 1.0)
    return np.where(dean > 11.6, 1.0 + 0.033 * np.log10(dean) ** 2, 1.0)


def stress_concentration_factor(X_raw: np.ndarray) -> np.ndarray:
    X = _prepared(X_raw)
    t_over_d = np.maximum(X[:, 14:15], 1e-4)
    r_over_d = np.clip(X[:, 9:10], 0.3, 5.0)
    return 0.9 / np.maximum(t_over_d, 1e-4) ** (2.0 / 3.0) * np.maximum(r_over_d, 0.3) ** (1.0 / 3.0)


def elbow_stress_multiplier(X_raw: np.ndarray) -> np.ndarray:
    scf = stress_concentration_factor(X_raw)
    dean = dean_stress_factor(X_raw)
    # Keep the synthetic prior bounded; the residual learner can absorb FEM detail.
    return np.clip(0.35 * scf * dean, 1.05, 4.0)


def elbow_residuals(X_raw: np.ndarray, sigma_pred: np.ndarray, dp_pred: np.ndarray) -> dict:
    X = _prepared(X_raw)
    sigma = np.asarray(sigma_pred, dtype=float).reshape(-1, 1)
    dP = np.asarray(dp_pred, dtype=float).reshape(-1, 1)
    hoop = np.maximum(hoop_stress(X), 1.0)
    straight_sigma = hoop
    dean_limit = straight_sigma * dean_stress_factor(X)
    scf = stress_concentration_factor(X)
    return {
        "ito": np.mean((dP - elbow_pressure_drop(X)) ** 2),
        "dean": np.mean(np.maximum(0.0, sigma - dean_limit) ** 2),
        "scf": np.mean((sigma / hoop - scf) ** 2),
        "darcy": np.mean((dP - darcy_pressure_drop(X)) ** 2),
    }
