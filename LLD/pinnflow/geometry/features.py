"""Curvature and topology features derived from the legacy 10-D state."""
from __future__ import annotations

import numpy as np

GEOMETRY_CLASS_NAMES = ("straight", "elbow", "t-junction", "reducer")
CURVATURE_FEATURE_NAMES = (
    "bend_radius",
    "curvature_ratio",
    "sin_elbow_angle",
    "cos_elbow_angle",
    "thickness_ratio",
    "dean_number",
)


def _as_2d(X_raw: np.ndarray) -> np.ndarray:
    X = np.asarray(X_raw, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] < 10:
        raise ValueError("Geometry features require at least the legacy 10 input columns.")
    return X


def geometry_labels(X_raw: np.ndarray, n_classes: int = 4) -> np.ndarray:
    """Return clipped integer geometry labels from column 8."""
    X = _as_2d(X_raw)
    return np.clip(np.rint(X[:, 8]), 0, n_classes - 1).astype(int)


def extract_curvature_features(X_raw: np.ndarray) -> np.ndarray:
    """
    Build the 6-D elbow-aware feature block.

    Input columns follow the legacy contract:
    [d, t, L, P, u, dT, v, k_soil, shape_id, shape_param, ...].
    ``shape_param`` is interpreted as R/D for elbows and clipped to a
    physically useful range. Returned columns are:
    [R_bend, kappa, sin(theta), cos(theta), t/D, Dean].
    """
    X = _as_2d(X_raw)
    d = np.maximum(X[:, 0], 1e-6)
    t = np.maximum(X[:, 1], 1e-6)
    v = np.maximum(X[:, 6], 1e-9)
    shape_id = geometry_labels(X)
    r_over_d = np.clip(X[:, 9], 0.3, 5.0)

    bend_radius = r_over_d * d
    kappa = d / (2.0 * np.maximum(bend_radius, 1.0))
    theta_lookup = np.array([0.0, np.pi / 4.0, np.pi / 2.0, np.pi], dtype=float)
    theta_rad = theta_lookup[np.clip(shape_id, 0, len(theta_lookup) - 1)]
    thickness_ratio = t / d
    reynolds = 1000.0 * v * (d / 1000.0) / 0.001
    dean = reynolds * np.sqrt(np.maximum(kappa, 1e-6))

    return np.column_stack(
        [
            bend_radius,
            kappa,
            np.sin(theta_rad),
            np.cos(theta_rad),
            thickness_ratio,
            dean,
        ]
    )


def append_curvature_features(X_raw: np.ndarray) -> np.ndarray:
    """Append curvature features to 10-D legacy states, preserving 16-D states."""
    X = _as_2d(X_raw)
    if X.shape[1] >= 16:
        out = X[:, :16].copy()
        missing = np.isclose(np.abs(out[:, 10:16]).sum(axis=1), 0.0)
        if np.any(missing):
            out[missing, 10:16] = extract_curvature_features(out[missing, :10])
        return out
    return np.hstack([X[:, :10], extract_curvature_features(X)])


def ensure_geometry_state(X_raw: np.ndarray, mode: str = "derive") -> np.ndarray:
    """
    Return a 16-D state.

    ``mode='derive'`` computes curvature features. ``mode='zero'`` pads legacy
    callers with zeros for strict backwards compatibility.
    """
    X = _as_2d(X_raw)
    if X.shape[1] >= 16:
        return X[:, :16].copy()
    if mode == "zero":
        return np.hstack([X[:, :10], np.zeros((len(X), 6), dtype=float)])
    return append_curvature_features(X)
