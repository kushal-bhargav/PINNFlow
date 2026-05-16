"""Fourier features with separate bandwidths for base and curvature inputs."""
from __future__ import annotations

import numpy as np

from pinnflow.geometry.features import ensure_geometry_state


class GeometryAwareFourierLayer:
    """Random Fourier features for 16-D geometry-aware states."""

    def __init__(
        self,
        n_in: int = 16,
        n_features_base: int = 64,
        n_features_curve: int = 32,
        sigma_base: float = 1.0,
        sigma_curve: float = 0.3,
        random_state: int | None = None,
    ):
        if n_in < 16:
            raise ValueError("GeometryAwareFourierLayer expects at least 16 inputs.")
        rng = np.random.default_rng(random_state)
        self.n_in = n_in
        self.n_base_inputs = min(10, n_in)
        self.n_curve_inputs = 6
        self.B_std = rng.normal(0.0, sigma_base, size=(self.n_base_inputs, n_features_base))
        self.B_curve = rng.normal(0.0, sigma_curve, size=(self.n_curve_inputs, n_features_curve))
        extra_dim = max(n_in - 16, 0)
        self.B_extra = (
            rng.normal(0.0, sigma_base, size=(extra_dim, max(1, n_features_curve // 2)))
            if extra_dim
            else None
        )
        self.output_dim = (n_features_base + n_features_curve) * 2
        if self.B_extra is not None:
            self.output_dim += self.B_extra.shape[1] * 2

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[1] < 16:
            X_arr = ensure_geometry_state(X_arr)

        p_base = X_arr[:, :10] @ self.B_std
        p_curve = X_arr[:, 10:16] @ self.B_curve
        blocks = [np.cos(p_base), np.sin(p_base), np.cos(p_curve), np.sin(p_curve)]
        if self.B_extra is not None and X_arr.shape[1] > 16:
            p_extra = X_arr[:, 16 : 16 + self.B_extra.shape[0]] @ self.B_extra
            blocks.extend([np.cos(p_extra), np.sin(p_extra)])
        return np.hstack(blocks)
