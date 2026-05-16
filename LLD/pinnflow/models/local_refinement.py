"""Elbow-only residual correction branch."""
from __future__ import annotations

import numpy as np

from pinnflow.geometry.features import ensure_geometry_state
from pinnflow.layers import AdamLayer


class ElbowRefinementBranch:
    """
    Predict additive stress residuals for elbow cases.

    Inputs are the six curvature features plus the global stress prediction.
    """

    def __init__(self, hidden: int = 64, lr: float = 1e-3, refinement_alpha: float = 0.8):
        self.refinement_alpha = refinement_alpha
        self.is_trained = False
        self.layers = [
            AdamLayer(7, hidden, "swish", lr),
            AdamLayer(hidden, hidden, "swish", lr),
            AdamLayer(hidden, 1, "linear", lr),
        ]

    def forward(self, X: np.ndarray, sigma_global: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.zeros((len(ensure_geometry_state(X)), 1), dtype=float)
        Xg = ensure_geometry_state(X)
        sg = np.asarray(sigma_global, dtype=float).reshape(-1, 1)
        h = np.hstack([Xg[:, 10:16], sg])
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def activation(self, gate_probs: np.ndarray, elbow_index: int = 1) -> np.ndarray:
        ge = np.asarray(gate_probs, dtype=float)[:, elbow_index : elbow_index + 1]
        return self.refinement_alpha * (1.0 / (1.0 + np.exp(-8.0 * (ge - 0.6)))) * 2.0

    def apply(self, X: np.ndarray, sigma_global: np.ndarray, gate_probs: np.ndarray) -> np.ndarray:
        delta = self.forward(X, sigma_global)
        return np.asarray(sigma_global).reshape(-1, 1) + self.activation(gate_probs) * delta
