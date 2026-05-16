"""Expert and head subnetworks for the MoE PINN."""
from __future__ import annotations

import numpy as np

from pinnflow.layers import AdamLayer


class ExpertNetwork:
    """Independent expert MLP with its own Adam state."""

    def __init__(self, n_in: int, width: int, depth: int, lr: float = 1e-3):
        dims = [n_in] + [width] * max(1, depth)
        self.layers = [AdamLayer(dims[i], dims[i + 1], "swish", lr) for i in range(len(dims) - 1)]
        self.output_dim = dims[-1]

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = X
        for layer in self.layers:
            h = layer.forward(h)
        return h


class HeadNetwork:
    """Small output head used for stress and pressure-drop predictions."""

    def __init__(self, n_in: int, hidden: int = 128, n_out: int = 1, lr: float = 1e-3):
        self.layers = [
            AdamLayer(n_in, hidden, "swish", lr),
            AdamLayer(hidden, n_out, "linear", lr),
        ]

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = X
        for layer in self.layers:
            h = layer.forward(h)
        return h
