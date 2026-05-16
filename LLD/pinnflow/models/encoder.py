"""Geometry-aware shared encoder."""
from __future__ import annotations

import numpy as np

from pinnflow.geometry.fourier import GeometryAwareFourierLayer
from pinnflow.layers import AdamLayer


class GeometryAwareEncoder:
    """Fourier layer followed by dense encoder layers."""

    def __init__(self, n_in: int = 16, hidden: tuple[int, ...] = (192,), lr: float = 1e-3):
        self.fourier = GeometryAwareFourierLayer(n_in=n_in)
        dims = [self.fourier.output_dim, *hidden]
        self.layers = [AdamLayer(dims[i], dims[i + 1], "swish", lr) for i in range(len(dims) - 1)]
        self.output_dim = dims[-1]

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = self.fourier.transform(X)
        for layer in self.layers:
            h = layer.forward(h)
        return h
