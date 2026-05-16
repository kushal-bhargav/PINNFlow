"""Lightweight geometry gating network."""
from __future__ import annotations

import numpy as np

from pinnflow.geometry.features import ensure_geometry_state, geometry_labels
from pinnflow.layers import AdamLayer


class GeometryGatingNetwork:
    """Two-layer MLP that maps a geometry-aware state to class probabilities."""

    def __init__(self, n_in: int = 16, n_classes: int = 4, hidden: int = 32, lr: float = 1e-3):
        self.n_in = n_in
        self.n_classes = n_classes
        self.layers = [
            AdamLayer(n_in, hidden, "swish", lr),
            AdamLayer(hidden, n_classes, "linear", lr),
        ]
        self.is_pretrained = False

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        Xg = ensure_geometry_state(X)
        if Xg.shape[1] < self.n_in:
            Xg = np.hstack([Xg, np.zeros((len(Xg), self.n_in - Xg.shape[1]))])
        return Xg[:, : self.n_in]

    def logits(self, X: np.ndarray) -> np.ndarray:
        h = self._prepare(X)
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def forward(self, X: np.ndarray) -> np.ndarray:
        z = self.logits(X)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def pretrain(self, X: np.ndarray, y: np.ndarray | None = None, epochs: int = 100, batch: int = 128) -> None:
        Xg = self._prepare(X)
        labels = geometry_labels(Xg, self.n_classes) if y is None else np.asarray(y, dtype=int)
        labels = np.clip(labels, 0, self.n_classes - 1)
        Y = np.zeros((len(Xg), self.n_classes), dtype=float)
        Y[np.arange(len(Xg)), labels] = 1.0

        n = len(Xg)
        for _ in range(max(1, epochs)):
            for st in range(0, n, batch):
                idx = np.random.choice(n, min(batch, n), replace=False)
                probs = self.forward(Xg[idx])
                grad = (probs - Y[idx]) / max(len(idx), 1)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
        self.is_pretrained = True
