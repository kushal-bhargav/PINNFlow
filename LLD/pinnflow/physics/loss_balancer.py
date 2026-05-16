"""Adaptive loss weighting utilities."""
from __future__ import annotations

import numpy as np


class LossBalancer:
    """Small stateful balancer implementing GradNorm, SoftAdapt, and uncertainty weighting proxies."""

    def __init__(self, names: list[str], strategy: str = "gradnorm", alpha: float = 0.12):
        self.names = list(names)
        self.strategy = strategy
        self.alpha = alpha
        self.weights = {name: 1.0 for name in self.names}
        self.prev_losses: dict[str, float] | None = None
        self.log_vars = {name: 0.0 for name in self.names}

    def update(self, losses: dict[str, float], grad_norms: dict[str, float] | None = None) -> dict[str, float]:
        vals = {k: float(max(losses.get(k, 0.0), 1e-12)) for k in self.names}
        if self.strategy == "softadapt":
            if self.prev_losses is None:
                self.prev_losses = vals
                return self.weights.copy()
            deltas = np.array([vals[k] - self.prev_losses[k] for k in self.names])
            exp = np.exp(deltas - deltas.max())
            probs = exp / np.maximum(exp.sum(), 1e-12)
            self.weights = {k: float(probs[i] * len(self.names)) for i, k in enumerate(self.names)}
            self.prev_losses = vals
        elif self.strategy == "uncertainty":
            for k in self.names:
                self.log_vars[k] = 0.98 * self.log_vars[k] + 0.02 * np.log(vals[k])
                self.weights[k] = float(1.0 / (2.0 * np.exp(self.log_vars[k]) + 1e-12))
        else:
            norms = grad_norms or vals
            mean_norm = float(np.mean([max(norms.get(k, vals[k]), 1e-12) for k in self.names]))
            for k in self.names:
                gi = max(norms.get(k, vals[k]), 1e-12)
                self.weights[k] = float(np.clip(self.weights[k] * (mean_norm / gi) ** self.alpha, 1e-3, 1e3))
        return self.weights.copy()

    def total(self, losses: dict[str, float]) -> float:
        return float(sum(self.weights.get(k, 1.0) * float(losses.get(k, 0.0)) for k in self.names))
