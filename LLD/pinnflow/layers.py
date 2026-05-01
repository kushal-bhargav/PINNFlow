"""
pinnflow/layers.py
───────────────────
AdamLayer — a dense layer with per-weight Adam optimiser state.
Shared by PINN, VAE, PPO actor/critic, and all baselines.
"""
import numpy as np
from pinnflow.activations import tanh, tanh_g, relu, relu_g, swish, swish_g


class AdamLayer:
    """Dense layer with per-weight Adam optimiser state."""

    def __init__(self, n_in: int, n_out: int,
                 activation: str = "swish", lr: float = 1e-3):
        scale = np.sqrt(2.0 / max(n_in, 1))
        self.W   = np.random.randn(n_in, n_out) * scale
        self.b   = np.zeros(n_out)
        self.act = activation
        self.lr  = lr
        # Adam moment estimates
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.t  = 0
        self.cache: dict = {}

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache["x"] = x
        z = x @ self.W + self.b
        self.cache["z"] = z
        act_fn = {"tanh": tanh, "relu": relu, "swish": swish}.get(
            self.act, lambda v: v
        )
        h = act_fn(z)
        self.cache["h"] = h
        return h

    # ── Backward ─────────────────────────────────────────────────────────────
    def backward(self, dh: np.ndarray) -> np.ndarray:
        z  = self.cache["z"]
        grad_fn = {"tanh": tanh_g, "relu": relu_g, "swish": swish_g}.get(
            self.act, lambda v: np.ones_like(v)
        )
        dz = grad_fn(z) * dh
        x  = self.cache["x"]
        dW = x.T @ dz / max(len(x), 1)
        db = dz.mean(0)
        dx = dz @ self.W.T
        self.t += 1
        for p, m, v, g in [
            ("W", self.mW, self.vW, dW),
            ("b", self.mb, self.vb, db),
        ]:
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g ** 2
            mh   = m / (1 - 0.9  ** self.t)
            vh   = v / (1 - 0.999 ** self.t)
            getattr(self, p)[:] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        return dx
