"""
pinnflow/diffusion.py
──────────────────────
MODULE 7 — Physics-Guided Diffusion (DDPM) [V6]

[V6] Physics-Guided Diffusion for Design Parameter Synthesis.
     8-dimensional tabular DDPM with sinusoidal time embeddings
     and PINN-classifier guidance.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Callable
from pinnflow.pinn import MultiTaskPINN
from pinnflow.layers import AdamLayer

def sinusoidal_embedding(t: np.ndarray, dim: int = 64) -> np.ndarray:
    """Standard sinusoidal time embedding for DDPM."""
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
    return emb

class PipelineDDPM:
    """
    [P2.1] Denoising Diffusion Probabilistic Model (DDPM) for 8-dim design vectors.
    Features Sinusoidal time embeddings and PINN guidance.
    """
    def __init__(
        self,
        dim: int = 8,
        steps: int = 200,
        lr: float = 1e-3,
        pinn: Optional[MultiTaskPINN] = None
    ):
        self.dim = dim
        self.steps = steps
        self.pinn = pinn
        
        # Linear noise schedule
        self.beta = np.linspace(1e-4, 0.02, steps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = np.cumprod(self.alpha)
        
        # Denoising Network (MLP)
        # Input: [x_t (8), t_emb (64)] -> 72 dims
        emb_dim = 64
        h = 256
        self.model = [
            AdamLayer(dim + emb_dim, h, "swish", lr),
            AdamLayer(h, h, "swish", lr),
            AdamLayer(h, h, "swish", lr),
            AdamLayer(h, dim, "linear", lr)
        ]

    def _forward_model(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        t_emb = sinusoidal_embedding(t, dim=64)
        h = np.hstack([x, t_emb])
        for l in self.model:
            h = l.forward(h)
        return h

    def fit(self, X_train: np.ndarray, epochs: int = 300, batch: int = 128):
        """[P2.1] Standard DDPM training: predict added noise epsilon."""
        n = len(X_train)
        for ep in range(epochs):
            idx = np.random.permutation(n)
            for st in range(0, n, batch):
                sl = idx[st : st + batch]
                x0 = X_train[sl]
                
                # Sample random t
                t = np.random.randint(0, self.steps, size=(len(sl),))
                eps = np.random.randn(*x0.shape)
                
                ah = self.alpha_hat[t][:, None]
                xt = np.sqrt(ah) * x0 + np.sqrt(1 - ah) * eps
                
                # Predict noise
                eps_pred = self._forward_model(xt, t)
                
                # Backprop
                grad = 2 * (eps_pred - eps) / len(sl)
                for l in reversed(self.model):
                    grad = l.backward(grad)
            
            if (ep + 1) % 50 == 0:
                loss = np.mean((eps_pred - eps)**2)
                print(f"  DDPM Ep {ep+1}/{epochs} | Loss: {loss:.6f}")

    def sample(self, n: int = 1, guidance_scale: float = 3.0) -> np.ndarray:
        """
        [P2.2] Reverse Diffusion Loop with PINN-classifier guidance.
        """
        x = np.random.randn(n, self.dim)
        
        for t_val in reversed(range(self.steps)):
            t = np.full((n,), t_val)
            z = np.random.randn(n, self.dim) if t_val > 0 else 0
            
            eps_pred = self._forward_model(x, t)
            
            # [P2.2] PINN-Classifier Guidance
            if self.pinn is not None and guidance_scale > 0:
                # Approximate grad of log p(ASME|x) to push toward safety
                dx = 1e-3
                grad_p = np.zeros_like(x)
                for i in range(self.dim):
                    x_plus = x.copy(); x_plus[:, i] += dx
                    x_minus = x.copy(); x_minus[:, i] -= dx
                    s_plus = self.pinn.predict(x_plus)[:, 0]
                    s_minus = self.pinn.predict(x_minus)[:, 0]
                    # Gradient points toward HIGHER stress, so we SUBTRACT it from noise
                    grad_p[:, i] = (s_plus - s_minus) / (2 * dx)
                
                # Steer AWAY from higher stress
                eps_pred += guidance_scale * grad_p * np.sqrt(1 - self.alpha_hat[t_val])
                
            ah_t = self.alpha_hat[t_val]
            a_t  = self.alpha[t_val]
            
            # One denoising step
            x = (1 / np.sqrt(a_t)) * (x - (1 - a_t) / np.sqrt(1 - ah_t) * eps_pred) + np.sqrt(self.beta[t_val]) * z
            
        return x
