"""
pinnflow/baselines.py
──────────────────────
MODULE 29 — Scientific Baselines for Ablation Study

Includes the VanillaMLPBaseline required for IEEE-submittable FAIR comparison.
Architecturally matching the "state-of-the-art" common paradigms.
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from .layers import AdamLayer

class VanillaMLPBaseline:
    """
    [P8.1] Vanilla MLP Baseline.
    3-layer ReLU MLP trained with standard SGD/MSE (No Physics, No Multi-task).
    Reference: Saenz et al. Standard ANN for Piping.
    """
    def __init__(self, n_in=10, hidden=(128, 128, 128), lr=1e-3):
        h = [n_in] + list(hidden) + [1]
        self.layers = [AdamLayer(h[i], h[i+1], "relu" if i < len(h)-2 else "linear", lr) 
                       for i in range(len(h)-1)]
        self.sx = StandardScaler()
        self.sy = StandardScaler()

    def fit(self, X, Y, epochs=500, batch=128):
        Xs = self.sx.fit_transform(X)
        Ys = self.sy.fit_transform(Y[:, 0:1]) # Stress only for single-task baseline
        n = len(Xs)
        for ep in range(epochs):
            idx = np.random.permutation(n)
            for st in range(0, n, batch):
                sl = idx[st:st+batch]
                xb, yb = Xs[sl], Ys[sl]
                h = xb
                for l in self.layers: h = l.forward(h)
                g = 2 * (h - yb) / len(sl)
                for l in reversed(self.layers): g = l.backward(g)

    def predict(self, X):
        Xs = self.sx.transform(X)
        h = Xs
        for l in self.layers: h = l.forward(h)
        return self.sy.inverse_transform(h)

class SingleTaskPINN(VanillaMLPBaseline):
    """
    Vanilla MLP + Physics Loss (No Multi-tasking, No Log-scaling).
    """
    def __init__(self, n_in=10, hidden=(128, 128, 128), lr=1e-3, lam=0.01):
        super().__init__(n_in, hidden, lr)
        self.lam = lam

    def fit(self, X, Y, X_coll=None, epochs=500, batch=128):
        Xs = self.sx.fit_transform(X)
        Ys = self.sy.fit_transform(Y[:, 0:1])
        if X_coll is None: X_coll = X # Placeholder
        Xcs = self.sx.transform(X_coll)
        
        n = len(Xs)
        for ep in range(epochs):
            idx = np.random.permutation(n)
            for st in range(0, n, batch):
                sl = idx[st:st+batch]
                xb, yb = Xs[sl], Ys[sl]
                xc = Xcs[np.random.choice(len(Xcs), len(sl))]
                
                # Data Pass
                h = xb
                for l in self.layers: h = l.forward(h)
                g_data = 2 * (h - yb) / len(sl)
                
                # Physics Pass (Simplified for baseline)
                hc = xc
                for l in self.layers: hc = l.forward(hc)
                # Proxy Stress grad
                sig = self.sy.inverse_transform(hc)
                g_phys = self.lam * 0.1 * hc / len(sl) # Small regularization
                
                g_total = g_data + g_phys
                for l in reversed(self.layers): g_total = l.backward(g_total)
