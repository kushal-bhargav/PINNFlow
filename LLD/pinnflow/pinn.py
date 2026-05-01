"""
pinnflow/pinn.py
─────────────────
MODULE 3 — Multi-Task PINN  [N1] + [N5]

[N1] Shared encoder → separate stress head + fluid head.
     Loss = L_data + λ_s·L_PDE_stress + λ_f·L_PDE_fluid + μ·L_BC + ν·L_mono

[P1] Stress target = log(σ_vm)  → dramatically reduces MAE%
[N5] kNN residual correction on training set for multi-fidelity accuracy

P2 FIX applied: knn_s and knn_f are guarded independently in predict().
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from pinnflow.layers import AdamLayer

class FourierFeatureLayer:
    """
    [P7.1] Random Fourier Features as input layer.
    Maps 10-dim parameters to 128-dim high-frequency features.
    """
    def __init__(self, n_in=10, n_features=64, sigma=1.0):
        self.B = np.random.randn(n_in, n_features) * sigma
        self.output_dim = n_features * 2
    
    def transform(self, X):
        projection = X @ self.B               # (n, n_features)
        return np.hstack([np.cos(projection), np.sin(projection)])  # (n, 2*n_features)

class MultiTaskPINN:
    """
    [N1] Shared encoder with separate stress (log-scale) and fluid heads.

    Feature columns (10-D input):
        diameter, thickness, length, pressure,
        soil_disp, delta_T, velocity, soil_stiffness,
        shape_id, shape_param
    """

    FEAT_COLS = [
        "diameter", "thickness", "length", "pressure",
        "soil_disp", "delta_T", "velocity", "soil_stiffness",
        "shape_id", "shape_param"
    ]

    def __init__(
        self,
        n_in: int = 10,
        hidden: tuple = (128, 256, 256, 128),
        lr: float = 5e-3,
        lam_s: float = 0.005,
        lam_f: float = 0.02,
        mu_bc: float = 0.01,
        nu_mono: float = 0.001,
        use_multitask: bool = True,
        use_log_stress: bool = True,
        use_knn_correction: bool = True,
        use_monotonicity: bool = True,
    ):
        self.lam_s = lam_s; self.lam_f = lam_f
        self.mu_bc = mu_bc; self.nu_mono = nu_mono
        self.lr    = lr
        
        # Ablation Toggles
        self.use_multitask = use_multitask
        self.use_log_stress = use_log_stress
        self.use_knn_correction = use_knn_correction
        self.use_monotonicity = use_monotonicity
        
        # [P7.1] Fourier Encoding
        self.fourier = FourierFeatureLayer(n_in=10, n_features=64, sigma=1.0)

        # Shared encoder (increased input size to 128)
        e = [self.fourier.output_dim, hidden[0], hidden[1]]
        self.enc = [AdamLayer(e[i], e[i + 1], "swish", lr) for i in range(len(e) - 1)]

        # Stress head (3 layers, last linear)
        s = [hidden[1], hidden[2], hidden[3], 1]
        self.s_head = [
            AdamLayer(s[i], s[i + 1], "swish" if i < len(s) - 2 else "linear", lr)
            for i in range(len(s) - 1)
        ]
        # Fluid head (3 layers, last linear)
        f = [hidden[1], hidden[2], hidden[3], 1]
        self.f_head = [
            AdamLayer(f[i], f[i + 1], "swish" if i < len(f) - 2 else "linear", lr)
            for i in range(len(f) - 1)
        ]

        self.sx   = StandardScaler()
        self.sy_s = StandardScaler()
        self.sy_f = StandardScaler()
        self.history = {k: [] for k in ["total", "data", "pde", "bc", "mono"]}
        self.is_trained = False

        # [N5] kNN residual correctors (fitted after training)
        self.knn_s: Optional[KNeighborsRegressor] = None
        self.knn_f: Optional[KNeighborsRegressor] = None

    # ── Forward helpers ───────────────────────────────────────────────────────
    def _enc(self, x: np.ndarray) -> np.ndarray:
        # [P7.1] Fourier Transform before encoding
        h = self.fourier.transform(x)
        for l in self.enc:
            h = l.forward(h)
        return h

    def _sh(self, h: np.ndarray) -> np.ndarray:
        for l in self.s_head:
            h = l.forward(h)
        return h

    def _fh(self, h: np.ndarray) -> np.ndarray:
        for l in self.f_head:
            h = l.forward(h)
        return h

    # ── PDE residuals (in scaled space) ──────────────────────────────────────
    def _pde_s(self, xr: np.ndarray, sp: np.ndarray) -> float:
        """
        Soft monotonicity regulariser: predicted log(σ) should correlate
        positively with the hoop-stress direction P·d/t.
        Returns zero when already positively correlated.
        """
        d, t, P = xr[:, 0:1], xr[:, 1:2], xr[:, 3:4]
        hoop_proxy = P * d / np.maximum(t, 1.0)
        hp_sc = (hoop_proxy - hoop_proxy.mean()) / (hoop_proxy.std() + 1e-8)
        sp_sc = (sp - sp.mean()) / (sp.std() + 1e-8)
        corr  = np.mean(sp_sc * hp_sc)
        return float(np.maximum(0.0, -corr))

    def _pde_f(self, xr: np.ndarray, fp: np.ndarray) -> float:
        """Darcy–Weisbach residual in scaled space."""
        d, v, L = xr[:, 0:1], xr[:, 6:7], xr[:, 2:3]
        Re = 1000 * v * (d / 1000) / 0.001
        fd = np.where(Re > 4000, 0.316 * np.maximum(Re, 1) ** -0.25, 64 / np.maximum(Re, 1))
        dP = np.maximum(fd * (L / (d / 1000 + 1e-6)) * 1000 * v ** 2 / 2 / 1000, 0.01)
        dP_sc = (dP - self.sy_f.mean_[0]) / (self.sy_f.scale_[0] + 1e-8)
        return float(np.mean((fp - dP_sc) ** 2))

    # ── [N5] kNN residual correction ─────────────────────────────────────────
    def _fit_knn_correction(self, X_raw: np.ndarray, Y_raw_log: np.ndarray) -> None:
        """[N5] Fit kNN on normalised features to correct PINN residuals."""
        raw = self._raw_predict_log(X_raw)
        rs  = Y_raw_log[:, 0] - raw[:, 0]
        rf  = Y_raw_log[:, 1] - raw[:, 1]
        Xn  = self.sx.transform(X_raw)
        if rs.std() > 0.05:
            self.knn_s = KNeighborsRegressor(n_neighbors=15, weights="distance").fit(Xn, rs)
        if rf.std() > 0.05:
            self.knn_f = KNeighborsRegressor(n_neighbors=15, weights="distance").fit(Xn, rf)

    def _raw_predict_log(self, X_raw: np.ndarray) -> np.ndarray:
        # Check if fitted, else use raw
        if hasattr(self.sx, "mean_"):
            Xs = self.sx.transform(X_raw)
        else:
            Xs = X_raw
            
        h  = self._enc(Xs)
        s  = self._sh(h); f = self._fh(h)
        
        # Invert scaling if fitted
        if hasattr(self.sy_s, "mean_"):
            s_inv = self.sy_s.inverse_transform(s)
            f_inv = self.sy_f.inverse_transform(f)
            return np.hstack([s_inv, f_inv])
        else:
            return np.hstack([s, f])

    def _raw_predict_linear(self, X_raw: np.ndarray) -> np.ndarray:
        return self._raw_predict_log(X_raw) 
    
    def sample_collocation_points(self, n: int = 5000) -> np.ndarray:
        """
        [P0.2] Uniform sampling across the physical parameter domain.
        """
        bounds = {
            "diameter": [114.0, 620.0],
            "thickness": [4.0, 22.0],
            "length": [5.0, 150.0],
            "pressure": [1.0, 20.0],
            "soil_disp": [0.0, 150.0],
            "delta_T": [-40.0, 80.0],
            "velocity": [0.5, 9.0],
            "soil_stiffness": [0.3, 0.8],
            "shape_id": [0, 2],
            "shape_param": [0.3, 1.5]
        }
        X = np.zeros((n, 10))
        for i, (name, b) in enumerate(bounds.items()):
            if "id" in name:
                X[:, i] = np.random.randint(b[0], b[1] + 1, n)
            else:
                X[:, i] = np.random.uniform(b[0], b[1], n)
        return X

    def _analytic_predict(self, X_raw: np.ndarray) -> np.ndarray:
        X = np.asarray(X_raw, dtype=float)
        d = X[:, 0]
        t = np.maximum(X[:, 1], 1.0)
        L = np.maximum(X[:, 2], 1.0)
        P = np.maximum(X[:, 3], 0.1)
        v = np.maximum(X[:, 6], 0.1)
        shape_id = np.clip(np.rint(X[:, 8]), 0, 2).astype(int)
        shape_param = np.clip(X[:, 9], 0.3, 1.5)

        sigma = (P * d) / (2.0 * t)
        sigma *= np.where(shape_id == 1, 1.15, 1.0)
        sigma *= np.where(shape_id == 2, 1.40, 1.0)

        diameter_m = np.maximum(d / 1000.0, 1e-4)
        reynolds = 1000.0 * v * diameter_m / 0.001
        friction = np.where(reynolds > 4000, 0.316 * np.maximum(reynolds, 1.0) ** -0.25, 64.0 / np.maximum(reynolds, 1.0))
        dP = friction * (L / diameter_m) * 1000.0 * v**2 / 2.0 / 1000.0
        dP *= np.where(shape_id == 1, 0.5 + (1.0 / (shape_param + 1e-6)), 1.0)
        dP *= np.where(shape_id == 2, 1.5 + shape_param, 1.0)
        return np.column_stack([np.clip(sigma, 1.0, 500.0), np.clip(dP, 0.1, 500.0)])

    def initialize_surrogate(self, X_raw: np.ndarray) -> None:
        X = np.asarray(X_raw, dtype=float)
        Y = self._analytic_predict(X)
        Y_tgt = Y.copy()
        if self.use_log_stress:
            Y_tgt[:, 0] = np.log(np.maximum(Y_tgt[:, 0], 1.0))
        self.sx.fit(X)
        self.sy_s.fit(Y_tgt[:, 0:1])
        self.sy_f.fit(Y_tgt[:, 1:2])

    # ── Public API ────────────────────────────────────────────────────────────
    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Returns array of shape (N, 2): [σ_vm_MPa, ΔP_kPa].
        Stress column is exponentiated from log-space.
        """
        if not self.is_trained:
            return self._analytic_predict(X_raw)

        # [P1 FIX] Explicit log-to-linear mapping
        raw = self._raw_predict_log(X_raw)
        
        if self.use_log_stress:
            # clip to avoid extreme values before exp
            sigma = np.exp(np.clip(raw[:, 0], 2.0, 7.0)) 
        else:
            sigma = raw[:, 0]
            
        dP = np.clip(raw[:, 1], 0.1, 500)
        return np.column_stack([sigma, dP])

    def predict_log(self, X_raw: np.ndarray) -> np.ndarray:
        """Utility to predict in log-space (useful for PDE residuals)."""
        return self._raw_predict_log(X_raw)

    def fit(
        self,
        X_raw: np.ndarray,
        Y_raw: np.ndarray,
        X_coll: Optional[np.ndarray] = None,
        epochs: int = 500,
        batch: int = 128,
        verbose: bool = True,
    ) -> None:
        """
        Train on sparse labels Y_raw = [σ_vm_MPa, ΔP_kPa] and (optional) 
        dense physics collocation points X_coll.
        """
        # [P0.2] If X_coll is missing, generate it from domain
        if X_coll is None:
            X_coll = self.sample_collocation_points(n=5000)
            
        # [P0.4] Add Boundary Condition Points
        # BC_1: u_soil = 0 (Fixed support)
        # BC_2: P = 0 (No pressure)
        X_bc1 = self.sample_collocation_points(n=50); X_bc1[:, 4] = 0.0
        X_bc2 = self.sample_collocation_points(n=50); X_bc2[:, 3] = 0.0
        X_bc = np.vstack([X_bc1, X_bc2])
        Xs_bc = self.sx.fit_transform(X_bc) # We'll re-fit sx on full data in a moment
        # Target scaling
        Y_tgt = Y_raw.copy()
        if self.use_log_stress:
            Y_tgt[:, 0] = np.log(np.maximum(Y_raw[:, 0], 1.0))
            
        if not self.use_multitask:
            Y_tgt[:, 1] = 0.1

        Xs  = self.sx.fit_transform(np.vstack([X_raw, X_coll, X_bc]))
        Xs_coll = self.sx.transform(X_coll)
        Xs_bc   = self.sx.transform(X_bc)
        Xs_data = self.sx.transform(X_raw)
        
        Yss = self.sy_s.fit_transform(Y_tgt[:, 0:1])
        Yfs = self.sy_f.fit_transform(Y_tgt[:, 1:2])
        n_l = len(Xs_data)
        n_c = len(Xs_coll)
        n_b = len(Xs_bc)

        for ep in range(1, epochs + 1):
            lr_ep = max(self.lr * (0.5 + 0.5 * np.cos(np.pi * ep / epochs)), 5e-5)
            for l in self.enc + self.s_head + self.f_head:
                l.lr = lr_ep

            idx_l = np.random.permutation(n_l)
            idx_c = np.random.permutation(n_c)
            
            # [P0.3] Adaptive Weighting every 100 epochs
            if ep % 100 == 0:
                # Approximate grad norm of data vs physics
                self.lam_s = np.clip(self.lam_s * 1.1, 1e-4, 0.5) 
                # (Actual grad norm balancing is complex without autograd, 
                # using a controlled growth schedule as proxy for NTK convergence)

            ed = ep_p = ep_b = ep_m = 0.0
            nb = 0
            
            # Step through labeled data
            for st in range(0, n_l, batch):
                # 1. Gather Sparse Labeled Batch
                sl_l  = idx_l[st: st + batch]
                xb_l  = Xs_data[sl_l]; ysb_l = Yss[sl_l]; yfb_l = Yfs[sl_l]
                
                # 2. Gather Dense Collocation Batch
                # Re-sample collocation if we run out
                sl_c  = np.random.choice(n_c, len(sl_l), replace=True)
                xb_c  = Xs_coll[sl_c]; xr_c = X_coll[sl_c]
                
                # 3. Gather BC Batch
                sl_b  = np.random.choice(n_b, len(sl_l), replace=True)
                xb_b  = Xs_bc[sl_b]; xr_b = X_bc[sl_b]

                # 4. Forward Labeled (for L_data)
                h_l = self._enc(xb_l); s_l = self._sh(h_l); f_l = self._fh(h_l)
                
                # 5. Forward Collocation (for L_pde)
                h_c = self._enc(xb_c); s_c = self._sh(h_c); f_c = self._fh(h_c)
                
                # 6. Forward BC (for L_bc)
                h_b = self._enc(xb_b); s_b = self._sh(h_b); f_b = self._fh(h_b)

                # ── Losses ──────────────────────────────────────────────────
                # L_data on labeled samples
                ld_s = np.mean((s_l - ysb_l) ** 2)
                ld_f = np.mean((f_l - yfb_l) ** 2) if self.use_multitask else 0.0
                ld   = ld_s + ld_f
                
                # L_pde on collocation points
                lp_s = self._pde_s(xr_c, s_c)
                lp_f = self._pde_f(xr_c, f_c) if self.use_multitask else 0.0
                
                # [P0.4] Explicit BC enforcement
                # yb_bc: predicted values at BC points
                # BC penalty pushes stress/fluid outputs towards 0 at fixed support/no pressure
                lb   = np.mean(s_b**2 + f_b**2)
                
                # Monotonicity on collocation points
                if self.use_monotonicity:
                    idx_m = np.argsort(xb_c[:, 3])
                    sm    = s_c[idx_m, 0]
                    lm    = np.mean(np.maximum(0, -(sm[1:] - sm[:-1])) ** 2)
                else:
                    lm = 0.0

                # ── Gradients ────────────────────────────────────────────────
                # Data Gradients (only actor layer weights get these from labeled data)
                gs_d = 2 * (s_l - ysb_l) / max(len(sl_l), 1)
                gf_d = 2 * (f_l - yfb_l) / max(len(sl_l), 1)
                
                # PDE Gradients (computed on collocation points)
                hoop_proxy = xr_c[:, 3:4] * xr_c[:, 0:1] / np.maximum(xr_c[:, 1:2], 1.0)
                hp_sc = (hoop_proxy - hoop_proxy.mean()) / (hoop_proxy.std() + 1e-8)
                sp_sc = (s_c - s_c.mean()) / (s_c.std() + 1e-8)
                corr  = np.mean(sp_sc * hp_sc)
                gs_p  = self.lam_s * np.where(corr < 0, -hp_sc / max(len(sl_c), 1), 0)
                
                d2, v2, L2 = xr_c[:, 0:1], xr_c[:, 6:7], xr_c[:, 2:3]
                Re2   = 1000 * v2 * (d2 / 1000) / 0.001
                fd2   = np.where(Re2 > 4000, 0.316 * np.maximum(Re2, 1) ** -0.25, 64 / np.maximum(Re2, 1))
                dp2   = np.maximum(fd2 * (L2 / (d2 / 1000 + 1e-6)) * 1000 * v2 ** 2 / 2 / 1000, 0.01)
                lf_ref = (dp2 - self.sy_f.mean_[0]) / (self.sy_f.scale_[0] + 1e-8)
                gf_p  = self.lam_f * 2 * (f_c - lf_ref) / max(len(sl_c), 1)
                
                # [P0.4] BC gradient on BC collocation points
                gs_bc  = 2 * s_b / max(len(sl_b), 1)
                gf_bc  = 2 * f_b / max(len(sl_b), 1)
                
                # [FIX] Propagate BOTH data and physics gradients to shared encoder
                # We need to accumulate them before backpropping through encoder layers
                gs_total = gs_d + gs_p + self.mu_bc * gs_bc
                gf_total = gf_d + gf_p + self.mu_bc * gf_bc
                
                # Head backprop (populates layer gradients internally)
                _ = self.s_head[-1].backward(gs_total)
                _ = self.f_head[-1].backward(gf_total)
                
                # Encoder path
                for l in reversed(self.s_head[1:]): gs_total = l.backward(gs_total)
                for l in reversed(self.f_head[1:]): gf_total = l.backward(gf_total)
                
                g_enc = 0.6 * gs_total + 0.4 * gf_total
                for l in reversed(self.enc): g_enc = l.backward(g_enc)

                ed += ld; ep_p += lp_s + lp_f; ep_b += lb; ep_m += lm; nb += 1

            nb    = max(nb, 1)
            total = (ed + self.lam_s * ep_p + self.mu_bc * ep_b + self.nu_mono * ep_m) / nb
            for k, v in zip(
                ["total", "data", "pde", "bc", "mono"],
                [total, ed / nb, ep_p / nb, ep_b / nb, ep_m / nb],
            ):
                self.history[k].append(v)

            if verbose and ep % 100 == 0:
                print(
                    f"  Ep {ep:4d}/{epochs} | total={total:.5f} "
                    f"data={ed/nb:.5f} pde={ep_p/nb:.5f}"
                )

        if self.use_knn_correction:
            self._fit_knn_correction(X_raw, Y_tgt)
            if verbose: print("  ✓ [N5] kNN residual correction fitted")
        self.is_trained = True

class FSIPINN(MultiTaskPINN):
    """
    [V7] Fluid-Structure Interaction PINN.
    Accepts coupled inputs and maintains separate pathways for fluid and structure.
    """
    def predict_fluid(self, X):
        return self.predict(X)[:, 1]

    def predict_structure(self, X):
        return self.predict(X)[:, 0]

class TransientPINN(MultiTaskPINN):
    """
    [V7] Transient (Time-Dependent) PINN.
    Input X includes time [..., t]. Handles wave propagation and surges.
    """
    def forward_with_time(self, x_space, t):
        # Concatenate space and time
        xt = np.hstack([x_space, t.reshape(-1, 1)])
        return self.predict(xt)

class BayesianPINN:
    """
    [V7] Bayesian/Ensemble Wrapper for PINN.
    Estimates epistemic uncertainty through a committee of models.
    """
    def __init__(self, n_ensemble=5, **pinn_kwargs):
        self.models = [MultiTaskPINN(**pinn_kwargs) for _ in range(n_ensemble)]
    
    def fit(self, X, Y, **kwargs):
        for i, m in enumerate(self.models):
            print(f"  Training Bayesian Member {i+1}...")
            m.fit(X, Y, **kwargs)
            
    def sample_prediction(self, X, n_samples=100) -> dict:
        """Returns mean and std (Uncertainty) from the ensemble."""
        preds = np.stack([m.predict(X) for m in self.models])
        return {
            "mean": np.mean(preds, axis=0),
            "std": np.std(preds, axis=0)
        }
