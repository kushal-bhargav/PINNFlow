"""
pinnflow/vae.py
────────────────
MODULE 4 — Constraint-Aware VAE  [N2]

[P2 FIX] KL annealing (β: 0→1) + free bits per dimension.
[P1 FIX] True VAE encoder backward:
  - Reconstruction gradient flows from decoder into encoder.
  - KL gradients computed for both mu and lv heads.
  - lv_l.backward() is called (was previously never updated).
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from pinnflow.layers import AdamLayer


class CAVAE:
    """
    [N2] Constraint-Aware VAE.

    Physics penalty in ELBO pushes generated layouts toward ASME feasibility.
    [P2] KL annealing: β ramps 0→1 over first half of training.
         Free bits: min KL per dim = FREE_BITS (prevents posterior collapse).
    """

    FREE_BITS = 0.5  # minimum KL nats per dimension
    DEFAULT_BOUNDS_10 = np.array(
        [
            [114, 620],
            [4, 22],
            [5, 150],
            [1, 20],
            [0, 150],
            [-40, 80],
            [0.5, 9],
            [0.3, 0.8],
            [0, 2],
            [0.3, 1.5],
        ],
        dtype=float,
    )
    DEFAULT_BOUNDS_8 = DEFAULT_BOUNDS_10[:8].copy()
    DEFAULT_BOUNDS_16 = np.vstack(
        [
            DEFAULT_BOUNDS_10[:8],
            np.array(
                [
                    [0, 3],
                    [0.3, 5.0],
                    [0, 3100],
                    [0, 1.0],
                    [-1, 1],
                    [-1, 1],
                    [0, 0.2],
                    [0, 1.0e7],
                ],
                dtype=float,
            ),
        ]
    )

    def __init__(
        self,
        x_dim: int = 8,
        z_dim: int = 16,
        hidden: tuple = (128, 64),
        lr: float = 3e-3,
        gamma_phys: float = 2.0,
    ):
        if x_dim not in (8, 10, 16):
            raise ValueError("CAVAE currently supports x_dim=8, x_dim=10, or x_dim=16.")

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.lr    = lr
        self.gamma = gamma_phys
        self.scaler = StandardScaler()
        if x_dim == 16:
            self.bounds = self.DEFAULT_BOUNDS_16.copy()
        elif x_dim == 10:
            self.bounds = self.DEFAULT_BOUNDS_10.copy()
        else:
            self.bounds = self.DEFAULT_BOUNDS_8.copy()
        self.condition_dim = 4

        # ── Encoder ──────────────────────────────────────────────────────────
        enc = [x_dim + self.condition_dim] + list(hidden)
        self.enc_l = [AdamLayer(enc[i], enc[i + 1], "swish", lr) for i in range(len(enc) - 1)]
        self.mu_l  = AdamLayer(hidden[-1], z_dim, "linear", lr)
        self.lv_l  = AdamLayer(hidden[-1], z_dim, "linear", lr)

        # ── Decoder ──────────────────────────────────────────────────────────
        dec = [z_dim + self.condition_dim] + list(hidden[::-1]) + [x_dim]
        self.dec_l = [
            AdamLayer(dec[i], dec[i + 1], "swish" if i < len(dec) - 2 else "linear", lr)
            for i in range(len(dec) - 1)
        ]

        self.history = {k: [] for k in ["elbo", "recon", "kl", "phys"]}
        self.is_trained = False

    # ── Encoder / Decoder helpers ─────────────────────────────────────────────
    def _encode(self, x: np.ndarray, condition: np.ndarray):
        h = np.concatenate([x, condition], axis=1)
        for l in self.enc_l:
            h = l.forward(h)
        mu = self.mu_l.forward(h)
        lv = np.clip(self.lv_l.forward(h), -4, 4)
        return mu, lv

    def _decode(self, z: np.ndarray, condition: np.ndarray) -> np.ndarray:
        h = np.concatenate([z, condition], axis=1)
        for l in self.dec_l:
            h = l.forward(h)
        return h

    def _reparam(self, mu: np.ndarray, lv: np.ndarray) -> np.ndarray:
        return mu + np.exp(0.5 * lv) * np.random.randn(*mu.shape)

    def _condition_vector(self, condition=None) -> np.ndarray:
        if condition is None:
            return np.zeros(4, dtype=float)

        if isinstance(condition, dict):
            max_p = float(condition.get("max_p", condition.get("pressure", 0.0)))
            max_t = float(condition.get("max_t", condition.get("temperature", 0.0)))
            topology = str(condition.get("topology", "")).lower()
            penalty_weight = float(condition.get("codal_penalty_weight", 1.0))
            topology_code = 2.0 if "fsi" in topology else (1.0 if "refinery" in topology else 0.0)
            return np.array(
                [
                    np.clip(max_p / 100.0, 0.0, 2.0),
                    np.clip((max_t + 40.0) / 200.0, 0.0, 2.0),
                    topology_code,
                    np.clip(penalty_weight / 5.0, 0.0, 2.0),
                ],
                dtype=float,
            )

        arr = np.asarray(condition, dtype=float).flatten()
        if arr.size >= 4:
            return arr[:4].astype(float)
        padded = np.zeros(4, dtype=float)
        padded[: arr.size] = arr
        return padded

    def _derive_conditions_from_raw(self, X_raw: np.ndarray) -> np.ndarray:
        x = np.asarray(X_raw, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] < 8:
            raise ValueError("Raw design data must include at least 8 columns to infer conditions.")

        cond = np.zeros((len(x), self.condition_dim), dtype=float)
        cond[:, 0] = np.clip(x[:, 3] / 20.0, 0.0, 2.0)
        cond[:, 1] = np.clip(x[:, 6] / 9.0, 0.0, 2.0)
        cond[:, 2] = np.clip(x[:, 7] / 0.8, 0.0, 2.0)
        cond[:, 3] = np.clip((x[:, 5] + 40.0) / 120.0, 0.0, 2.0)
        return cond

    def _condition_matrix(self, n: int, condition=None, X_raw=None) -> np.ndarray:
        if condition is None:
            if X_raw is not None:
                cond = self._derive_conditions_from_raw(X_raw)
                if len(cond) == n:
                    return cond
                return np.repeat(cond[:1], n, axis=0)
            return np.zeros((n, self.condition_dim), dtype=float)

        if isinstance(condition, dict):
            row = self._condition_vector(condition)
            return np.tile(row, (n, 1))

        arr = np.asarray(condition, dtype=float)
        if arr.ndim == 0:
            arr = np.full((n, self.condition_dim), float(arr))
        elif arr.ndim == 1:
            if arr.size < self.condition_dim:
                padded = np.zeros(self.condition_dim, dtype=float)
                padded[: arr.size] = arr
                arr = padded
            arr = np.tile(arr[: self.condition_dim], (n, 1))
        elif arr.ndim == 2:
            if arr.shape[1] < self.condition_dim:
                padded = np.zeros((arr.shape[0], self.condition_dim), dtype=float)
                padded[:, : arr.shape[1]] = arr
                arr = padded
            arr = arr[:, : self.condition_dim]
            if arr.shape[0] == 1 and n > 1:
                arr = np.repeat(arr, n, axis=0)
        else:
            raise ValueError("Unsupported condition format.")

        if len(arr) != n:
            if len(arr) == 1:
                arr = np.repeat(arr, n, axis=0)
            else:
                raise ValueError(f"Expected {n} condition rows, got {len(arr)}.")
        return arr.astype(float)

    def _map_to_bounds(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=float).copy()
        if out.ndim == 1:
            out = out.reshape(1, -1)

        if out.shape[1] != self.x_dim:
            raise ValueError(f"Expected {self.x_dim} columns, got {out.shape[1]}.")

        for idx, (low, high) in enumerate(self.bounds):
            span = max(high - low, 1e-6)
            if self.x_dim >= 10 and idx == 8:
                out[:, idx] = np.clip(np.rint(out[:, idx]), low, high)
            else:
                out[:, idx] = low + span * (1.0 / (1.0 + np.exp(-out[:, idx])))

        if self.x_dim >= 10:
            out[:, 8] = np.clip(np.rint(out[:, 8]), self.bounds[8, 0], self.bounds[8, 1])
        return out

    def _score_candidates(self, candidates: np.ndarray, pinn=None, condition=None) -> np.ndarray:
        if len(candidates) == 0:
            return np.array([])

        if pinn is not None and self.x_dim >= 10:
            preds = pinn.predict(candidates)
            sigma = preds[:, 0]
            delta_p = preds[:, 1]
        else:
            sigma = (candidates[:, 3] * candidates[:, 0]) / (2.0 * np.maximum(candidates[:, 1], 1.0))
            delta_p = candidates[:, 6] * candidates[:, 2] / np.maximum(candidates[:, 0], 1.0)

        if self.x_dim >= 10:
            target_pressure = np.clip(self._condition_vector(condition)[0] * 10.0, 1.0, 20.0)
            pressure_gap = np.abs(candidates[:, 3] - target_pressure)
        else:
            pressure_gap = np.abs(candidates[:, 3] - np.mean(self.bounds[3]))

        cost_proxy = candidates[:, 0] * candidates[:, 1] * np.maximum(candidates[:, 2], 1.0) / 1e5
        feasibility = np.maximum(0.0, sigma - 200.0) / 200.0
        hydraulic = np.clip(delta_p / 150.0, 0.0, 1.5)
        score = 1.0 - (
            0.45 * np.clip(feasibility, 0.0, 1.0)
            + 0.30 * np.clip(hydraulic, 0.0, 1.0)
            + 0.15 * np.clip(pressure_gap / 10.0, 0.0, 1.0)
            + 0.10 * np.clip(cost_proxy, 0.0, 1.0)
        )
        return score

    # ── Physics penalty ───────────────────────────────────────────────────────
    def _phys_penalty(self, xr: np.ndarray) -> float:
        """ASME B31.3 hoop-stress violation penalty."""
        d, t, P = xr[:, 0], xr[:, 1], xr[:, 3]
        sig = P * d / (2 * np.maximum(t, 1.0))
        viol = np.maximum(0, sig - 200.0)
        return float(np.mean(viol ** 2) / 5000.0)

    def _grad_phys(self, xr: np.ndarray) -> np.ndarray:
        """Analytical gradient of physics penalty w.r.t. raw inputs."""
        d, t, P = xr[:, 0], xr[:, 1], xr[:, 3]
        sig = P * d / (2 * np.maximum(t, 1.0))
        viol = np.maximum(0, sig - 200.0)
        dg = np.zeros_like(xr)
        dg[:, 0] = (P / (2 * t)) * (2 * viol / 5000.0)
        dg[:, 3] = (d / (2 * t)) * (2 * viol / 5000.0)
        dg[:, 1] = (-P * d / (2 * t**2)) * (2 * viol / 5000.0)
        return dg

    # ── Training ─────────────────────────────────────────────────────────────
    def fit(
        self,
        X_raw: np.ndarray,
        epochs: int = 300,
        batch: int = 128,
        verbose: bool = True,
        conditions=None,
    ) -> None:
        Xs = self.scaler.fit_transform(X_raw)
        self.is_trained = True
        n  = len(Xs)
        conds = self._condition_matrix(n, condition=conditions, X_raw=X_raw)

        for ep in range(1, epochs + 1):
            # [P2] KL annealing: β ramps 0→1 over first 150 epochs
            beta = min(1.0, ep / (epochs * 0.5))

            idx = np.random.permutation(n)
            ee = er = ek = ep_ph = 0.0
            nb = 0

            for st in range(0, n, batch):
                sl = idx[st: st + batch]
                xb = Xs[sl]
                cb = conds[sl]

                mu, lv   = self._encode(xb, cb)
                z        = self._reparam(mu, lv)
                xh       = self._decode(z, cb)
                xhr      = self.scaler.inverse_transform(xh)

                # ── Losses ───────────────────────────────────────────────────
                recon = np.mean((xh - xb) ** 2)
                # [P0.5] Free bits: clamp per-dim KL from below
                kl_per_dim = -0.5 * (1 + lv - mu ** 2 - np.exp(lv))
                kl_free    = np.maximum(kl_per_dim, self.FREE_BITS)
                kl    = kl_free.mean()
                phys  = self._phys_penalty(xhr)
                loss  = recon + beta * kl + self.gamma * phys

                # ── Decoder backward ─────────────────────────────────────────
                # dL/dx_scaled = d_recon + d_phys
                g_recon = 2 * (xh - xb) / max(len(sl), 1)
                
                # [P0.5] Physics gradient corrected for scaling
                # dPenalty/dx_scaled = (dPenalty/dx_raw) / scale
                d_phys_raw = self._grad_phys(xhr)
                g_phys = self.gamma * d_phys_raw / (self.scaler.scale_ + 1e-8)
                
                g = g_recon + g_phys / max(len(sl), 1)
                for l in reversed(self.dec_l):
                    g = l.backward(g)

                # ── [P1 FIX] True VAE encoder backward ───────────────────────
                # g is the gradient w.r.t. z (shape: batch × z_dim) after
                # backprop through the decoder + physics gradient.
                g_enc = g[:, : self.z_dim]  # dL/dz, shape (batch, z_dim)

                # KL gradients (w.r.t. mu and lv outputs, shape batch×z_dim)
                g_mu = beta * mu / max(len(sl), 1)
                g_lv = beta * 0.5 * (np.exp(lv) - 1) / max(len(sl), 1)

                # Reparameterisation: z = mu + exp(0.5·lv)·ε
                #   dz/dmu = 1           → reconstruction grad for mu = g_enc
                #   dz/dlv = 0.5·(z−mu) → reconstruction grad for lv
                g_mu_total = g_mu + g_enc                      # (batch, z_dim)
                g_lv_total = g_lv + g_enc * 0.5 * (z - mu)    # (batch, z_dim)

                # Backprop through heads — .backward() accepts gradient w.r.t.
                # the layer's OUTPUT and returns gradient w.r.t. its INPUT.
                # It also applies the @ W.T projection internally.
                ge_mu = self.mu_l.backward(g_mu_total)   # → (batch, hidden[-1])
                ge_lv = self.lv_l.backward(g_lv_total)   # → (batch, hidden[-1])

                # Merge and propagate through shared encoder layers
                ge = ge_mu + ge_lv
                for l in reversed(self.enc_l):
                    ge = l.backward(ge)

                ee += loss; er += recon; ek += kl; ep_ph += phys; nb += 1

            nb = max(nb, 1)
            for k, v in zip(
                ["elbo", "recon", "kl", "phys"],
                [ee / nb, er / nb, ek / nb, ep_ph / nb],
            ):
                self.history[k].append(v)

            if verbose and ep % 100 == 0:
                print(
                    f"  CA-VAE Ep {ep:4d}/{epochs} | elbo={ee/nb:.5f} "
                    f"recon={er/nb:.5f} kl={ek/nb:.5f} phys={ep_ph/nb:.5f} beta={beta:.2f}"
                )

    def fit_with_gaslib(
        self,
        synthetic_X: np.ndarray,
        gaslib_X: np.ndarray,
        blend_ratio: float = 0.5,
        epochs: int = 50,
    ) -> None:
        """Blend synthetic and GasLib samples, then train."""
        from data.gaslib_training_extractor import blend_with_synthetic

        X_blended = blend_with_synthetic(
            gaslib_X, synthetic_X, blend_ratio=blend_ratio
        )
        # We need to compute conditions if not provided, but fit() handles it if conditions=None
        self.fit(X_blended.to_numpy() if hasattr(X_blended, "to_numpy") else X_blended, epochs=epochs, verbose=True)

    # ── Generation helpers ────────────────────────────────────────────────────
    def _clip_designs(self, x: np.ndarray) -> np.ndarray:
        clipped = np.asarray(x, dtype=float).copy()
        if clipped.ndim == 1:
            clipped = clipped.reshape(1, -1)

        if clipped.shape[1] != self.x_dim:
            raise ValueError(f"Expected {self.x_dim} columns, got {clipped.shape[1]}.")

        for idx, (low, high) in enumerate(self.bounds):
            clipped[:, idx] = np.clip(clipped[:, idx], low, high)
        if self.x_dim >= 10:
            clipped[:, 8] = np.clip(np.rint(clipped[:, 8]), self.bounds[8, 0], self.bounds[8, 1])
            # Keep pressure inside the hoop-stress envelope after clipping.
            p_max = 200.0 * 2.0 * clipped[:, 1] / np.maximum(clipped[:, 0], 1.0) * 0.85
            clipped[:, 3] = np.minimum(clipped[:, 3], p_max)
        return clipped

    def initialize_scaler(self, X_raw: np.ndarray) -> None:
        self.scaler.fit(X_raw)

    def _sample_valid_designs(self, n: int) -> np.ndarray:
        x = np.zeros((n, self.x_dim), dtype=float)
        for idx, (low, high) in enumerate(self.bounds):
            if idx == 8 and self.x_dim >= 10:
                x[:, idx] = np.random.randint(int(low), int(high) + 1, n)
            else:
                x[:, idx] = np.random.uniform(low, high, n)
        return self._clip_designs(x)

    def generate(
        self,
        n: int = 100,
        condition=None,
        pinn=None,
        top_k: int | None = None,
        candidate_multiplier: int = 8,
    ) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, self.x_dim), dtype=float)

        if not self.is_trained or not hasattr(self.scaler, "mean_"):
            return self._sample_valid_designs(n)

        pool_n = max(int(n), int(top_k or n)) * max(int(candidate_multiplier), 1)
        z = np.random.randn(pool_n, self.z_dim)
        conds = self._condition_matrix(pool_n, condition=condition)
        decoded = self._decode(z, conds)
        decoded = self.scaler.inverse_transform(decoded)
        decoded = self._map_to_bounds(decoded)

        scores = self._score_candidates(decoded, pinn=pinn, condition=condition)
        order = np.argsort(scores)[::-1]
        keep = max(int(top_k or n), 1)
        chosen = decoded[order[:keep]]
        return self._clip_designs(chosen)

    def diversity_score(self, n: int = 500) -> float:
        """
        [P2 FIX] Ratio of generated variance vs training variance.
        A collapsed VAE (0 variance) scores 0, a diverse one scores ~1.
        """
        g = self.generate(n)
        # Assuming we have a cached train_std or we can just use the scaler
        if hasattr(self.scaler, "scale_"):
            train_std = self.scaler.scale_
            gen_std   = np.std(g, axis=0)
            return float(np.mean(gen_std / (train_std + 1e-8)))
        return 1.0

    def csr(self, n: int = 500, pinn=None, limit: float = 200.0):
        """Constraint Satisfaction Rate. Returns (rate, generated_array)."""
        g = self.generate(n)
        if pinn is not None:
            ok = pinn.predict(g)[:, 0] < limit
        else:
            ok = (g[:, 3] * g[:, 0]) / (2 * g[:, 1]) < limit
        return float(ok.mean()), g

    def sample_for_topology(
        self,
        topology_name: str,
        n: int = 20,
    ) -> np.ndarray:
        """
        Sample candidate designs conditioned on topology scale.
        """
        topology_key = str(topology_name).lower()
        if "134" in topology_key:
            # Medium-pressure, mid-scale network
            condition = {"max_p": 60.0, "max_t": 20.0, "topology": "GasLib-134"}
        elif "582" in topology_key:
            # High-density distribution network
            condition = {"max_p": 80.0, "max_t": 15.0, "topology": "GasLib-582"}
        elif "4197" in topology_key:
            # Large transmission network
            condition = {"max_p": 120.0, "max_t": 25.0, "topology": "GasLib-4197"}
        else:
            condition = {"max_p": 50.0, "max_t": 20.0, "topology": "synthetic"}
            
        return self.generate(n=n, condition=condition)


def decode_to_component_config(design: np.ndarray) -> dict:
    """
    Maps 16-D (or 10-D) design vector -> named component configuration.
    """
    x = np.asarray(design).ravel()
    D_mm = float(x[0])
    t_mm = float(x[1])
    L_m  = float(x[2])
    shape_id = int(round(x[8])) if len(x) >= 10 else 0
    shape_param = float(x[9]) if len(x) >= 10 else 1.0

    pipe_config = {
        "NPS_approx_mm": round(D_mm, 1),
        "thickness_mm": round(t_mm, 2),
        "length_m": round(L_m, 2),
    }

    fittings = []
    reducer_config = None
    tee_config = None

    if shape_id == 1:
        fittings.append({"type": f"elbow_LR_R{shape_param:.1f}", "qty": 1})
    elif shape_id == 2:
        branch_D = D_mm * shape_param
        tee_config = {"header_D_mm": round(D_mm, 1), "branch_D_mm": round(branch_D, 1)}
        fittings.append({"type": "tee", "qty": 1})
    elif shape_id == 3:
        D1 = D_mm / shape_param
        reducer_config = {"D1_in_mm": round(D1, 1), "D2_out_mm": round(D_mm, 1)}
        fittings.append({"type": "reducer", "qty": 1})
    
    return {
        "pipe": pipe_config,
        "fittings": fittings,
        "reducer": reducer_config,
        "tee": tee_config,
    }

