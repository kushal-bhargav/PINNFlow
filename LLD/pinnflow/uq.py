"""
pinnflow/uq.py
──────────────
MODULE 10 — Uncertainty Quantification (Deep Ensembles) [V6]

This module implements a deep ensemble of PINNs to estimate epistemic 
uncertainty and provide calibrated safe boundaries for RL.
"""
from __future__ import annotations
import numpy as np
from pinnflow.pinn import MultiTaskPINN
from pinnflow.geometry.features import GEOMETRY_CLASS_NAMES
from pinnflow.models.moe_pinn import MoEPINN

class PINNEnsemble:
    """
    [P5.1] Deep ensemble of N MultiTaskPINN models.
    At inference, returns mean and std (epistemic uncertainty).
    """
    def __init__(self, n_ensemble=3, model_cls=None, use_moe: bool = False, mc_dropout_p: float = 0.1, **pinn_kwargs):
        cls = model_cls or (MoEPINN if use_moe else MultiTaskPINN)
        self.models = [cls(**pinn_kwargs) for _ in range(n_ensemble)]
        self.n_ensemble = n_ensemble
        self.mc_dropout_p = mc_dropout_p

    def fit(self, X_obs, Y_obs, X_coll=None, epochs=200, **kwargs):
        """Fit each ensemble member with a different seed."""
        for i, model in enumerate(self.models):
            np.random.seed(i * 42) # Varied initialisation
            print(f"  Training Ensemble Member {i+1}/{self.n_ensemble}...")
            model.fit(X_obs, Y_obs, X_coll=X_coll, epochs=epochs, verbose=False, **kwargs)

    def predict_with_uncertainty(self, X: np.ndarray) -> dict:
        """
        [P5.1] Returns ensemble mean and standard deviation.
        """
        preds = []
        for m in self.models:
            preds.append(m.predict(X))
        
        preds = np.stack(preds) # (N_ens, N_samples, 2)
        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        
        gate = None
        if hasattr(self.models[0], "gating"):
            gate = self.models[0].gating.forward(X)
        sigma_std = std[:, 0]
        denom = np.maximum(np.abs(mean[:, 0]) + sigma_std, 1e-6)
        uncertainty_score = np.clip(sigma_std / denom, 0.0, 1.0)
        out = {
            "mean": mean,
            "std": std,
            "ci_hi": mean + 1.96 * std,
            "ci_lo": mean - 1.96 * std,
            "sigma_mean": mean[:, 0],
            "sigma_std": sigma_std,
            "sigma_ci_lo": mean[:, 0] - 1.96 * sigma_std,
            "sigma_ci_hi": mean[:, 0] + 1.96 * sigma_std,
            "uncertainty_score": uncertainty_score,
        }
        if gate is not None:
            labels = np.argmax(gate, axis=1)
            out["geometry_class"] = np.array([GEOMETRY_CLASS_NAMES[int(i)] for i in labels])
            out["gate_confidence"] = np.max(gate, axis=1)
        return out

    def predict_mc_dropout(self, X: np.ndarray, n_samples: int = 50) -> dict:
        """Approximate MC dropout by jittering predictions when explicit dropout is unavailable."""
        base = self.predict(X)
        samples = []
        for _ in range(n_samples):
            samples.append(base * (1.0 + np.random.randn(*base.shape) * self.mc_dropout_p))
        samples = np.stack(samples)
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        return {
            "mean": mean,
            "std": std,
            "ci_hi": mean + 1.96 * std,
            "ci_lo": mean - 1.96 * std,
            "sigma_mean": mean[:, 0],
            "sigma_std": std[:, 0],
            "sigma_ci_lo": mean[:, 0] - 1.96 * std[:, 0],
            "sigma_ci_hi": mean[:, 0] + 1.96 * std[:, 0],
            "uncertainty_score": np.clip(std[:, 0] / np.maximum(np.abs(mean[:, 0]) + std[:, 0], 1e-6), 0.0, 1.0),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transparent interface to match single-model API."""
        return self.predict_with_uncertainty(X)["mean"]

def evaluate_calibration(ensemble, X_test, Y_test):
    """
    [P5.3] Checks coverage of 95% confidence intervals.
    """
    uq = ensemble.predict_with_uncertainty(X_test)
    
    # Stress calibration (index 0)
    Y_s = Y_test[:, 0]
    covered = (Y_s >= uq["ci_lo"][:, 0]) & (Y_s <= uq["ci_hi"][:, 0])
    rate = np.mean(covered)
    
    print(f"  UQ Calibration: 95% CI Coverage = {rate:.2%}")
    return rate
