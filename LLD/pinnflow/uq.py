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

class PINNEnsemble:
    """
    [P5.1] Deep ensemble of N MultiTaskPINN models.
    At inference, returns mean and std (epistemic uncertainty).
    """
    def __init__(self, n_ensemble=3, **pinn_kwargs):
        self.models = [MultiTaskPINN(**pinn_kwargs) for _ in range(n_ensemble)]
        self.n_ensemble = n_ensemble

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
        
        return {
            "mean": mean,
            "std": std,
            "ci_hi": mean + 1.96 * std,
            "ci_lo": mean - 1.96 * std
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
