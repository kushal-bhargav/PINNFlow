"""
pinnflow/fatigue.py
───────────────────
MODULE 16 — Fatigue Lifecycle Model [V7]

Predicts cyclic damage and remaining life based on stress time-series from TransientPINN.
"""
from __future__ import annotations
import numpy as np

class FatigueModel:
    """
    [V7] Material Fatigue Lifetime Predictor.
    Uses Miner's Rule for cumulative damage.
    """
    def __init__(self, material_sn_curve: dict = None):
        # Default S-N curve for API 5L X65 Steel (Simplified)
        # log(S) = -0.1 * log(N) + 2.5
        self.sn_curve = material_sn_curve or {"A": 10**6, "m": 3.0}

    def cycles_to_failure(self, stress_range: float) -> float:
        """Calculate N for a given stress range using Basquin's equation."""
        A = self.sn_curve["A"]
        m = self.sn_curve["m"]
        return float(A / (stress_range ** m + 1e-8))

    def damage_index(self, stress_history: np.ndarray, cycles_per_event: int = 1) -> float:
        """
        [V7] Computes Cumulative Damage Index (D).
        D = sum(n_i / N_i). Failure occurs at D >= 1.0.
        """
        damage = 0.0
        # In a real transient case, we'd use Rainflow counting here.
        # For the v7 prototype, we assume the history is a series of peak stresses.
        for sigma in stress_history:
            Ni = self.cycles_to_failure(sigma)
            damage += cycles_per_event / (Ni + 1e-8)
        
        return float(damage)

    def remaining_life_years(self, annual_damage: float) -> float:
        """Estimes years until D=1.0."""
        return 1.0 / (annual_damage + 1e-9)
