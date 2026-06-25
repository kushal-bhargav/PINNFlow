"""
pinnflow/codal_engine/agents/elbow_sif_agent.py
─────────────────────────────────────────────────
Phase 4-E — ASME B31.3 Elbow SIF and Flexibility Factor Check

Checks elbow flexibility factor k and stress intensification factor i
per ASME B31.3 Appendix D.

Formulae (in-plane bending, long-radius elbow):
    h = t × R / (D/2)²    (characteristic parameter)
        where R = R_bend = shape_param × D
    i = 0.9 / h^(2/3)     (SIF, in-plane)
    k = 1.65 / h           (flexibility factor)

Compliance check:
    σ_SIF = i × P × D / (2 × t) ≤ σ_allowable (200 MPa for A106-B)

Design vector mapping:
    col 0: diameter_mm  (D)
    col 1: thickness_mm (t)
    col 3: pressure_MPa (P)
    col 8: shape_id     (1 = elbow)
    col 9: shape_param  (= R/D ratio, typically 1.5 for long-radius)

References:
    ASME B31.3-2020 Appendix D — Flexibility and Stress Intensification Factors
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from pinnflow.codal_engine.agents.base import CritiqueAgent

_ALLOWABLE_STRESS_MPa: float = 200.0   # A106-B basic allowable (conservative)


class ElbowSIFAgent(CritiqueAgent):
    """
    Checks elbow SIF and flexibility factor per ASME B31.3 Appendix D.
    """

    CLAUSE = "ASME B31.3 Appendix D — Elbow SIF and Flexibility Factor"

    def evaluate(self, design: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        x = np.asarray(design, dtype=float).ravel()

        if len(x) < 10:
            return self._missing_data("Design vector needs ≥ 10 elements for elbow check.")

        shape_id = float(x[8])
        if round(shape_id) != 1:
            return {
                "agent":          self.__class__.__name__,
                "clause":         self.CLAUSE,
                "pass":           True,
                "penalty":        0.0,
                "violation":      "",
                "recommendation": "Not applicable (shape_id ≠ 1, not an elbow).",
            }

        D_mm   = max(float(x[0]), 1.0)
        t_mm   = max(float(x[1]), 0.1)
        P_mpa  = max(float(x[3]), 0.0)
        R_over_D = float(np.clip(x[9] if x.shape[0] > 9 else 1.5, 0.5, 10.0))

        # Characteristic parameter h
        D_m    = D_mm / 1000.0
        t_m    = t_mm / 1000.0
        R_m    = R_over_D * D_m   # bend radius in metres
        r_m    = D_m / 2.0        # pipe radius

        h = max(t_m * R_m / r_m ** 2, 1e-6)

        # Flexibility factor and SIF
        k_flex = 1.65 / h
        i_sif  = 0.9 / h ** (2.0 / 3.0)
        i_sif  = max(i_sif, 1.0)   # SIF ≥ 1.0 always

        # SIF-amplified stress
        sigma_hoop = P_mpa * D_mm / (2.0 * t_mm)
        sigma_sif  = i_sif * sigma_hoop

        passes    = sigma_sif <= _ALLOWABLE_STRESS_MPa
        penalty   = 0.0
        violation = ""
        recommendation = ""

        if not passes:
            excess  = sigma_sif - _ALLOWABLE_STRESS_MPa
            penalty = float(np.clip(excess / _ALLOWABLE_STRESS_MPa, 0.0, 1.0))
            violation = (
                f"SIF-amplified stress σ_SIF = {sigma_sif:.1f} MPa > "
                f"allowable {_ALLOWABLE_STRESS_MPa:.0f} MPa "
                f"(i = {i_sif:.2f}, R/D = {R_over_D:.1f}, ASME B31.3 App. D)."
            )
            recommendation = (
                f"Increase bend radius to R/D ≥ {R_over_D * 1.5:.1f} to reduce SIF, "
                f"or increase wall thickness.  "
                f"Current k = {k_flex:.2f} (higher k means more flexibility; "
                f"consider expansion with R/D = 3.0)."
            )

        return {
            "agent":            self.__class__.__name__,
            "clause":           self.CLAUSE,
            "R_over_D":         round(R_over_D, 2),
            "h_parameter":      round(h, 6),
            "SIF_i":            round(i_sif, 3),
            "flexibility_k":    round(k_flex, 3),
            "sigma_hoop_MPa":   round(sigma_hoop, 2),
            "sigma_SIF_MPa":    round(sigma_sif, 2),
            "allowable_MPa":    _ALLOWABLE_STRESS_MPa,
            "pass":             passes,
            "penalty":          round(penalty, 4),
            "violation":        violation,
            "recommendation":   recommendation,
        }

    @staticmethod
    def _missing_data(reason: str) -> Dict[str, Any]:
        return {
            "agent":   "ElbowSIFAgent",
            "clause":  "ASME B31.3 Appendix D",
            "pass":    False,
            "penalty": 0.5,
            "violation": reason,
            "recommendation": "Provide complete design vector.",
        }
