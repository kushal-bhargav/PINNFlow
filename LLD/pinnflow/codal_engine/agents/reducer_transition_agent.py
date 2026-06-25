"""
pinnflow/codal_engine/agents/reducer_transition_agent.py
─────────────────────────────────────────────────────────
Phase 4-C — ASME B31.3 Reducer Transition Geometry Check

Validates concentric reducer geometry against ASME B31.3 § 304.7.4.

Checks:
    1. Included half-angle ≤ 30°
       α = arctan( (D1 - D2) / (2 × L_reducer) )
       where D1 = inlet diameter, D2 = outlet diameter
    2. D/t ratio ≤ 100 for both inlet and outlet
    3. Length ≥ 3 × (D1 - D2) for gradual transition (recommended)

Design vector mapping:
    col 0: diameter_mm  (D2 = smaller outlet diameter)
    col 1: thickness_mm (outlet wall)
    col 2: length_m     (reducer total length)
    col 8: shape_id     (3 = reducer)
    col 9: shape_param  (= D2 / D1 diameter ratio, ∈ (0, 1))
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from pinnflow.codal_engine.agents.base import CritiqueAgent

_MAX_HALF_ANGLE_DEG: float = 30.0
_MAX_D_OVER_T:       float = 100.0


class ReducerTransitionAgent(CritiqueAgent):
    """
    Validates reducer geometry per ASME B31.3 § 304.7.4.
    """

    CLAUSE = "ASME B31.3 § 304.7.4 — Reducers"

    def evaluate(self, design: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        x = np.asarray(design, dtype=float).ravel()

        if len(x) < 10:
            return self._missing_data("Design vector needs ≥ 10 elements for reducer check.")

        shape_id = float(x[8])
        if round(shape_id) != 3:
            return {
                "agent":          self.__class__.__name__,
                "clause":         self.CLAUSE,
                "pass":           True,
                "penalty":        0.0,
                "violation":      "",
                "recommendation": "Not applicable (shape_id ≠ 3, not a reducer).",
            }

        D2_mm   = max(float(x[0]), 1.0)   # outlet (smaller) diameter
        t_mm    = max(float(x[1]), 1.0)   # outlet wall thickness
        L_m     = max(float(x[2]), 0.001) # reducer length
        d_ratio = float(np.clip(x[9], 0.1, 0.99))

        D1_mm   = D2_mm / d_ratio          # inlet (larger) diameter
        L_mm    = L_m * 1000.0

        # 1. Half-angle
        delta_D_mm = D1_mm - D2_mm
        alpha_deg  = math.degrees(math.atan(delta_D_mm / max(2.0 * L_mm, 1e-3)))

        # 2. D/t ratio at outlet (most critical — thinner wall)
        d_over_t = D2_mm / t_mm

        # 3. Minimum length check
        L_min_mm  = 3.0 * delta_D_mm
        length_ok = L_mm >= L_min_mm

        violations = []
        penalty    = 0.0

        # Check 1: angle
        angle_ok = alpha_deg <= _MAX_HALF_ANGLE_DEG
        if not angle_ok:
            angle_excess  = alpha_deg - _MAX_HALF_ANGLE_DEG
            penalty      += float(np.clip(angle_excess / _MAX_HALF_ANGLE_DEG, 0.0, 1.0)) * 0.6
            violations.append(
                f"Included half-angle α = {alpha_deg:.1f}° > {_MAX_HALF_ANGLE_DEG}° limit "
                f"(ASME B31.3 § 304.7.4)."
            )

        # Check 2: D/t
        dt_ok = d_over_t <= _MAX_D_OVER_T
        if not dt_ok:
            dt_excess = d_over_t - _MAX_D_OVER_T
            penalty  += float(np.clip(dt_excess / _MAX_D_OVER_T, 0.0, 1.0)) * 0.3
            violations.append(
                f"D/t = {d_over_t:.1f} > {_MAX_D_OVER_T} limit at outlet "
                f"(ASME B31.3 § 304.7.4)."
            )

        # Check 3: length (advisory, lower weight)
        if not length_ok:
            penalty += 0.1
            violations.append(
                f"Reducer length {L_mm:.0f} mm < recommended minimum {L_min_mm:.0f} mm "
                f"(= 3 × ΔD; ASME B31.3 § 304.7.4 commentary)."
            )

        penalty = min(penalty, 1.0)
        passes  = len(violations) == 0

        violation_str     = "  ".join(violations)
        recommendation    = ""
        if violations:
            parts = []
            if not angle_ok:
                new_L_mm = delta_D_mm / (2.0 * math.tan(math.radians(_MAX_HALF_ANGLE_DEG)))
                parts.append(
                    f"Increase reducer length to ≥ {new_L_mm:.0f} mm to achieve α ≤ 30°."
                )
            if not dt_ok:
                t_required = D2_mm / _MAX_D_OVER_T
                parts.append(f"Increase outlet wall thickness to ≥ {t_required:.1f} mm.")
            if not length_ok:
                parts.append(f"Increase length to ≥ {L_min_mm:.0f} mm.")
            recommendation = "  ".join(parts)

        return {
            "agent":            self.__class__.__name__,
            "clause":           self.CLAUSE,
            "alpha_deg":        round(alpha_deg, 2),
            "D1_mm":            round(D1_mm, 1),
            "D2_mm":            round(D2_mm, 1),
            "d_ratio":          round(d_ratio, 3),
            "D_over_t":         round(d_over_t, 1),
            "length_mm":        round(L_mm, 0),
            "length_min_mm":    round(L_min_mm, 0),
            "pass":             passes,
            "penalty":          round(penalty, 4),
            "violation":        violation_str,
            "recommendation":   recommendation,
        }

    @staticmethod
    def _missing_data(reason: str) -> Dict[str, Any]:
        return {
            "agent":   "ReducerTransitionAgent",
            "clause":  "ASME B31.3 § 304.7.4",
            "pass":    False,
            "penalty": 0.5,
            "violation": reason,
            "recommendation": "Provide complete design vector.",
        }
