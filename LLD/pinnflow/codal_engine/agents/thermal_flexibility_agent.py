"""
pinnflow/codal_engine/agents/thermal_flexibility_agent.py
──────────────────────────────────────────────────────────
Phase 4-A — ASME B31.3 Thermal Flexibility Check

Checks displacement stress range SE against allowable SA per
ASME B31.3 Appendix C.

Formulae (simplified proxy for 10-D/16-D design vectors):
    SE_proxy = E × α_thermal × |ΔT|    (thermal strain × Young's modulus)
    SA       = f × (1.25 × Sc + 0.25 × Sh)   [MPa]
    f        = 1.0 for < 7000 cycles (default for initial design)
    Sc       = 138 MPa  (A106-B cold allowable)
    Sh       = 131 MPa  (A106-B hot allowable at 300°C)
    Combined SA ≈ 345 MPa

References:
    ASME B31.3-2020 Appendix C — Flexibility and SIF
    ASME B31.3-2020 Table A-1  — Allowable Stresses
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pinnflow.codal_engine.agents.base import CritiqueAgent

# Material constants (A106 Grade B — default process piping material)
_E_GPA: float        = 200.0      # Young's modulus in GPa
_E_MPA: float        = _E_GPA * 1000.0   # MPa
_ALPHA_STEEL: float  = 12.0e-6   # thermal expansion coefficient 1/°C
_SC_MPA: float       = 138.0     # cold allowable stress (MPa)
_SH_MPA: float       = 131.0     # hot allowable stress at 300°C (MPa)
_F_FACTOR: float     = 1.0       # stress-range reduction factor (< 7000 cycles)
_SA_MPA: float       = _F_FACTOR * (1.25 * _SC_MPA + 0.25 * _SH_MPA)   # ≈ 205 MPa


class ThermalFlexibilityAgent(CritiqueAgent):
    """
    Checks thermal expansion stress range against ASME B31.3 Appendix C limit.

    Input contract (design vector col mapping):
        col 5: delta_T_degC  — operating temperature differential vs ambient
        col 0: diameter_mm
        col 1: thickness_mm
    """

    CLAUSE = "ASME B31.3 Appendix C — Thermal Flexibility"

    def evaluate(self, design: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate thermal flexibility compliance.

        Args:
            design:  1-D array-like with at least 6 elements (standard design vector).
            context: dict (unused for this check; retained for interface compatibility).

        Returns:
            dict with keys:
                "agent"       — agent class name
                "clause"      — ASME clause reference
                "SE_MPa"      — computed displacement stress range [MPa]
                "SA_MPa"      — allowable displacement stress range [MPa]
                "pass"        — True if SE ≤ SA
                "penalty"     — 0.0 if pass, else (SE - SA) / SA normalised [0, 1]
                "violation"   — human-readable violation string (empty if pass)
                "recommendation" — corrective action string (empty if pass)
        """
        x = np.asarray(design, dtype=float).ravel()

        if len(x) < 6:
            return self._missing_data("Design vector too short (need ≥ 6 elements).")

        delta_T = abs(float(x[5]))   # °C

        # Simplified thermal strain × modulus proxy
        SE_proxy = _E_MPA * _ALPHA_STEEL * delta_T   # MPa

        passes = bool(SE_proxy <= _SA_MPA)
        penalty = 0.0
        violation = ""
        recommendation = ""

        if not passes:
            excess = SE_proxy - _SA_MPA
            penalty = float(np.clip(excess / _SA_MPA, 0.0, 1.0))
            violation = (
                f"Displacement stress SE = {SE_proxy:.1f} MPa exceeds "
                f"allowable SA = {_SA_MPA:.1f} MPa "
                f"(ASME B31.3 Appendix C, ΔT = {delta_T:.1f}°C)."
            )
            recommendation = (
                "Add expansion loops or bellows to reduce thermal strain.  "
                "Consider re-routing to shorten unsupported spans, "
                "or use a flexible material with lower E or α."
            )

        return {
            "agent":          self.__class__.__name__,
            "clause":         self.CLAUSE,
            "SE_MPa":         round(SE_proxy, 2),
            "SA_MPa":         round(_SA_MPA, 2),
            "delta_T_deg":    round(delta_T, 2),
            "pass":           passes,
            "penalty":        round(penalty, 4),
            "violation":      violation,
            "recommendation": recommendation,
        }

    @staticmethod
    def _missing_data(reason: str) -> Dict[str, Any]:
        return {
            "agent":   "ThermalFlexibilityAgent",
            "clause":  "ASME B31.3 Appendix C",
            "pass":    False,
            "penalty": 0.5,
            "violation": reason,
            "recommendation": "Provide complete design vector.",
        }
