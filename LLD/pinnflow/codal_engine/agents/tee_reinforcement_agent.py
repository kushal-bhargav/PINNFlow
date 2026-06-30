"""
pinnflow/codal_engine/agents/tee_reinforcement_agent.py
────────────────────────────────────────────────────────
Phase 4-B — ASME B31.3 Tee Branch Reinforcement Check

Checks branch connection area replacement per ASME B31.3 § 304.3.

Key check:
    d_ratio = d_branch / d_run  (= shape_param, col 9)
    Requirements:
        ratio < 0.25  → half-coupling or threadolet acceptable
        0.25 ≤ ratio < 0.50 → sockolet / weldolet required
        0.50 ≤ ratio < 0.75 → full-reinforced weldolet required
        ratio ≥ 0.75  → reducing tee or integrally reinforced fitting required

    Area check (simplified):
        A_req  = d_branch_mm × t_branch_min   (approximate per §304.3.1)
        A_avail = (2 × d_branch_mm) × (t_run_actual - t_run_min)
        Condition: A_avail ≥ A_req

References:
    ASME B31.3-2020 § 304.3 — Reinforcement of Welded Branch Connections
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pinnflow.codal_engine.agents.base import CritiqueAgent

# Allowable stress for area check (A106-B, cold)
_S_MPA: float = 138.0


def _min_thickness_mm(D_mm: float, P_mpa: float, S_mpa: float = _S_MPA) -> float:
    """ASME B31.3 § 304.1.2 minimum required wall thickness [mm]."""
    if P_mpa <= 0 or D_mm <= 0:
        return 1.0
    t_min = P_mpa * D_mm / (2.0 * S_mpa + 0.4 * P_mpa)
    return max(t_min, 1.0)


def _required_fitting(d_ratio: float) -> str:
    if d_ratio < 0.25:
        return "half-coupling or threadolet"
    elif d_ratio < 0.50:
        return "sockolet or weldolet"
    elif d_ratio < 0.75:
        return "full-reinforced weldolet"
    else:
        return "reducing tee or integrally reinforced tee fitting"


class TeeReinforcementAgent(CritiqueAgent):
    """
    Checks branch connection area replacement for tee junctions
    per ASME B31.3 § 304.3.

    Design vector mapping:
        col 0: run pipe diameter_mm    (d_run)
        col 1: run wall thickness_mm   (t_run)
        col 3: pressure_MPa
        col 8: shape_id                (must equal 2 for tee)
        col 9: shape_param             (= d_branch / d_run)
    """

    CLAUSE = "ASME B31.3 § 304.3 — Branch Connection Reinforcement"

    def evaluate(self, design: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        x = np.asarray(design, dtype=float).ravel()

        if len(x) < 10:
            return self._missing_data("Design vector needs ≥ 10 elements for tee check.")

        shape_id = float(x[8])
        if round(shape_id) != 2:
            # Not a tee — not applicable.  Return a clearly marked N/A so that:
            #   - The codal violation counter does NOT count this as a FAIL
            #   - The convergence gate can detect shape mismatch if needed
            return {
                "agent":          self.__class__.__name__,
                "clause":         self.CLAUSE,
                "pass":           True,
                "not_applicable": True,
                "status":         "N/A",
                "penalty":        0.0,
                "violation":      "",
                "recommendation": (
                    f"Not applicable: design has shape_id={round(shape_id)}, "
                    "not a tee junction (shape_id=2). No reinforcement check required."
                ),
            }

        D_run_mm  = max(float(x[0]), 1.0)
        t_run_mm  = max(float(x[1]), 1.0)
        P_mpa     = max(float(x[3]), 0.0)
        d_ratio   = float(np.clip(x[9], 0.0, 1.0))

        D_branch_mm = D_run_mm * d_ratio
        t_branch_min = _min_thickness_mm(D_branch_mm, P_mpa)
        t_run_min    = _min_thickness_mm(D_run_mm, P_mpa)

        # Simplified area replacement check (ASME B31.3 §304.3.2)
        A_required   = D_branch_mm * t_branch_min
        excess_t     = max(t_run_mm - t_run_min, 0.0)
        A_available  = 2.0 * D_branch_mm * excess_t

        area_ok    = A_available >= A_required
        fitting    = _required_fitting(d_ratio)
        passes     = area_ok

        verbose = context.get("verbose", False)
        if verbose:
            print(f"[TeeReinforcementAgent] Evaluating: D_run={D_run_mm:.1f}mm, t_run={t_run_mm:.2f}mm, P={P_mpa:.2f}MPa | d/D ratio={d_ratio:.2f} (Required fitting: {fitting})")
            print(f"[TeeReinforcementAgent] Area check: A_required={A_required:.1f} mm² | A_available={A_available:.1f} mm² | status={'PASS' if passes else 'FAIL'}")

        penalty       = 0.0
        violation     = ""
        recommendation = ""

        if not passes:
            shortfall = A_required - A_available
            penalty   = float(np.clip(shortfall / max(A_required, 1e-3), 0.0, 1.0))
            violation = (
                f"Insufficient branch reinforcement: "
                f"A_available = {A_available:.1f} mm² < A_required = {A_required:.1f} mm² "
                f"(d/D = {d_ratio:.2f}, {self.CLAUSE})."
            )
            recommendation = (
                f"Use a {fitting} at this branch connection.  "
                f"Add a reinforcing pad to increase available area by {shortfall:.0f} mm²."
            )
            if verbose:
                print(f"[TeeReinforcementAgent] Violation: {violation}")
        else:
            recommendation = f"Use a {fitting} at this branch connection."

        return {
            "agent":          self.__class__.__name__,
            "clause":         self.CLAUSE,
            "d_ratio":        round(d_ratio, 3),
            "A_required_mm2": round(A_required, 1),
            "A_available_mm2": round(A_available, 1),
            "required_fitting": fitting,
            "pass":           passes,
            "penalty":        round(penalty, 4),
            "violation":      violation,
            "recommendation": recommendation,
        }

    @staticmethod
    def _missing_data(reason: str) -> Dict[str, Any]:
        return {
            "agent":   "TeeReinforcementAgent",
            "clause":  "ASME B31.3 § 304.3",
            "pass":    False,
            "penalty": 0.5,
            "violation": reason,
            "recommendation": "Provide complete design vector.",
        }
