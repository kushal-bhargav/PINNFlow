"""
pinnflow/codal_engine/agents/branch_rules_agent.py
────────────────────────────────────────────────────
Phase 4-F — ASME B31.3 Branch Connection Rules

Checks branch-to-header diameter ratio and required connection type
per ASME B31.3 § 304.3.1 — Extruded Outlets / Welded-In Connections.

Rules:
    d/D < 0.25          → half-coupling or threadolet acceptable
    0.25 ≤ d/D < 0.50  → sockolet or weldolet required
    0.50 ≤ d/D < 0.75  → full-reinforced weldolet required
    d/D ≥ 0.75          → reducing tee or integrally reinforced fitting required

Manufacturing constraint:
    Minimum weld clearance ≥ 25 mm between adjacent branch welds.
    Checked via shape_param (interpreted as d/D for tee junctions).

Design vector mapping:
    col 0: run pipe diameter_mm     (D = header)
    col 1: run wall thickness_mm
    col 8: shape_id                 (2 = tee/branch)
    col 9: shape_param              (d_branch / d_run ratio)

References:
    ASME B31.3-2020 § 304.3.1 — Extruded Outlets and Welded-In Connections
    ASME B31.3-2020 § 328.5.5 — Branch Connection Weld Clearance
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pinnflow.codal_engine.agents.base import CritiqueAgent

# Weld clearance minimum (mm) — from ASME B31.3 §328.5.5
_MIN_WELD_CLEARANCE_MM: float = 25.0

# d/D thresholds
_THRESHOLD_LOW:  float = 0.25
_THRESHOLD_MED:  float = 0.50
_THRESHOLD_HIGH: float = 0.75


def _required_connection(d_ratio: float) -> tuple:
    """Return (required fitting description, penalty weight for violation)."""
    if d_ratio < _THRESHOLD_LOW:
        return "half-coupling or threadolet", 0.0
    elif d_ratio < _THRESHOLD_MED:
        return "sockolet or weldolet", 0.1
    elif d_ratio < _THRESHOLD_HIGH:
        return "full-reinforced weldolet", 0.3
    else:
        return "reducing tee or integrally reinforced fitting", 0.8


class BranchRulesAgent(CritiqueAgent):
    """
    Checks branch-to-header ratio and required connection type
    per ASME B31.3 § 304.3.1.
    """

    CLAUSE = "ASME B31.3 § 304.3.1 — Branch Connection Rules"

    def evaluate(self, design: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        x = np.asarray(design, dtype=float).ravel()

        if len(x) < 10:
            return self._missing_data("Design vector needs ≥ 10 elements for branch check.")

        shape_id = float(x[8])
        if round(shape_id) != 2:
            return {
                "agent":          self.__class__.__name__,
                "clause":         self.CLAUSE,
                "pass":           True,
                "penalty":        0.0,
                "violation":      "",
                "recommendation": "Not applicable (shape_id ≠ 2, not a branch connection).",
            }

        D_run_mm = max(float(x[0]), 1.0)
        d_ratio  = float(np.clip(x[9], 0.0, 1.0))
        D_branch_mm = D_run_mm * d_ratio

        fitting, base_penalty = _required_connection(d_ratio)

        # Weld clearance check: estimated spacing from branch OD
        # For a single branch, the adjacent clearance is estimated as
        # (D_run - D_branch) / 2 — conservative
        weld_clearance_mm = max((D_run_mm - D_branch_mm) / 2.0, 0.0)
        clearance_ok = weld_clearance_mm >= _MIN_WELD_CLEARANCE_MM

        violations = []
        penalty    = 0.0

        # The main rule: d/D ≥ 0.75 is only acceptable with a specific fitting
        # Flag this as a warning (not a hard violation unless context indicates wrong fitting)
        # Hard violation: weld clearance insufficient
        if not clearance_ok:
            shortfall = _MIN_WELD_CLEARANCE_MM - weld_clearance_mm
            penalty  += float(np.clip(shortfall / _MIN_WELD_CLEARANCE_MM, 0.0, 1.0)) * 0.5
            violations.append(
                f"Estimated weld clearance {weld_clearance_mm:.1f} mm < "
                f"{_MIN_WELD_CLEARANCE_MM:.0f} mm minimum "
                f"(ASME B31.3 § 328.5.5).  d/D = {d_ratio:.2f}."
            )

        # High d/D ratio: enforce fitting type requirement
        if d_ratio >= _THRESHOLD_HIGH:
            # High-ratio branch is acceptable but requires specific fitting — flagged as advisory
            violations.append(
                f"Branch d/D = {d_ratio:.2f} ≥ {_THRESHOLD_HIGH}: "
                f"requires {fitting} per ASME B31.3 § 304.3.1."
            )
            penalty += base_penalty * 0.4   # advisory, not hard fail

        passes  = penalty == 0.0
        penalty = min(penalty, 1.0)

        recommendation = (
            f"Use a {fitting} at this branch connection "
            f"(d/D = {d_ratio:.2f}, D_branch ≈ {D_branch_mm:.1f} mm, "
            f"D_header = {D_run_mm:.1f} mm)."
        )
        if not clearance_ok:
            recommendation += (
                f"  Increase spacing between adjacent branch welds to ≥ {_MIN_WELD_CLEARANCE_MM:.0f} mm."
            )

        return {
            "agent":           self.__class__.__name__,
            "clause":          self.CLAUSE,
            "d_ratio":         round(d_ratio, 3),
            "D_run_mm":        round(D_run_mm, 1),
            "D_branch_mm":     round(D_branch_mm, 1),
            "required_fitting": fitting,
            "weld_clearance_mm": round(weld_clearance_mm, 1),
            "clearance_ok":    clearance_ok,
            "pass":            passes,
            "penalty":         round(penalty, 4),
            "violation":       "  ".join(violations),
            "recommendation":  recommendation,
        }

    @staticmethod
    def _missing_data(reason: str) -> Dict[str, Any]:
        return {
            "agent":   "BranchRulesAgent",
            "clause":  "ASME B31.3 § 304.3.1",
            "pass":    False,
            "penalty": 0.5,
            "violation": reason,
            "recommendation": "Provide complete design vector.",
        }
