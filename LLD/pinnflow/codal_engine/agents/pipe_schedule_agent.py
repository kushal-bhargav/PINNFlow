"""
pinnflow/codal_engine/agents/pipe_schedule_agent.py
─────────────────────────────────────────────────────
Phase 4-D — ASME B36.10M Pipe Schedule Validation

Validates that the design's diameter + wall thickness combination
corresponds to a recognised ASME B36.10M pipe schedule.

Logic:
    1. Find the nearest NPS (Nominal Pipe Size) to the design diameter.
    2. Find the nearest standard schedule for that NPS.
    3. Flag if the deviation in thickness exceeds 5% of the standard value.

References:
    ASME B36.10M-2018 — Welded and Seamless Wrought Steel Pipe
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from pinnflow.codal_engine.agents.base import CritiqueAgent

# ── ASME B36.10M Schedule Table ────────────────────────────────────────────
# Format: (NPS_OD_mm, schedule_name, wall_thickness_mm)
_B36_TABLE: List[Tuple[float, str, float]] = [
    # NPS 4
    (114.3, "Sch 40", 6.02), (114.3, "Sch 80", 8.56), (114.3, "Sch 160", 13.49),
    # NPS 6
    (168.3, "Sch 40", 7.11), (168.3, "Sch 80", 10.97), (168.3, "Sch 160", 18.26),
    # NPS 8
    (219.1, "Sch 40", 8.18), (219.1, "Sch 80", 12.70), (219.1, "Sch 160", 23.01),
    # NPS 10
    (273.0, "Sch 40", 9.27), (273.0, "Sch 80", 12.70), (273.0, "Sch 160", 28.58),
    # NPS 12
    (323.9, "Sch 40", 10.31), (323.9, "Sch 80", 17.48), (323.9, "Sch 160", 33.32),
    # NPS 14
    (355.6, "Sch 40", 11.13), (355.6, "Sch 80", 19.05),
    # NPS 16
    (406.4, "Sch 40", 12.70), (406.4, "Sch 80", 21.44),
    # NPS 18
    (457.2, "Sch 40", 14.27), (457.2, "Sch 80", 23.83),
    # NPS 20
    (508.0, "Sch 40", 15.09), (508.0, "Sch 80", 26.19),
    # NPS 24
    (609.6, "Sch 40", 17.48), (609.6, "Sch 80", 30.96),
    # NPS 26
    (660.4, "Sch 40", 18.26),
    # NPS 30
    (762.0, "Sch 40", 20.62),
    # NPS 36
    (914.4, "Sch 40", 22.23),
]

_DEVIATION_LIMIT_FRAC: float = 0.05   # 5% tolerance


def _find_nearest_schedule(
    diameter_mm: float,
    thickness_mm: float,
) -> Tuple[float, str, float, float]:
    """
    Find the nearest ASME B36.10M schedule entry.

    Returns:
        (nps_od_mm, schedule_name, standard_thickness_mm, deviation_fraction)
    """
    # Step 1: find all rows for the nearest NPS OD
    nps_ods = sorted(set(row[0] for row in _B36_TABLE))
    nearest_od = min(nps_ods, key=lambda od: abs(od - diameter_mm))

    # Step 2: among rows with that NPS, find nearest thickness
    candidates = [(sch, t) for od, sch, t in _B36_TABLE if od == nearest_od]
    best_sch, best_t = min(candidates, key=lambda ct: abs(ct[1] - thickness_mm))
    deviation = abs(thickness_mm - best_t) / max(best_t, 1e-3)

    return nearest_od, best_sch, best_t, deviation


class PipeScheduleAgent(CritiqueAgent):
    """
    Validates diameter + thickness against ASME B36.10M schedule table.

    Design vector mapping:
        col 0: diameter_mm
        col 1: thickness_mm
    """

    CLAUSE = "ASME B36.10M-2018 — Pipe Schedule Validation"

    def evaluate(self, design: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        x = np.asarray(design, dtype=float).ravel()

        if len(x) < 2:
            return self._missing_data("Design vector needs ≥ 2 elements.")

        diameter_mm  = max(float(x[0]), 1.0)
        thickness_mm = max(float(x[1]), 0.1)

        nearest_od, schedule, std_t, deviation = _find_nearest_schedule(
            diameter_mm, thickness_mm
        )

        passes = deviation <= _DEVIATION_LIMIT_FRAC
        penalty = 0.0
        violation = ""
        recommendation = ""

        if not passes:
            penalty = float(np.clip((deviation - _DEVIATION_LIMIT_FRAC) / (1.0 + _DEVIATION_LIMIT_FRAC), 0.0, 1.0))
            violation = (
                f"Thickness {thickness_mm:.2f} mm deviates {deviation*100:.1f}% "
                f"from ASME B36.10M {schedule} standard ({std_t:.2f} mm) "
                f"for NPS {nearest_od:.1f} mm OD."
            )
            recommendation = (
                f"Use standard ASME B36.10M {schedule} pipe for NPS {nearest_od:.1f} mm OD: "
                f"wall thickness = {std_t:.2f} mm."
            )

        return {
            "agent":           self.__class__.__name__,
            "clause":          self.CLAUSE,
            "input_D_mm":      round(diameter_mm, 1),
            "input_t_mm":      round(thickness_mm, 2),
            "nearest_NPS_mm":  round(nearest_od, 1),
            "schedule":        schedule,
            "standard_t_mm":   round(std_t, 2),
            "deviation_pct":   round(deviation * 100.0, 2),
            "pass":            passes,
            "penalty":         round(penalty, 4),
            "violation":       violation,
            "recommendation":  recommendation,
        }

    @staticmethod
    def _missing_data(reason: str) -> Dict[str, Any]:
        return {
            "agent":   "PipeScheduleAgent",
            "clause":  "ASME B36.10M-2018",
            "pass":    False,
            "penalty": 0.5,
            "violation": reason,
            "recommendation": "Provide diameter and thickness in design vector.",
        }
