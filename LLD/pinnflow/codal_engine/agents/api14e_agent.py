"""
API 14E compliance agent.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pinnflow.codal_engine.agents.base import CritiqueAgent


class API14EAgent(CritiqueAgent):
    """
    Audits erosional velocity limits for gas/liquid fluids.
    """

    VELOCITY_INDEX = 6
    DEFAULT_LIMIT = 60.0

    def evaluate(self, design: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query") or "API 14E erosional velocity flow"
        rule_context = {
            "query": query,
            "velocity": context.get("velocity", float(design[self.VELOCITY_INDEX])),
            "flow": context.get("flow", float(design[self.VELOCITY_INDEX])),
        }
        rule = self.rule_store.best_rule(rule_context, rule_type="erosional_velocity", code="API 14E")

        velocity = float(context.get("velocity", design[self.VELOCITY_INDEX]))
        limit = float(rule["limit"]) if rule else self.DEFAULT_LIMIT

        violation = max(0.0, velocity - limit)
        penalty = (violation / max(limit, 1e-6)) ** 2 * 10.0

        recommended_velocity = min(velocity, limit * 0.95)
        if velocity > 0:
            diameter_scale = float(np.sqrt(max(velocity, 1e-6) / max(recommended_velocity, 1e-6)))
        else:
            diameter_scale = 1.0
        current_diameter = float(design[0])
        diameter_target = current_diameter * diameter_scale
        diameter_delta = max(0.0, diameter_target - current_diameter)
        shape_id = int(np.clip(np.rint(float(design[8])), 0, 2))
        shape_param = float(design[9])
        shape_priority = "geometry" if shape_id in (1, 2) else "velocity"

        return {
            "agent": "API 14E",
            "penalty": float(penalty),
            "status": "PASS" if violation == 0 else "FAIL",
            "explanation": f"Fluid velocity {velocity:.1f} m/s vs limit {limit:.1f} m/s.",
            "rule": rule,
            "rule_value": {"limit": limit, "measured_velocity": velocity},
            "recommendations": [
                {
                    "parameter": "diameter",
                    "index": 0,
                    "current": current_diameter,
                    "target": diameter_target,
                    "delta": diameter_delta,
                    "unit": "mm",
                    "direction": "increase" if diameter_delta > 0 else "hold",
                    "rationale": "Increase pipe diameter to lower flow velocity under API 14E.",
                },
                {
                    "parameter": "shape_param",
                    "index": 9,
                    "current": shape_param,
                    "target": max(0.3, shape_param * 0.95),
                    "delta": max(0.0, shape_param - max(0.3, shape_param * 0.95)),
                    "unit": "-",
                    "direction": "decrease" if shape_id in (1, 2) and shape_param > 0.3 else "hold",
                    "rationale": "Reduce junction severity or bend intensity where topology allows.",
                },
            ],
            "design_feedback": {
                "priority": shape_priority,
                "velocity_margin": limit - velocity,
            },
        }
