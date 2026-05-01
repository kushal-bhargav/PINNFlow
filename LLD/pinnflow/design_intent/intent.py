"""
Design intent engine.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from pinnflow.agent import LagrangianPPOAgent
from pinnflow.pinn import MultiTaskPINN


class IntentEngine:
    """
    Applies engineering heuristics and candidate selection logic.
    """

    def __init__(self, pinn: MultiTaskPINN, agent: Optional[LagrangianPPOAgent] = None):
        self.pinn = pinn
        self.agent = agent
        self.rules = {
            "min_spacing_mm": 500,
            "max_bend_angle": 90,
            "accessibility_zone_m": 1.2,
        }

    def _topology_summary(self, graph_data: Any) -> Dict[str, float]:
        node_pressures = graph_data.get("node_pressures")
        flows = graph_data.get("flows")
        return {
            "mean_pressure": float(node_pressures.mean().item()),
            "max_pressure": float(node_pressures.max().item()),
            "mean_flow": float(flows.mean().item()),
            "max_flow": float(flows.max().item()),
        }

    def propose_variants(self, graph_data: Any) -> List[Dict[str, Any]]:
        print("[Intent] Proposing design variants based on maintenance heuristics...")
        summary = self._topology_summary(graph_data)
        variants = [
            {
                "name": "Shortest Path",
                "heuristic": "Cost",
                "score": round(0.70 + min(summary["mean_flow"], 1.0) * 0.10, 3),
            },
            {
                "name": "Max Accessibility",
                "heuristic": "Maintenance",
                "score": round(0.75 + min(summary["max_pressure"], 1.0) * 0.08, 3),
            },
            {
                "name": "Low Vibration",
                "heuristic": "Safety",
                "score": round(0.72 + min(summary["mean_pressure"], 1.0) * 0.09, 3),
            },
        ]
        return sorted(variants, key=lambda item: item["score"], reverse=True)

    def apply_heuristics(self, layout: np.ndarray) -> np.ndarray:
        refined = np.asarray(layout, dtype=float).copy()
        nps_standards = np.array([114.3, 168.3, 219.1, 273.0, 323.8, 355.6, 406.4])
        refined[0] = nps_standards[(np.abs(nps_standards - refined[0])).argmin()]
        refined[8] = float(np.clip(np.rint(refined[8]), 0, 2))
        refined[9] = float(np.clip(refined[9], 0.3, 1.5))
        return refined

    def validate_intent(self, layout: np.ndarray) -> bool:
        length = float(layout[2])
        delta_t = float(layout[5])
        pressure = float(layout[3])
        if abs(delta_t) > 50 and length < 10:
            return False
        if pressure > 15 and layout[1] < 8:
            return False
        return True

    def select_best_candidate(
        self,
        candidates: np.ndarray,
        graph_data: Any,
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        topology = self._topology_summary(graph_data)
        target_pressure = float(scenario["inputs"]["max_p"]) / 10.0
        scored = []

        for idx, candidate in enumerate(candidates):
            refined = self.apply_heuristics(candidate)
            if not self.validate_intent(refined):
                continue

            sigma, delta_p = self.pinn.predict(refined.reshape(1, -1))[0]
            pressure_gap = abs(refined[3] - target_pressure)
            topology_penalty = max(topology["mean_flow"], 0.0) * 0.05
            score = (
                1.0
                - min(float(sigma) / 250.0, 1.0) * 0.40
                - min(float(delta_p) / 150.0, 1.0) * 0.25
                - min(pressure_gap / 10.0, 1.0) * 0.20
                - topology_penalty
            )
            scored.append(
                {
                    "index": idx,
                    "candidate": refined,
                    "score": score,
                    "predicted_sigma": float(sigma),
                    "predicted_delta_p": float(delta_p),
                }
            )

        if not scored:
            fallback = self.apply_heuristics(np.asarray(candidates[0], dtype=float))
            return {"candidate": fallback, "score": 0.0, "ranking": []}

        scored.sort(key=lambda item: item["score"], reverse=True)
        return {"candidate": scored[0]["candidate"], "score": scored[0]["score"], "ranking": scored}
