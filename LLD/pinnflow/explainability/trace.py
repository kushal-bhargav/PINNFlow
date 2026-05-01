"""
Explainability and traceability engine.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


class ExplainEngine:
    """
    Provides decision trace capture and metrics-driven summaries.
    """

    def __init__(self):
        self.trace_log: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.trace_log = []

    def record_decision(
        self,
        step: int,
        action: Any,
        result: Any,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.trace_log.append(
            {
                "step": step,
                "action": str(action),
                "result": str(result),
                "reason": reason,
                "metadata": metadata or {},
            }
        )

    def generate_explanation(
        self,
        design_id: str,
        final_state: Any,
        metrics: Dict[str, Any],
        scenario: Optional[Dict[str, Any]] = None,
        convergence_history: Optional[List[Dict[str, Any]]] = None,
        codal_report: Optional[List[Dict[str, Any]]] = None,
        trace_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        compliance_score = float(metrics.get("compliance_score", 1.0))
        sigma = float(metrics.get("sigma", 0.0))
        delta_p = float(metrics.get("delta_P", 0.0))
        iterations = len(convergence_history or [])
        topology = (scenario or {}).get("inputs", {}).get("topology", "unknown")
        status = "PASS" if metrics.get("constraint_ok", False) and compliance_score >= 0.90 else "REVIEW"
        confidence_value = max(0.10, min(0.99, 0.55 + 0.4 * compliance_score))

        return {
            "design_id": design_id,
            "summary": (
                f"Design {design_id} was optimized over {max(iterations, 1)} refinement iteration(s) "
                f"for topology {topology}."
            ),
            "physics_basis": (
                f"Predicted stress is {sigma:.2f} MPa and pressure-drop proxy is {delta_p:.2f} kPa; "
                f"constraint status is {status}."
            ),
            "rl_strategy": f"Closed-loop convergence reason: {metrics.get('convergence_reason', 'unknown')}.",
            "code_compliance": status,
            "industrial_confidence": f"MODERATE-HIGH ({confidence_value:.2f})",
            "compliance_score": compliance_score,
            "codal_report": codal_report or [],
            "trace_path": trace_path,
        }

    def export_report(self) -> str:
        return json.dumps(self.trace_log, indent=2)

    def export_report_to_file(self, design_id: str, output_dir: str = "results/trace") -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"TRACE_{design_id}.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.trace_log, file, indent=2)
        return path
