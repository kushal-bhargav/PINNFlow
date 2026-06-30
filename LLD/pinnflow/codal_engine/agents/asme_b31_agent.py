"""
pinnflow/codal_engine/agents/asme_b31_agent.py
─────────────────────────────────────────────
MODULE 26 — ASME B31.3 Compliance Agent

Specialized in piping stress, pressure limits, and wall thickness.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any
from pinnflow.codal_engine.agents.base import CritiqueAgent

class ASMEB31Agent(CritiqueAgent):
    """
    [V8] ASME B31.3 Audit Agent.
    """
    def evaluate(self, design: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query") or "ASME B31.3 stress limit thickness pressure"
        rule_context = {
            "query": query,
            "pressure": context.get("pressure", float(design[3])),
            "temperature": context.get("delta_T", float(design[5])),
        }
        rule = self.rule_store.best_rule(rule_context, rule_type="stress_limit", code="ASME B31.3")
        
        # design vector: [d, t, L, P, ...]
        d, t, p = float(design[0]), float(design[1]), float(design[3])
        current_sigma = float(context.get("sigma", (p * d) / (2 * np.maximum(t, 1e-3))))
        
        # Hoop stress calculation
        sigma = current_sigma
        limit = float(rule["limit"]) if rule else 200.0
        target_sigma = min(limit * 0.95, sigma)

        violation = max(0, sigma - limit)
        penalty = (violation / max(limit, 1e-6)) ** 2 * 10.0
        recommended_thickness = max(
            t,
            (p * d) / (2.0 * max(target_sigma, 1e-6)),
        )
        thickness_delta = max(0.0, recommended_thickness - t)
        pressure_delta = max(0.0, p - (2.0 * t * target_sigma / max(d, 1e-6)))
        recommendations = [
            {
                "parameter": "thickness",
                "index": 1,
                "current": t,
                "target": recommended_thickness,
                "delta": thickness_delta,
                "unit": "mm",
                "direction": "increase" if thickness_delta > 0 else "hold",
                "rationale": "Increase wall thickness to recover ASME stress margin.",
            },
            {
                "parameter": "pressure",
                "index": 3,
                "current": p,
                "target": max(1.0, p - pressure_delta),
                "delta": pressure_delta,
                "unit": "MPa",
                "direction": "decrease" if pressure_delta > 0 else "hold",
                "rationale": "Reduce operating pressure if thickness is constrained.",
            },
        ]

        status_str = "PASS" if violation == 0 else "FAIL"
        if context.get("verbose", False):
            print(f"[ASMEB31Agent] Evaluating: D={d:.1f}mm, t={t:.2f}mm, P={p:.2f}MPa | stress={sigma:.1f} MPa (Limit: {limit:.1f} MPa) | status={status_str}")
            if violation > 0:
                print(f"[ASMEB31Agent] Violation: Stress {sigma:.1f} MPa exceeds limit {limit:.1f} MPa! Penalty={penalty:.4f}")
                print(f"[ASMEB31Agent] Recommendation: Increase thickness to target={recommended_thickness:.2f} mm")

        return {
            "agent": "ASME B31.3",
            "penalty": float(penalty),
            "status": status_str,
            "pass": violation == 0,
            "explanation": f"Stress {sigma:.1f} MPa vs limit {limit:.1f} MPa.",
            "rule": rule,
            "rule_value": {"limit": limit, "measured_sigma": sigma},
            "recommendations": recommendations,
            "design_feedback": {
                "stress_margin": limit - sigma,
                "priority": "thickness" if thickness_delta > pressure_delta else "pressure",
            },
        }
