"""
pinnflow/compliance.py
──────────────────────
MODULE 17 — Explainable ASME Compliance Engine [V7]

Rule-based expert system that checks stresses against ASME B31.3 standards.
"""
from __future__ import annotations
import numpy as np

class ASMECompliance:
    """
    [V7] Explainable Compliance Checker.
    Maps quantitative outputs to qualitative failure modes.
    """
    STRESS_LIMIT = 200.0  # MPa for basic steel grade
    FATIGUE_LIMIT = 0.8   # Damage index threshold

    def check(self, results: dict) -> dict:
        """
        [V7] Checks stress, flow, and fatigue for violations.
        Returns a dict of boolean flags and textual explanations.
        """
        violations = []
        is_safe = True

        sigma = results.get("sigma", 0.0)
        fatigue = results.get("fatigue_damage", 0.0)
        
        # 1. Primary Stress Check (Hoop/Von Mises)
        if sigma > self.STRESS_LIMIT:
            is_safe = False
            violations.append({
                "mode": "Primary Stress Failure",
                "severity": "CRITICAL",
                "explanation": f"Von Mises stress ({sigma:.1f} MPa) exceeds ASME B31.3 allowable limit ({self.STRESS_LIMIT} MPa)."
            })

        # 2. Fatigue Check
        if fatigue > self.FATIGUE_LIMIT:
            is_safe = False
            violations.append({
                "mode": "Cyclic Fatigue Risk",
                "severity": "HIGH",
                "explanation": f"Cumulative damage index ({fatigue:.3f}) exceeds safety threshold ({self.FATIGUE_LIMIT})."
            })

        return {
            "is_compliant": is_safe,
            "violation_count": len(violations),
            "details": violations
        }

    def explain(self, results: dict) -> str:
        """Returns a human-readable compliance summary."""
        report = self.check(results)
        if report["is_compliant"]:
            return "✅ DESIGN COMPLIANT: All ASME B31.3 criteria satisfied."
        
        summary = f"❌ COMPLIANCE FAILURE: {report['violation_count']} safety violations detected.\n"
        for v in report["details"]:
            summary += f"- [{v['severity']}] {v['mode']}: {v['explanation']}\n"
        return summary
