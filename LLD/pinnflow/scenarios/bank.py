"""
pinnflow/scenarios/bank.py
─────────────────────────
MODULE 17 — Multi-Scenario Generator (Synthetic Data Engine)

Defines industrial templates and failure cases for stress testing the pipeline.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Any

class ScenarioBank:
    """
    [V8] Industrial Scenario Generator.
    Creates structured test cases for verification.
    """
    TEMPLATES = {
        "refinery_congested": {
            "pressure": [10.0, 50.0],
            "temp": [20.0, 400.0],
            "complexity": "HIGH",
            "congested": True,
            "topology": "GasLib-134",
        },
        "high_pressure_gas": {
            "pressure": [70.0, 100.0],
            "temp": [0.0, 50.0],
            "complexity": "MEDIUM",
            "congested": False,
            "topology": "GasLib-582",
        },
        "deep_sea_fsi": {
            "pressure": [20.0, 40.0],
            "temp": [4.0, 20.0],
            "complexity": "EXTREME",
            "fsi": True,
            "topology": "Synthetic-FSI",
        },
        "refinery_compliance": {
            "pressure": [30.0, 60.0],
            "temp": [150.0, 450.0],
            "complexity": "HIGH",
            "congested": True,
            "codal_penalty_weight": 2.5,
            "topology": "GasLib-4197",
        },
        "extreme_conditions": {
            "pressure": [80.0, 150.0],
            "temp": [-50.0, 100.0],
            "complexity": "EXTREME",
            "fail_recovery": True,
            "topology": "GasLib-582",
        }
    }

    def generate_scenario(self, name: str) -> Dict[str, Any]:
        """Generates a specific instance of a scenario based on templates."""
        if name not in self.TEMPLATES:
            raise ValueError(f"Scenario {name} not found in bank.")
            
        tmpl = self.TEMPLATES[name]
        p = np.random.uniform(tmpl["pressure"][0], tmpl["pressure"][1])
        t = np.random.uniform(tmpl["temp"][0], tmpl["temp"][1])
        
        return {
            "scenario_name": name,
            "inputs": {
                "max_p": round(p, 2),
                "max_t": round(t, 2),
                "topology": tmpl.get("topology", "GasLib-134"),
            },
            "meta": tmpl
        }

    def get_diverse_suite(self) -> List[Dict[str, Any]]:
        """Returns a set of cases covering standard and failure modes."""
        return [self.generate_scenario(n) for n in self.TEMPLATES]
