"""
pinnflow/benchmarks/asme_adapter.py
────────────────────────────────────
MODULE 30 — ASME Benchmark Case Adapter

Incorporate 5 specific engineering benchmark cases for ground-truth verification.
Reference: ASME B31.3 Appendix S Example Cases.
"""
from __future__ import annotations
import numpy as np

class ASMEBenchmarkAdapter:
    """
    [P8.2] Gold-Standard Verification Suite.
    Provides 5 canonical cases with analytical Ground Truth (GT).
    """
    def __init__(self):
        self.cases = {
            "case1_straight_hp": {
                "params": [323.9, 12.7, 100.0, 15.0, 0.0, 20.0, 0.0, 0.5, 0, 1.0],
                "gt_stress": 191.3, # Calculated via Barlow's
                "desc": "NPS 12 Sch 80 Straight Pipe, High Pressure"
            },
            "case2_elbow_thermal": {
                "params": [219.1, 8.18, 50.0, 5.0, 0.0, 150.0, 0.0, 0.5, 1, 1.5],
                "gt_stress": 145.2, # Calculated via SIF (i=2.1)
                "desc": "NPS 8 Sch 40 90-Deg Elbow, Thermal Expansion"
            },
            "case3_tee_branch": {
                "params": [406.4, 9.53, 30.0, 8.0, 0.0, 50.0, 0.0, 0.5, 2, 1.0],
                "gt_stress": 172.5, # Calculated via SIF (i=1.5)
                "desc": "NPS 16 Sch STD Tee, Pressure + Branch Loading"
            },
            "case4_soil_subsidence": {
                "params": [610.0, 15.0, 200.0, 12.0, 100.0, 10.0, 0.0, 0.8, 0, 1.0],
                "gt_stress": 210.4,
                "desc": "Large Diameter Burial, Soil Displacement Effect"
            },
            "case5_velocity_limit": {
                "params": [168.3, 7.11, 20.0, 2.0, 0.0, 25.0, 8.5, 0.5, 0, 1.0],
                "gt_stress": 23.7,
                "gt_dp": 42.1,
                "desc": "NPS 6 Sch 40, High Velocity / Erosion Study"
            }
        }

    def get_case(self, name: str) -> dict:
        return self.cases.get(name)

    def run_benchmark(self, model) -> dict:
        """Evaluates model against all 5 ASME cases."""
        results = {}
        for name, data in self.cases.items():
            X = np.array(data["params"]).reshape(1, -1)
            pred = model.predict(X)[0]
            sigma_err = abs(pred[0] - data["gt_stress"]) / data["gt_stress"]
            results[name] = {
                "pred_stress": float(pred[0]),
                "gt_stress": data["gt_stress"],
                "mae_percent": float(sigma_err * 100)
            }
        return results
