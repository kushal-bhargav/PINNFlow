"""
Deliverables generation engine.
"""
from __future__ import annotations

import json
import os
from math import cos, sin, pi
from typing import Any, Dict

import pandas as pd


class DeliverableGenerator:
    """
    Converts optimized designs into scenario-specific deliverables.
    """

    def __init__(self, output_dir: str = "results/deliverables"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _case_dir(self, design_id: str) -> str:
        case_dir = os.path.join(self.output_dir, design_id)
        os.makedirs(case_dir, exist_ok=True)
        return case_dir

    def generate_bom(self, design_id: str, design_state: Any) -> str:
        d, _, L = design_state[0], design_state[1], design_state[2]
        bom = pd.DataFrame(
            [
                {"Item": "Main Pipe Segment", "Spec": f"NPS {d/25.4:.1f} Sch Std", "Length_m": round(L, 2), "Qty": 1},
                {"Item": "Elbow 90 LR", "Spec": f"NPS {d/25.4:.1f} BW", "Qty": 2},
                {"Item": "Flange WN", "Spec": "Class 150 RF", "Qty": 2},
            ]
        )
        path = os.path.join(self._case_dir(design_id), f"BOM_{design_id}.csv")
        bom.to_csv(path, index=False)
        return path

    def generate_iso_mock(self, design_id: str) -> str:
        """
        Backward-compatible shim for older callers.

        The real deterministic router now lives in generate_iso().
        """
        return self.generate_iso(design_id, [273.0, 9.27, 120.0, 5.0, 15.0, 20.0, 2.5, 0.55, 0.0, 0.9])

    def _router_direction(self, angle_deg: float) -> list[float]:
        angle = angle_deg * pi / 180.0
        return [round(cos(angle), 6), round(sin(angle), 6), 0.0]

    def generate_iso(self, design_id: str, design_state: Any) -> str:
        """
        Deterministic piping router for the 10-D state vector.

        The geometry is intentionally simple but real:
        - Straight: one run along +X
        - Elbow: run along +X then turn to +Y
        - Tee: main run along +X with a branch at midspan
        """
        d = float(design_state[0])
        L = max(float(design_state[2]), 1.0)
        shape_id = int(round(float(design_state[8])))
        shape_param = float(design_state[9])

        half = L * 0.5
        main_dir = self._router_direction(0.0)
        turn_dir = self._router_direction(90.0)
        branch_dir = self._router_direction(90.0)
        arc_scale = max(d * max(shape_param, 0.3), 0.1)

        nodes = {"n0": [0.0, 0.0, 0.0]}
        segments = []

        if shape_id == 1:
            bend_len = min(max(arc_scale * 1.5, 1.0), half)
            nodes["n1"] = [round(half - bend_len, 3), 0.0, 0.0]
            nodes["n2"] = [round(nodes["n1"][0], 3), round(bend_len, 3), 0.0]
            nodes["n3"] = [round(L, 3), round(bend_len, 3), 0.0]
            segments = [
                {"start": nodes["n0"], "end": nodes["n1"], "dir": main_dir, "type": "straight"},
                {"start": nodes["n1"], "end": nodes["n2"], "dir": turn_dir, "type": "elbow"},
                {"start": nodes["n2"], "end": nodes["n3"], "dir": main_dir, "type": "straight"},
            ]
        elif shape_id == 2:
            branch_len = min(max(arc_scale, 1.0), half * 0.75)
            nodes["n1"] = [round(half, 3), 0.0, 0.0]
            nodes["n2"] = [round(L, 3), 0.0, 0.0]
            nodes["n3"] = [round(half, 3), round(branch_len, 3), 0.0]
            segments = [
                {"start": nodes["n0"], "end": nodes["n1"], "dir": main_dir, "type": "straight"},
                {"start": nodes["n1"], "end": nodes["n2"], "dir": main_dir, "type": "straight"},
                {"start": nodes["n1"], "end": nodes["n3"], "dir": branch_dir, "type": "branch"},
            ]
        else:
            nodes["n1"] = [round(L, 3), 0.0, 0.0]
            segments = [
                {"start": nodes["n0"], "end": nodes["n1"], "dir": main_dir, "type": "straight"},
            ]

        iso_data = {
            "project": "PINNFlow v8 E2E",
            "line_id": design_id,
            "shape_id": shape_id,
            "shape_param": round(shape_param, 3),
            "nominal_diameter_mm": round(d, 3),
            "nodes": nodes,
            "segments": segments,
        }
        path = os.path.join(self._case_dir(design_id), f"ISO_{design_id}.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(iso_data, file, indent=2)
        return path

    def generate_compliance_matrix(self, design_id: str, metrics: Dict[str, Any]) -> str:
        compliance_score = metrics.get("compliance_score", 1.0)
        
        # Extract dynamic values from codal report
        codal_report = metrics.get("codal_report", [])
        
        asme_limit_str = "200.0 MPa"
        asme_status = "PASS" if metrics.get("constraint_ok", False) else "FAIL"
        
        api_limit_str = "60.0 m/s"
        api_status = "REVIEW"
        api_val_str = f"{metrics.get('velocity', 0):.1f} m/s"
        
        for rep in codal_report:
            if rep.get("agent") == "ASME B31.3":
                limit = rep.get("rule_value", {}).get("limit", 200.0)
                asme_limit_str = f"{limit:.1f} MPa"
                asme_status = rep.get("status", asme_status)
            elif rep.get("agent") == "API 14E":
                limit = rep.get("rule_value", {}).get("limit", 60.0)
                api_limit_str = f"{limit:.1f} m/s"
                api_status = rep.get("status", "REVIEW")
                api_val_str = f"{rep.get('rule_value', {}).get('measured_velocity', 0):.1f} m/s"

        matrix = {
            "Requirement": ["ASME B31.3 Stress", "API 14E Erosional Velocity", "Internal Pressure", "Compliance Score"],
            "Limit": [asme_limit_str, api_limit_str, "Design envelope", ">= 0.90"],
            "Value": [
                f"{metrics.get('sigma', 0):.1f} MPa",
                api_val_str,
                f"{metrics.get('delta_P', 0):.2f} kPa",
                f"{compliance_score:.3f}",
            ],
            "Status": [
                asme_status,
                api_status,
                "PASS" if metrics.get("delta_P", 0) < 100 else "REVIEW",
                "PASS" if compliance_score >= 0.90 else "REVIEW",
            ],
        }
        df = pd.DataFrame(matrix)
        path = os.path.join(self._case_dir(design_id), f"Compliance_Matrix_{design_id}.csv")
        df.to_csv(path, index=False)
        return path

    def generate_all(self, design_id: str, state: Any, metrics: Dict[str, Any]) -> Dict[str, str]:
        print(f"[Deliverables] Generating full package for {design_id}...")
        return {
            "bom": self.generate_bom(design_id, state),
            "iso": self.generate_iso(design_id, state),
            "matrix": self.generate_compliance_matrix(design_id, metrics),
        }
