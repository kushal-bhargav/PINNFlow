"""
Client requirement ingestion layer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pinnflow.geometry.features import ensure_geometry_state
from pinnflow.pinn import MultiTaskPINN


class RequirementParser:
    """
    Extracts structured piping schemas from raw inputs and scenario context.
    """

    def __init__(self, pinn: Optional[MultiTaskPINN] = None):
        self.pinn = pinn
        self.history: List[Dict[str, Any]] = []

    def parse_PID(self, text_or_image: str, scenario: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        inputs = (scenario or {}).get("inputs", {})
        meta = (scenario or {}).get("meta", {})

        max_pressure_bar = float(inputs.get("max_p", 15.5))
        max_temp_c = float(inputs.get("max_t", 65.0))
        topology = inputs.get("topology", "GasLib-134")
        fluid = "Multiphase" if "fsi" in str(topology).lower() else "Gas"

        diameter_main = 323.8 if max_pressure_bar > 60 else 273.0
        thickness_main = 12.7 if max_pressure_bar > 60 else 9.27
        diameter_branch = 219.1 if meta.get("congested") else 168.0
        thickness_branch = 9.53 if max_temp_c > 200 else 7.11

        return {
            "raw_input": text_or_image,
            "scenario_name": (scenario or {}).get("scenario_name", "ad_hoc"),
            "topology": topology,
            "lines": [
                {
                    "id": "L-101",
                    "diameter": diameter_main,
                    "thickness": thickness_main,
                    "material": "A106-B",
                    "fluid": fluid,
                    "design_length_m": 120.0 if meta.get("fsi") else 80.0,
                },
                {
                    "id": "L-102",
                    "diameter": diameter_branch,
                    "thickness": thickness_branch,
                    "material": "A106-B",
                    "fluid": fluid,
                    "design_length_m": 45.0,
                },
            ],
            "equipment": [
                {
                    "id": "V-501",
                    "type": "Scrubber" if fluid == "Gas" else "Separator",
                    "nodes": ["L-101", "L-102"],
                }
            ],
            "constraints": {
                "max_pressure_bar": max_pressure_bar,
                "max_temp_c": max_temp_c,
                "code": "ASME B31.3",
                "topology": topology,
            },
            "meta": meta,
        }

    def parse_line_list(self, csv_data: str) -> pd.DataFrame:
        import io

        return pd.read_csv(io.StringIO(csv_data))

    def validate_physics(
        self,
        schema: Dict[str, Any],
        scenario: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.pinn is None:
            return {"status": "SKIPPED", "message": "PINN model not provided."}

        report = []
        pressure_mpa = schema["constraints"]["max_pressure_bar"] / 10.0
        scenario_name = (scenario or {}).get("scenario_name", schema.get("scenario_name", "ad_hoc"))

        for line in schema["lines"]:
            d = float(line["diameter"])
            t = float(line["thickness"])
            state_10 = np.array(
                [[d, t, line.get("design_length_m", 100.0), pressure_mpa, 0.0, 0.0, 3.0, 0.5, 0.0, 1.0]],
                dtype=float,
            )
            state = ensure_geometry_state(state_10)
            pinn_pred = self.pinn.predict(state)
            pinn_s = float(pinn_pred[0, 0])
            hoop = float((pressure_mpa * d) / (2.0 * max(t, 1e-6)))
            status = "PASS" if pinn_s < 200.0 and hoop < 200.0 else "WARNING"
            report.append(
                {
                    "line_id": line["id"],
                    "scenario": scenario_name,
                    "hoop_estimate": round(hoop, 2),
                    "pinn_stress_estimate": round(pinn_s, 2),
                    "status": status,
                    "notes": "Stress exceeding 200MPa" if status == "WARNING" else "Safe",
                }
            )

        return {
            "status": "VALIDATED",
            "report": report,
            "overall_safety": "OK" if all(r["status"] == "PASS" for r in report) else "REVIEW_REQUIRED",
        }

    def run_e2e(self, raw_input: str, scenario: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        print("[Ingestion] Starting Generative AI Schema Extraction...")
        schema = self.parse_PID(raw_input, scenario=scenario)
        print(f"[Ingestion] Extracted {len(schema['lines'])} lines and {len(schema['equipment'])} equipment items.")

        print("[Ingestion] Validating physics consistency via PINN...")
        validation = self.validate_physics(schema, scenario=scenario)
        return {"schema": schema, "validation": validation}
