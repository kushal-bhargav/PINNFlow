"""
pinnflow/benchmarks/asme_cases.py
───────────────────────────────────
Phase 6 — Validation Benchmarks

ASME B31.3 specific test cases designed to trigger the Codal Intelligence Engine.
These include known-good (compliant) and known-bad (violating) designs.
"""
from __future__ import annotations

import numpy as np


def get_asme_benchmark_suite() -> dict:
    """
    Returns a suite of 16-D design vectors for testing Codal Agents.
    Structure: { "case_name": (design_vector_16D, expected_violations) }
    """
    suite = {}

    # Case 1: Standard compliant straight pipe (NPS 10 Sch 40)
    # D=273.0, t=9.27, L=100.0, P=2.0, disp=0.0, dT=20.0, v=15.0, soil=0.5, shape=0
    x_compliant = np.array([273.0, 9.27, 100.0, 2.0, 0.0, 20.0, 15.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    suite["Straight_Compliant"] = (x_compliant, [])

    # Case 2: High pressure, thin wall -> Hoop stress violation
    # D=406.4 (NPS 16), t=6.35 (too thin), P=12.0 MPa -> sigma > 380 MPa
    x_thin = np.array([406.4, 6.35, 50.0, 12.0, 0.0, 10.0, 10.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    suite["Thin_Wall_Hoop"] = (x_thin, ["ASME B31.3 § 304.1.2"])

    # Case 3: High thermal delta -> Flexibility violation
    # dT=150.0 °C -> SE_proxy = 360 MPa > 205 MPa
    x_thermal = np.array([219.1, 8.18, 50.0, 2.0, 0.0, 150.0, 5.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    suite["High_Thermal_Strain"] = (x_thermal, ["ASME B31.3 Appendix C"])

    # Case 4: Non-standard pipe schedule
    # NPS 12 OD=323.9, but t=14.0 mm (Sch 40 is 10.31, Sch 80 is 17.48)
    x_non_std = np.array([323.9, 14.0, 10.0, 5.0, 0.0, 0.0, 5.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    suite["NonStandard_Schedule"] = (x_non_std, ["ASME B36.10M-2018"])

    # Case 5: Erosional velocity (API 14E)
    # v = 85.0 m/s > 60 m/s
    x_erosional = np.array([168.3, 7.11, 20.0, 5.0, 0.0, 10.0, 85.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    suite["High_Velocity_API14E"] = (x_erosional, ["API 14E"])

    # Case 6: Tight Elbow SIF violation
    # shape=1 (elbow), R/D = 0.5 -> tight bend -> high SIF
    x_tight_elbow = np.array([219.1, 8.18, 5.0, 8.0, 0.0, 0.0, 5.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    suite["Tight_Elbow_SIF"] = (x_tight_elbow, ["ASME B31.3 Appendix D"])

    # Case 7: Unreinforced Tee Branch
    # shape=2 (tee), d_ratio = 0.9, high pressure, standard wall (A_available < A_required)
    x_tee = np.array([508.0, 15.09, 10.0, 8.0, 0.0, 0.0, 5.0, 0.5, 2.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    suite["Unreinforced_Tee"] = (x_tee, ["ASME B31.3 § 304.3"])

    # Case 8: Abrupt Reducer Angle
    # shape=3 (reducer), L=1.0m, D1 to D2 ratio = 0.5 -> D1=546, D2=273, L=1.0m -> alpha > 30 deg
    x_abrupt_reducer = np.array([273.0, 9.27, 0.1, 5.0, 0.0, 0.0, 5.0, 0.5, 3.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    suite["Abrupt_Reducer"] = (x_abrupt_reducer, ["ASME B31.3 § 304.7.4"])

    return suite
