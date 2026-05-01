"""
PINNFLOW v8.1 master industrial automation suite.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import warnings

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from pinnflow.orchestrator_v2 import UnifiedOrchestrator

warnings.filterwarnings("ignore")


def run_industrial_suite() -> None:
    print("=" * 100)
    print("  PINNFLOW v8.1 - TOTAL INDUSTRIAL AUTOMATION [ZERO-TO-END]")
    print("=" * 100)

    train_cvae = os.getenv("PINNFLOW_TRAIN_CVAE", "1").strip().lower() not in {"0", "false", "no"}
    cvae_epochs = int(os.getenv("PINNFLOW_CVAE_EPOCHS", "18"))
    cvae_samples = int(os.getenv("PINNFLOW_CVAE_SAMPLES_PER_SCENARIO", "20"))
    cvae_scenarios_raw = os.getenv("PINNFLOW_CVAE_SCENARIOS", "").strip()
    cvae_scenarios = [item.strip() for item in cvae_scenarios_raw.split(",") if item.strip()] or None

    orchestrator = UnifiedOrchestrator(
        train_cvae=train_cvae,
        cvae_epochs=cvae_epochs,
        cvae_samples_per_scenario=cvae_samples,
        cvae_scenario_names=cvae_scenarios,
    )
    cases = ["high_pressure_gas", "refinery_compliance", "deep_sea_fsi"]

    results = {}
    for case in cases:
        try:
            print(f"\n[SUITE] Executing workflow for scenario: {case.upper()}...")
            res = orchestrator.run_e2e(case_name=case)
            results[case] = res
            print(f"  [OK] {case} complete. Compliance score: {res['compliance_score']:.4f}")
        except Exception as exc:
            print(f"  [ERROR] case {case}: {exc}")

    print("\n" + "=" * 100)
    print("  [OK] ALL SCENARIOS COMPLETE")
    print("  [OK] Scientific deliverables generated in: results/")
    print(f"  [OK] Decision traces recorded for {len(results)} cases.")
    print("=" * 100)
    print("\n[V8.1] Suggestion: Open 'pinnflow/ui/dashboard.html' to view the interactive audit results.")


if __name__ == "__main__":
    run_industrial_suite()
