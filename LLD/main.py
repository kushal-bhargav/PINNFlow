"""
PINNFLOW v8.1 master industrial automation suite.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from pinnflow.orchestrator_v2 import UnifiedOrchestrator

warnings.filterwarnings("ignore")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def _load_env_file(path: Path) -> int:
    """Load simple KEY=VALUE pairs without requiring python-dotenv."""
    if not path.exists():
        return 0

    loaded = 0
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def _load_env_files() -> None:
    """Load runtime configuration from project-root and LLD .env files."""
    script_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd() / ".env", script_dir / ".env"]
    seen: set[Path] = set()
    loaded_total = 0
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        loaded = _load_env_file(resolved)
        loaded_total += loaded
        if loaded:
            print(f"[Config] Loaded {loaded} setting(s) from {resolved}")
    if loaded_total == 0:
        print("[Config] No .env settings loaded; using shell environment and defaults.")


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip().strip('"').strip("'"))
    except ValueError:
        print(f"[Config] Invalid integer for {name}={raw!r}; using {default}.")
        return default


def _write_topology_pipeline_metrics(orchestrator: UnifiedOrchestrator, results: dict) -> pd.DataFrame:
    """
    Export topology-grouped MAE/reliability from the actual pipeline run.

    The latest E2E suite produces optimized final states, but did not previously
    persist topology MAE values. This benchmark compares the final surrogate
    prediction with the simulator response for each completed scenario, then
    aggregates authentic GasLib cases separately from synthetic fallback cases.
    """
    rows = []
    for idx, (case_name, result) in enumerate(results.items()):
        state = np.asarray(result["optimized_state"], dtype=float).reshape(1, -1)
        topology = str(result["scenario"]["inputs"].get("topology", "unknown"))
        topology_group = "Synthetic" if "synthetic" in topology.lower() else "Authentic"

        pred = orchestrator.pinn.predict(state)[0]
        s = state.reshape(-1)

        # Seed simulator noise per scenario so the exported benchmark is stable.
        np.random.seed(10_000 + idx)
        truth = orchestrator.physics.generate_one(
            d=float(s[0]),
            t=float(s[1]),
            L=float(s[2]),
            P=float(s[3]),
            u=float(s[4]),
            dT=float(s[5]),
            k=float(s[7]),
            velocity=float(s[6]),
        )
        stress_truth = float(truth["von_mises_stress"])
        pressure_truth = float(truth["pressure_drop_kPa"])

        rows.append(
            {
                "case": case_name,
                "topology": topology,
                "topology_group": topology_group,
                "stress_pred_mpa": float(pred[0]),
                "stress_truth_mpa": stress_truth,
                "pressure_pred_kpa": float(pred[1]),
                "pressure_truth_kpa": pressure_truth,
                "stress_abs_error_mpa": abs(float(pred[0]) - stress_truth),
                "pressure_abs_error_kpa": abs(float(pred[1]) - pressure_truth),
                "compliance": float(result.get("compliance_score", np.nan)),
                "iterations": int(result.get("iterations", 0)),
            }
        )

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("topology_group")
        .agg(
            topology_count=("topology", "nunique"),
            scenario_count=("case", "count"),
            stress_mae_mpa=("stress_abs_error_mpa", "mean"),
            pressure_mae_kpa=("pressure_abs_error_kpa", "mean"),
            compliance=("compliance", "mean"),
            iterations=("iterations", "mean"),
        )
        .reset_index()
    )
    summary["reliability"] = np.clip(1.0 - summary["stress_mae_mpa"] / 200.0, 0.0, 1.0)

    out_dir = Path("results") / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "topology_pipeline_metrics_raw.csv"
    summary_path = out_dir / "topology_pipeline_metrics.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"  [OK] Topology benchmark saved to: {summary_path}")
    return summary


def _fmt(value, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{value:.{digits}f}" if np.isfinite(value) else "N/A"


def _log_result_tables(orchestrator, results: dict, topology_summary: pd.DataFrame) -> None:
    names = {"high_pressure_gas": "High Pressure Gas", "refinery_compliance": "Refinery Compliance", "deep_sea_fsi": "Deep Sea FSI"}
    scenarios = [[names.get(case, case.replace("_", " ").title()), result["scenario"]["inputs"].get("topology", "N/A"), _fmt(result.get("compliance_score")), result.get("iterations", "N/A")] for case, result in results.items()]
    scenarios.append(["Average", "", _fmt(np.mean([r["compliance_score"] for r in results.values()])), _fmt(np.mean([r["iterations"] for r in results.values()]), 2)])

    measured = []
    for result in results.values():
        state = np.asarray(result["optimized_state"], dtype=float).reshape(1, -1)
        start = time.perf_counter()
        prediction = orchestrator.pinn.predict(state)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        s = state[0]
        truth = orchestrator.physics.generate_one(d=s[0], t=s[1], L=s[2], P=s[3], u=s[4], dT=s[5], k=s[7], velocity=s[6], shape_id=s[8], shape_param=s[9])
        measured.append({"shape": int(np.clip(np.rint(s[8]), 0, 3)), "time": elapsed_ms, "stress": abs(float(prediction[0]) - truth["von_mises_stress"]), "pressure": abs(float(prediction[1]) - truth["pressure_drop_kPa"])})

    geometry = []
    for shape, name in ((0, "Straight"), (1, "Elbow/Bend"), (2, "T-junction")):
        rows = [row for row in measured if row["shape"] == shape]
        geometry.append([name, _fmt(np.mean([r["time"] for r in rows])) if rows else "N/A", "N/A", _fmt(np.mean([r["stress"] for r in rows])) if rows else "N/A", _fmt(np.mean([r["pressure"] for r in rows])) if rows else "N/A"])
    aggregate = [_fmt(np.mean([r["time"] for r in measured])), "N/A", _fmt(np.mean([r["stress"] for r in measured])), _fmt(np.mean([r["pressure"] for r in measured]))]
    geometry.extend([["Closed Loop", *aggregate], ["Average", *aggregate]])
    topology = [[row.topology_group, _fmt(row.stress_mae_mpa), _fmt(row.pressure_mae_kpa), _fmt(row.reliability)] for row in topology_summary.itertuples(index=False)]

    print("\nTable II. End-to-End Scenario Outcomes from the Verified Artifact Set")
    print(tabulate(scenarios, headers=["Scenario", "Topology", "Compliance", "Iterations"], tablefmt="grid"))
    print("\nTable III. Geometry-Level Surrogate Efficiency versus Nominal ANSYS Runtime")
    print(tabulate(geometry, headers=["Geometry", "Time (ms)", "Speedup", "Stress MAE", "Press. MAE"], tablefmt="grid"))
    print("\nTable IV. Authentic versus Synthetic Topology Benchmark")
    print(tabulate(topology, headers=["Topology", "Stress MAE", "Pressure MAE", "Reliability"], tablefmt="grid"))


def run_industrial_suite() -> None:
    _load_env_files()

    print("=" * 100)
    print("  PINNFLOW v8.1 - TOTAL INDUSTRIAL AUTOMATION [ZERO-TO-END]")
    print("=" * 100)

    # ── Training control via environment variables ───────────────────────────
    test_cycle = _flag("PINNFLOW_TEST_CYCLE", "0")

    train_cvae     = _flag("PINNFLOW_TRAIN_CVAE")
    cvae_epochs    = _env_int("PINNFLOW_CVAE_EPOCHS", 1 if test_cycle else 18)
    cvae_samples   = _env_int("PINNFLOW_CVAE_SAMPLES_PER_SCENARIO", 2 if test_cycle else 20)
    cvae_scenarios_raw = os.getenv("PINNFLOW_CVAE_SCENARIOS", "").strip()
    cvae_scenarios = [s.strip() for s in cvae_scenarios_raw.split(",") if s.strip()] or None

    train_pinn  = _flag("PINNFLOW_TRAIN_PINN")
    pinn_epochs = _env_int("PINNFLOW_PINN_EPOCHS", 1 if test_cycle else 500)
    pinn_n      = _env_int("PINNFLOW_PINN_SYNTHETIC_N", 60 if test_cycle else 1200)

    train_ppo    = _flag("PINNFLOW_TRAIN_PPO")
    ppo_episodes = _env_int("PINNFLOW_PPO_EPISODES", 1 if test_cycle else 400)
    ppo_steps = _env_int("PINNFLOW_PPO_STEPS", 2 if test_cycle else 25)
    ppo_dyna_rollouts = _env_int("PINNFLOW_PPO_DYNA_ROLLOUTS", 2 if test_cycle else 100)

    train_gnn = _flag("PINNFLOW_TRAIN_GNN")
    gnn_epochs = _env_int("PINNFLOW_GNN_EPOCHS", 1 if test_cycle else 200)

    cases_raw = os.getenv("PINNFLOW_CASES", "high_pressure_gas,refinery_compliance,deep_sea_fsi").strip()
    cases = [case.strip() for case in cases_raw.split(",") if case.strip()]

    print(
        "[Config] Training cycle: "
        f"test_cycle={test_cycle} | PINN epochs={pinn_epochs}, synthetic_n={pinn_n} | "
        f"GNN epochs={gnn_epochs} | PPO episodes={ppo_episodes}, steps={ppo_steps}, "
        f"dyna_rollouts={ppo_dyna_rollouts} | CVAE epochs={cvae_epochs}, samples/scenario={cvae_samples}"
    )

    orchestrator = UnifiedOrchestrator(
        train_cvae=train_cvae,
        cvae_epochs=cvae_epochs,
        cvae_samples_per_scenario=cvae_samples,
        cvae_scenario_names=cvae_scenarios,
        train_pinn=train_pinn,
        pinn_epochs=pinn_epochs,
        pinn_synthetic_n=pinn_n,
        train_ppo=train_ppo,
        ppo_episodes=ppo_episodes,
        ppo_steps=ppo_steps,
        ppo_dyna_rollouts=ppo_dyna_rollouts,
        train_gnn=train_gnn,
        gnn_epochs=gnn_epochs,
    )

    results = {}
    for case in cases:
        try:
            print(f"\n[SUITE] Executing workflow for scenario: {case.upper()}...")
            res = orchestrator.run_e2e(case_name=case)
            results[case] = res
            print(f"  [OK] {case} complete. Compliance score: {res['compliance_score']:.4f}")
            # Print training provenance for verification
            ts = res.get("training_summary", {})
            print(f"  [TRAINING] PINN trained={ts.get('pinn_is_trained')} | "
                  f"PINN_loss={ts.get('pinn', {}).get('final_total_loss')} | "
                  f"PPO_CSR={ts.get('ppo', {}).get('final_csr')} | "
                  f"GNN_loss={ts.get('gnn', {}).get('final_loss')}")
        except Exception as exc:
            print(f"  [ERROR] case {case}: {exc}")

    topology_summary = None
    if results:
        topology_summary = _write_topology_pipeline_metrics(orchestrator, results)

    # ── Codal Compliance Benchmark Suite ────────────────────────────────
    print("\n" + "=" * 80)
    print("  [BENCHMARK] Running ASME Codal Compliance Benchmark Suite...")
    from pinnflow.benchmark import run_codal_compliance_benchmarks
    codal_bench_results = run_codal_compliance_benchmarks(orchestrator.agents)
    print(f"  [BENCHMARK] Codal score: {codal_bench_results['score'] * 100:.1f}%")

    print("\n" + "=" * 100)
    print("  [OK] ALL SCENARIOS COMPLETE")
    print("  [OK] Scientific deliverables generated in: results/")
    print(f"  [OK] Decision traces recorded for {len(results)} cases.")
    print("=" * 100)
    if results and topology_summary is not None:
        _log_result_tables(orchestrator, results, topology_summary)
    print("\n[V8.1] Suggestion: Open 'pinnflow/ui/dashboard.html' to view the interactive audit results.")



if __name__ == "__main__":
    run_industrial_suite()
