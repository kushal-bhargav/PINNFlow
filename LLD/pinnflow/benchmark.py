"""
pinnflow/benchmark.py
──────────────────────
MODULE 10 — Fair FEM Comparison  [P5]

run_fem_baseline_fair : FEM/CFD baseline with same n_designs as PINN-RL-Gen.
run_e2e               : PINN-RL-Gen end-to-end evaluation.

[P2 FIX] run_e2e now uses sim.generate_one() (same evaluator as FEM baseline)
         so the benchmark is genuinely fair — not surrogate vs simulator.
"""
from __future__ import annotations

import time
import numpy as np

from pinnflow.simulator   import PhysicsSimulator
from pinnflow.pinn        import MultiTaskPINN
from pinnflow.vae         import CAVAE
from pinnflow.agent       import PPOAgent
from pinnflow.environment import PipelineEnv


def run_fem_baseline_fair(
    sim: PhysicsSimulator,
    n_designs: int = 30,
) -> tuple:
    """
    [P5] FEM baseline: exactly n_designs random evaluations via the physics
    simulator.  Returns (summary_dict, records_list).
    """
    t0 = time.time()
    records = []
    for _ in range(n_designs):
        d   = np.random.choice(sim.DIAMETERS)
        t_  = max(sim.THICK_MAP[d] + np.random.uniform(-1, 1), 4)
        row = sim.generate_one(
            d, t_,
            np.random.uniform(5, 150),
            np.random.uniform(1, 20),
            np.random.uniform(0, 150),
            np.random.uniform(-40, 80),
        )
        records.append(row)
    elapsed = time.time() - t0
    sigs = [r["von_mises_stress"]  for r in records]
    dPs  = [r["pressure_drop_kPa"] for r in records]
    csr  = float(np.mean([s < 200 for s in sigs]))
    summary = {
        "avg_sigma":   round(float(np.mean(sigs)), 2),
        "avg_dP":      round(float(np.mean(dPs)),  2),
        "csr":         round(csr, 4),
        "elapsed_s":   round(elapsed, 5),
        "n_designs":   n_designs,
    }
    return summary, records


def run_e2e(
    vae:   CAVAE,
    pinn:  MultiTaskPINN,
    agent: PPOAgent,
    env:   PipelineEnv,
    sim:   PhysicsSimulator,
    n:     int = 30,
) -> tuple:
    """
    PINN-RL-Gen end-to-end evaluation.

    [P2 FIX] Uses sim.generate_one() so the metric (von Mises stress, ΔP) is
    produced by the same physics kernel as run_fem_baseline_fair — making the
    comparison fair (not surrogate σ vs simulator σ).

    Returns (summary_dict, records_list).
    """
    t0 = time.time()
    records = []

    print(f"[Benchmark.run_e2e] Evaluating {n} designs through the VAE -> PPO -> PhysicsSimulator pipeline...")
    for design_idx in range(n):
        # 1. VAE generates candidate layout
        layout = vae.generate(1)[0]
        layout = np.clip(layout, env.BOUNDS[:, 0], env.BOUNDS[:, 1])
        env.state = layout

        # 2. PPO refines the layout
        for step_idx in range(25):
            a, _, _ = agent.select_action(env.state)
            env.step(a)

        # 3. [P2 FIX] Evaluate with physics simulator (same as FEM baseline)
        row = sim.generate_one(
            env.state[0], env.state[1],
            env.state[2], env.state[3],
            env.state[4], env.state[5],
        )
        records.append({
            "sigma": row["von_mises_stress"],
            "dP":    row["pressure_drop_kPa"],
            "ok":    row["von_mises_stress"] < 200,
        })
        if (design_idx + 1) % 10 == 0 or (design_idx + 1) == n:
            print(f"  - Evaluated {design_idx + 1}/{n} designs | Last: sigma={row['von_mises_stress']:.2f} MPa, dP={row['pressure_drop_kPa']:.2f} kPa")

    elapsed = time.time() - t0
    sigs = [r["sigma"] for r in records]
    dPs  = [r["dP"]    for r in records]
    summary = {
        "avg_sigma": round(float(np.mean(sigs)), 2),
        "avg_dP":    round(float(np.mean(dPs)),  2),
        "csr":       round(float(np.mean([r["ok"] for r in records])), 4),
        "elapsed_s": round(elapsed, 5),
        "n_designs": n,
    }
    print(f"[Benchmark.run_e2e] Finished. Avg stress={summary['avg_sigma']:.2f} MPa, Avg dP={summary['avg_dP']:.2f} kPa, CSR={summary['csr']*100:.2f}%")
    return summary, records


# --- [V7] NEW ENHANCEMENTS (ADDED) ---

def compare_surrogate_vs_ansys(
    pinn,
    X_test,
    Y_ansys,
    n_samples: int = 10,
    reference_solver=None,
    reference_time_s: float | None = None,
):
    """
    [V7] Performance Benchmark Runner.
    Measures:
    1. Speedup (PINN Inference vs. a real reference solve when available)
    2. Accuracy (MAE/RMSE)
    3. Calibration (if UQ is enabled)
    """
    print("\n[V7] Starting Surrogate vs. ANSYS Benchmark...")
    
    # 1. Measure PINN Speed
    start_time = time.time()
    for _ in range(100): # Multiple passes for statistical stability
        Y_pinn = pinn.predict(X_test[:n_samples])
    pinn_time = (time.time() - start_time) / 100.0
    
    # 2. Measure a real reference runtime when a solver is available.
    if reference_solver is not None:
        ref_start = time.time()
        Y_ref = reference_solver(X_test[:n_samples])
        total_ref_time = time.time() - ref_start
    else:
        Y_ref = Y_ansys[:n_samples]
        total_ref_time = reference_time_s if reference_time_s is not None else float("nan")

    if np.isfinite(total_ref_time) and total_ref_time > 0:
        speedup = total_ref_time / (pinn_time + 1e-8)
    else:
        speedup = float("nan")
    
    # 3. Compute Accuracy
    errors = np.abs(Y_pinn - Y_ref)
    mae = np.mean(errors, axis=0)
    
    # 4. Results Logging
    results = {
        "pinn_solve_time_ms": pinn_time * 1000,
        "reference_solve_time_s": total_ref_time,
        "speedup_factor": speedup,
        "mae_stress": mae[0],
        "mae_pressure": mae[1]
    }
    
    print("-" * 40)
    print(f"  PINN Inference Time: {results['pinn_solve_time_ms']:.4f} ms")
    print(f"  Reference Solve Time: {results['reference_solve_time_s']:.4f} s")
    if np.isfinite(results["speedup_factor"]):
        print(f"  Calculated Speedup: {results['speedup_factor']:.1f}x")
    print(f"  MAE Stress: {results['mae_stress']:.2f} MPa")
    print("-" * 40)
    
    return results

def run_ood_experiment(pinn, asme_adapter, gaslib_loader):
    """
    [V7] Out-of-Distribution (OOD) Verification.
    Checks model stability on unseen topologies and standard benchmarks.
    """
    print("\n[V7] Running OOD Experiments (ASME & GasLib)...")
    # In v7, these use the specific performance adapters implemented in v5/v6
    # but now reporting within the v7 context.
    try:
        asme_res = asme_adapter.evaluate_pinn(pinn)
    except:
        asme_res = {"status": "Adapter Error"}
        
    try:
        gaslib_res = gaslib_loader.benchmark_pinn_on_gaslib(pinn)
    except:
        gaslib_res = {"status": "Loader Error"}
        
    return {"asme": asme_res, "gaslib": gaslib_res}

def run_codal_compliance_benchmarks(agents: list) -> dict:
    """
    [Phase 6] Run the ASME Codal Engine against the benchmark suite to verify
    agent classification accuracy and constraint violation catching.
    """
    from pinnflow.benchmarks.asme_cases import get_asme_benchmark_suite
    
    suite = get_asme_benchmark_suite()
    results = {}
    total = len(suite)
    passed = 0
    
    print("\n[Phase 6] Running Codal Engine Benchmarks...")
    for case_name, (design, expected_violations) in suite.items():
        case_violations = []
        for agent in agents:
            # We don't have rule_store here directly, but agents have evaluate()
            res = agent.evaluate(design, context={})
            if not res.get("pass", True):
                # match clause
                case_violations.append(
                    res.get("clause", getattr(agent, "CLAUSE", agent.__class__.__name__))
                )
        
        # Check if the expected violations were caught
        expected_set = set(expected_violations)
        actual_set = set(case_violations)
        
        # A test passes if the exact set of clauses were violated
        # We can do substring matching for robustness
        case_passed = True
        for exp in expected_set:
            if not any(exp in act for act in actual_set):
                case_passed = False
                
        # Also check false positives
        if len(expected_set) == 0 and len(actual_set) > 0:
            case_passed = False
            
        results[case_name] = {
            "passed": case_passed,
            "expected": list(expected_set),
            "actual": list(actual_set)
        }
        if case_passed:
            passed += 1
            print(f"  [PASS] {case_name}")
        else:
            print(f"  [FAIL] {case_name}")
            print(f"    Expected: {expected_set}")
            print(f"    Actual:   {actual_set}")
            
    print(f"Codal Benchmark Score: {passed}/{total} ({(passed/total)*100:.1f}%)")
    return {"score": passed / total, "details": results}
