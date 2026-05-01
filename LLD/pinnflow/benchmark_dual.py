"""
pinnflow/benchmark_dual.py
──────────────────────────
DUAL-MODE TOPOLOGY COMPARISON

Evaluates the model on both Authentic GasLib topology (ASM) and Synthetic Graph theory proxies.
Generates the definitive comparison report for scientific integrity.
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from pinnflow.pinn import MultiTaskPINN
from data.gaslib_loader import GasLibLoader

def run_dual_comparison():
    print("="*80)
    print("  PINNFLOW DUAL-MODE COMPARISON: AUTHENTIC vs SYNTHETIC")
    print("="*80)
    
    # 1. Initialize PINN and data
    loader = GasLibLoader()
    pinn = MultiTaskPINN(use_log_stress=True)

    # Train on real GasLib topology + nomination rows so the benchmark uses
    # published network data rather than synthetic proxy samples.
    X_train, Y_train, _ = loader.build_real_benchmark_dataset(
        network_name="GasLib-134",
        scenario_limit=5,
        edge_limit=240,
    )
    pinn.fit(X_train, Y_train, epochs=50, verbose=False)
    
    # 2. Dual Evaluation
    print("\n[V8.1] Starting Dual-Mode Evaluation on GasLib-134...")
    dual_results = loader.benchmark_dual_mode(
        pinn,
        network_name="GasLib-134",
        scenario_limit=5,
        n_test=120,
    )
    
    # 3. Process and Report
    report = []
    for mode, metrics in dual_results.items():
        report.append({
            "Topology Mode": mode.upper(),
            "Network": "GasLib-134",
            "N_Edges": metrics["edge_count"],
            "Stress MAE (MPa)": metrics["mae_stress"],
            "Pressure MAE (kPa)": metrics["mae_dP"],
            "Reliability Index": float(np.clip(1.0 - (metrics["mae_stress"] / 200.0), 0.0, 1.0)),
        })
    
    df = pd.DataFrame(report)
    print("\n" + df.to_string(index=False))
    
    # Calculate Generalization Delta
    auth_mae = dual_results["authentic"]["mae_stress"]
    synt_mae = dual_results["synthetic"]["mae_stress"]
    delta = abs(auth_mae - synt_mae)
    print(f"\n[CONCLUSION] Topological Generalization Delta: {delta:.4f} MPa")
    
    os.makedirs("results/benchmarks", exist_ok=True)
    df.to_csv("results/benchmarks/synthetic_vs_authentic.csv", index=False)
    print("\n[OK] Comparison results saved to results/benchmarks/synthetic_vs_authentic.csv")
    print("="*80)

if __name__ == "__main__":
    run_dual_comparison()
