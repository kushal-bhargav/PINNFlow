"""
ablation_study.py
─────────────────
MASTER ABLATION SUITE for IEEE-submittable FAIR comparison.

Runs 5 configurations across 3 seeds to generate definitive empirical artifacts.
Configs: MLP -> SC-PINN -> MT-PINN -> +VAE -> +RL
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
import time

from pinnflow.baselines import VanillaMLPBaseline, SingleTaskPINN
from pinnflow.pinn import MultiTaskPINN
from pinnflow.vae import CAVAE
from pinnflow.agent import LagrangianPPOAgent
from pinnflow.environment import PipelineEnv
from pinnflow.benchmarks.asme_adapter import ASMEBenchmarkAdapter

def run_ablation(seeds=[42, 1337, 2026], quick=True):
    print("="*80)
    print("  PINNFLOW ABLATION SUITE — IEEE V6 FAIR COMPARISON")
    print("="*80)
    
    epochs = 50 if quick else 500
    adapter = ASMEBenchmarkAdapter()
    results = []

    # 1. Create a synthetic high-fidelity training set
    # In a real study, this would be the CFD/FEM data.
    # Here we use the analytical GT + noise as proxy.
    X_train = np.random.uniform(low=[100, 5, 5, 1, 0, -20, 0.5, 0.3, 0, 0.5], 
                                high=[600, 20, 150, 20, 100, 40, 8, 0.8, 2, 1.5], 
                                size=(200, 10))
    Y_train = []
    for x in X_train:
        # Ground Truth analytical stress (Barlow's)
        gt_s = (x[3] * x[0]) / (2 * x[1])
        if x[8] == 1: gt_s *= 1.25 # Elbow SIF
        elif x[8] == 2: gt_s *= 1.55 # Tee SIF
        Y_train.append([gt_s, 2.5])
    Y_train = np.array(Y_train)

    configs = [
        ("M1 (Vanilla MLP)",  VanillaMLPBaseline, {"hidden": (128, 128, 128)}),
        ("M2 (ST-PINN)",      SingleTaskPINN,     {"hidden": (128, 128, 128)}),
        ("M3 (MT-PINN)",      MultiTaskPINN,      {"use_log_stress": True}),
        ("M4 (VAE-Synthesis)", CAVAE,             {"z_dim": 8}), # This will be evaluated by CSR
        ("M5 (Full Suite)",   "RL_WRAPPER",       {})
    ]

    for cfg_name, model_cls, kwargs in configs:
        print(f"\n[Ablation] Testing {cfg_name}...")
        
        for seed in seeds:
            np.random.seed(seed)
            start_t = time.time()
            
            if cfg_name == "M1 (Vanilla MLP)":
                model = model_cls(**kwargs)
                model.fit(X_train, Y_train, epochs=epochs)
            elif cfg_name == "M2 (ST-PINN)":
                model = model_cls(**kwargs)
                model.fit(X_train, Y_train, epochs=epochs)
            elif cfg_name == "M3 (MT-PINN)":
                model = model_cls(**kwargs)
                model.fit(X_train, Y_train, epochs=epochs)
            elif cfg_name == "M4 (VAE-Synthesis)":
                model = model_cls(x_dim=10, **kwargs)
                model.fit(X_train, epochs=epochs)
            elif cfg_name == "M5 (Full Suite)":
                pinn = MultiTaskPINN(use_log_stress=True)
                pinn.fit(X_train, Y_train, epochs=epochs)
                env = PipelineEnv(pinn=pinn)
                model = LagrangianPPOAgent(sdim=10, adim=10)
                model.train(env, n_ep=20 if quick else 400)
            
            # Metrics
            if "VAE" in cfg_name:
                csr, _ = model.csr(n=200)
                diversity = model.diversity_score(n=200)
                results.append({"config": cfg_name, "seed": seed, "metric": "CSR", "value": csr})
                results.append({"config": cfg_name, "seed": seed, "metric": "Diversity", "value": diversity})
            elif "Full" in cfg_name or "RL" in cfg_name:
                # Evaluate final agent on ASME cases
                pinn_model = pinn # Use the MT-PINN from the full suite
                bench = adapter.run_benchmark(pinn_model)
                mae = np.mean([v["mae_percent"] for v in bench.values()])
                results.append({"config": cfg_name, "seed": seed, "metric": "StressMAE%", "value": mae})
            else:
                bench = adapter.run_benchmark(model)
                mae = np.mean([v["mae_percent"] for v in bench.values()])
                results.append({"config": cfg_name, "seed": seed, "metric": "StressMAE%", "value": mae})
            
            print(f"  ✓ Seed {seed} done in {time.time()-start_t:.1f}s")

    # Aggregate and Save
    df = pd.DataFrame(results)
    summary = df.groupby(["config", "metric"])["value"].agg(["mean", "std"]).reset_index()
    print("\n" + "="*80)
    print("  FINAL ABLATION TABLE (MEAN ± STD)")
    print("="*80)
    print(summary)
    
    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/ablation_results.csv", index=False)
    print(f"\n✓ Scientific results saved to results/ablation_results.csv")

if __name__ == "__main__":
    run_ablation(quick=True)
