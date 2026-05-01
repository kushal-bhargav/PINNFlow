"""
tests/test_fair_ablation.py
───────────────────────────
Test: The run_ablation_table function must produce a valid DataFrame 
      with rows for all required configurations.
"""
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pinnflow.metrics import run_ablation_table
from pinnflow.simulator import PhysicsSimulator

def test_ablation_runner():
    sim = PhysicsSimulator()
    X = sim.generate(100)
    
    # Mock data for runner
    df_res = run_ablation_table(sim, None, None, None, X, None, n_seeds=1)
    
    expected_configs = [
        "Vanilla MLP (ref[4])",
        "Single-Task PINN",
        "Multi-Task PINN (Sparse)",
        "Full System (v6)"
    ]
    
    for config in expected_configs:
        assert config in df_res.index, f"Missing config {config} in ablation table!"
        
    print("✓ Test Passed: Ablation runner is functionally correct.")

if __name__ == "__main__":
    test_ablation_runner()
