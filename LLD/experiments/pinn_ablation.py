"""
experiments/pinn_ablation.py
────────────────────────────
Ablation Study for PINN formulations:
1. Vanilla MLP
2. Single-Task PINN (PDE)
3. Multi-Task PINN (PDE + Fluid)
4. Multi-Task PINN + kNN (Multi-fidelity)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pinnflow.pinn import MultiTaskPINN
from pinnflow.config import EXP_DIRS, BLUE, LBLUE, GREEN, RED

def run_ablation():
    print("Starting PINN Ablation Study...")
    output_dir = EXP_DIRS["ablation"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Synthetic Dataset for Training
    n_samples = 500
    X = np.random.rand(n_samples, 8)
    # Target: σ = P*d/2t + some noise, ΔP = Darcy-Weisbach + some noise
    sigma = (X[:, 3] * X[:, 0] / (2 * X[:, 1] + 1e-6)) * 100 + np.random.randn(n_samples) * 5
    dp = (0.02 * (X[:, 2] / (X[:, 0] + 1e-6)) * 1000 * X[:, 6]**2 / 2) + np.random.randn(n_samples) * 2
    Y = np.column_stack([sigma, dp])
    
    # Split
    X_train, X_test = X[:400], X[400:]
    Y_train, Y_test = Y[:400], Y[400:]
    
    configs = [
        {"name": "MLP (Vanilla)", "use_multitask": False, "use_log_stress": False, "use_knn": False, "use_mono": False},
        {"name": "ST-PINN (PDE)", "use_multitask": False, "use_log_stress": True, "use_knn": False, "use_mono": True},
        {"name": "MT-PINN (PDE+Fluid)", "use_multitask": True, "use_log_stress": True, "use_knn": False, "use_mono": True},
        {"name": "MT-PINN + kNN", "use_multitask": True, "use_log_stress": True, "use_knn": True, "use_mono": True},
    ]
    
    results = []
    
    for conf in configs:
        print(f"  Evaluating {conf['name']}...")
        model = MultiTaskPINN(
            use_multitask=conf["use_multitask"],
            use_log_stress=conf["use_log_stress"],
            use_knn_correction=conf["use_knn"],
            use_monotonicity=conf["use_mono"]
        )
        
        model.fit(X_train, Y_train, epochs=100, verbose=False)
        Y_pred = model.predict(X_test)
        
        # Calculate MAE %
        mae_s = np.mean(np.abs(Y_test[:, 0] - Y_pred[:, 0]) / (np.mean(Y_test[:, 0]) + 1e-8)) * 100
        mae_f = np.mean(np.abs(Y_test[:, 1] - Y_pred[:, 1]) / (np.mean(Y_test[:, 1]) + 1e-8)) * 100
        
        results.append({
            "Config": conf["name"],
            "Stress_MAE_pct": mae_s,
            "Fluid_MAE_pct": mae_f
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "ablation_results.csv"), index=False)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    xb = np.arange(len(results))
    plt.bar(xb - 0.2, df["Stress_MAE_pct"], 0.4, label="Stress MAE %", color=BLUE)
    plt.bar(xb + 0.2, df["Fluid_MAE_pct"], 0.4, label="Fluid MAE %", color=GREEN)
    plt.xticks(xb, df["Config"])
    plt.ylabel("MAE % (Lower is Better)")
    plt.title("PINN Formulation Ablation Study")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(output_dir, "ablation_plot.png")
    plt.savefig(plot_path)
    print(f"Ablation results saved to {output_dir}")
    
if __name__ == "__main__":
    run_ablation()
