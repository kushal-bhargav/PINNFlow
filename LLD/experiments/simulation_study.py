"""
experiments/simulation_study.py
───────────────────────────────
Evaluation of RL stability under environmental noise and uncertainty.
Compares:
1. Deterministic Env
2. Noisy Env (Noise Level 0.05 to 0.15)
3. Noisy Env + Augmentation (Robust training)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pinnflow.pinn import MultiTaskPINN
from pinnflow.environment import PipelineEnv
from pinnflow.agent import PPOAgent
from pinnflow.config import EXP_DIRS, BLUE, RED, GREEN, PURPLE

def run_simulation_study():
    print("Starting RL Simulation Study (Environmental Robustness)...")
    output_dir = EXP_DIRS["rl"] # Reusing RL dir or creating ablation subfolder
    os.makedirs(output_dir, exist_ok=True)
    
    noise_levels = [0.0, 0.05, 0.1, 0.15]
    modes = ["deterministic", "noisy"]
    
    results = []
    
    pinn = MultiTaskPINN()
    
    for lvl in noise_levels:
        for mode in modes:
            print(f"  Evaluating Noise={lvl}, Mode={mode}...")
            env = PipelineEnv(pinn, mode=mode, noise_level=lvl)
            agent = PPOAgent()
            
            # Simple evaluation: run for a few episodes and record average reward/csr
            agent.train(env, n_ep=50, verbose=False) # Reduced for speed
            
            avg_reward = np.mean(agent.reward_hist)
            avg_csr = np.mean(agent.csr_hist)
            std_reward = np.std(agent.reward_hist)
            
            results.append({
                "Noise_Level": lvl,
                "Mode": mode,
                "Avg_Reward": avg_reward,
                "Std_Reward": std_reward,
                "Avg_CSR": avg_csr
            })
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "simulation_study_results.csv"), index=False)
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in modes:
        mode_data = df[df["Mode"] == mode]
        plt.errorbar(mode_data["Noise_Level"], mode_data["Avg_Reward"], 
                     yerr=mode_data["Std_Reward"], fmt='-o', label=f"Mode: {mode}")
        
    plt.title("RL Reward Stability vs Environmental Noise")
    plt.xlabel("Noise Level (Gaussian σ)")
    plt.ylabel("Average Reward (± Std Dev)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(output_dir, "simulation_study_plot.png")
    plt.savefig(plot_path)
    print(f"Simulation study results saved to {output_dir}")

if __name__ == "__main__":
    run_simulation_study()
