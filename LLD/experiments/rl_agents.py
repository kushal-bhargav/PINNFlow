"""
experiments/rl_agents.py
────────────────────────
Comparison of RL Agents in the Curriculum Pipeline Environment:
1. PPO Agent [Ours]
2. Random Baseline
3. Greedy Baseline
4. Evolutionary (CEM) Baseline
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pinnflow.pinn import MultiTaskPINN
from pinnflow.environment import PipelineEnv
from pinnflow.agent import PPOAgent, RandomAgent, GreedyAgent, EvolutionaryAgent
from pinnflow.config import EXP_DIRS, BLUE, RED, GREEN, PURPLE

def run_rl_comparison():
    print("Starting RL Agent Comparison...")
    output_dir = EXP_DIRS["rl"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Setup Environment
    # We use a dummy PINN for fast evaluation, or a pre-trained one
    pinn = MultiTaskPINN() # Fast init
    env = PipelineEnv(pinn, curriculum=True)
    
    n_episodes = 300 # Enough to see curriculum phase transitions
    
    agents = [
        {"name": "PPO [Ours]", "class": PPOAgent, "color": BLUE},
        {"name": "Random",     "class": RandomAgent, "color": RED},
        {"name": "Greedy",     "class": GreedyAgent, "color": GREEN},
        {"name": "CEM (Evo)",  "class": EvolutionaryAgent, "color": PURPLE},
    ]
    
    all_rewards = {}
    all_csr = {}
    
    for ag_conf in agents:
        print(f"  Training {ag_conf['name']}...")
        if ag_conf["class"] == EvolutionaryAgent:
            agent = ag_conf["class"](pop_size=20)
        else:
            agent = ag_conf["class"]()
            
        agent.train(env, n_ep=n_episodes, verbose=False)
        all_rewards[ag_conf["name"]] = agent.reward_hist
        all_csr[ag_conf["name"]] = agent.csr_hist

    # 2. Visualisation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    for ag_conf in agents:
        name = ag_conf["name"]
        color = ag_conf["color"]
        
        # Smooth rewards
        rewards = pd.Series(all_rewards[name]).rolling(10).mean()
        csr = pd.Series(all_csr[name]).rolling(10).mean()
        
        ax1.plot(rewards, label=name, color=color, lw=2)
        ax2.plot(csr, label=name, color=color, lw=2)
        
    ax1.axvline(100, color="gray", linestyle="--", alpha=0.5, label="Phase 2 Start")
    ax1.axvline(200, color="gray", linestyle="--", alpha=0.5, label="Phase 3 Start")
    ax1.set_title("Reward Comparison (Curriculum Phases)")
    ax1.set_ylabel("Mean Reward")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_title("Constraint Satisfaction Rate (CSR)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("CSR")
    ax2.set_ylim(0, 1.1)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "rl_comparison.png")
    plt.savefig(plot_path)
    
    # 3. Save Metrics
    summary = []
    for name in all_rewards.keys():
        summary.append({
            "Agent": name,
            "Final_Reward": np.mean(all_rewards[name][-50:]),
            "Final_CSR": np.mean(all_csr[name][-50:])
        })
    pd.DataFrame(summary).to_csv(os.path.join(output_dir, "rl_metrics.csv"), index=False)
    
    print(f"RL Agent comparison results saved to {output_dir}")

if __name__ == "__main__":
    run_rl_comparison()
