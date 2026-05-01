"""
experiments/runner.py
──────────────────────
Central Orchestrator for the PINN-RL Experimentation Framework.
Usage:
  python experiments/runner.py --type ablation
  python experiments/runner.py --type rl
  python experiments/runner.py --type benchmarks
  python experiments/runner.py --type all
"""
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from experiments.hyperopt import HyperparameterOptimizer
from experiments.pinn_ablation import run_ablation
from experiments.rl_agents import run_rl_comparison
from experiments.simulation_study import run_simulation_study
from experiments.literature_benchmarks import PipelineBenchmarks

def main():
    parser = argparse.ArgumentParser(description="PINN-RL Experiment Runner")
    parser.add_argument("--type", type=str, required=True, 
                        choices=["hyperopt", "ablation", "rl", "simulation", "benchmarks", "all"],
                        help="Type of experiment to run")
    
    args = parser.parse_args()
    
    print(f"====================================================")
    print(f"      PINN-RL Experiment Runner: {args.type.upper()}")
    print(f"====================================================")

    if args.type == "hyperopt" or args.type == "all":
        opt = HyperparameterOptimizer(model_type="pinn")
        opt.run_optimization(method="random", n_trials=5)

    if args.type == "ablation" or args.type == "all":
        run_ablation()

    if args.type == "rl" or args.type == "all":
        run_rl_comparison()

    if args.type == "simulation" or args.type == "all":
        run_simulation_study()

    if args.type == "benchmarks" or args.type == "all":
        bench = PipelineBenchmarks()
        bench.run_all()

    print(f"\n[DONE] All requested experiments completed.")

if __name__ == "__main__":
    main()
