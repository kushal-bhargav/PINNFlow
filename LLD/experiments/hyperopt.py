"""
experiments/hyperopt.py
───────────────────────
Hyperparameter Optimization for PINN and RL.
Supports Coarse Grid Search and Random Search.
"""
import os
import json
import itertools
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pinnflow.config import SEARCH_SPACE_PINN, SEARCH_SPACE_RL, EXP_DIRS

class HyperparameterOptimizer:
    def __init__(self, model_type="pinn"):
        self.model_type = model_type
        self.search_space = SEARCH_SPACE_PINN if model_type == "pinn" else SEARCH_SPACE_RL
        self.output_dir = EXP_DIRS["hyperopt"]
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _generate_grid(self):
        keys, values = zip(*self.search_space.items())
        grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return grid

    def _generate_random(self, n_trials=20):
        # Sample uniformly from the lists in search_space
        trials = []
        for _ in range(n_trials):
            trial = {k: random.choice(v) for k, v in self.search_space.items()}
            trials.append(trial)
        return trials

    def run_optimization(self, method="grid", n_trials=None):
        print(f"Starting {method} optimization for {self.model_type}...")
        
        if method == "grid":
            configs = self._generate_grid()
        else:
            configs = self._generate_random(n_trials or 10)
            
        results = []
        best_score = float('inf')
        best_config = None
        
        for i, config in enumerate(configs):
            print(f"Trial {i+1}/{len(configs)}: {config}")
            
            # MOCK EVALUATION (Replace with actual training loop call)
            # In a real scenario, this would import MultiTaskPINN or PPOAgent and call .fit()
            score = self._mock_train(config)
            
            result = config.copy()
            result["score"] = score
            results.append(result)
            
            if score < best_score:
                best_score = score
                best_config = config
                
            # Periodically save results
            if (i + 1) % 5 == 0:
                self._save_results(results, f"temp_{self.model_type}_{method}.csv")

        final_filename = f"{self.model_type}_{method}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        self._save_results(results, final_filename)
        
        print(f"Optimization finished. Best score: {best_score:.4f}")
        print(f"Best config: {best_config}")
        return best_config

    def _mock_train(self, config):
        """Placeholder for actual training loop."""
        # Simulate a loss function that favors lower learning rates and specific penalties
        lr = config.get("lr", 0.001)
        lam_f = config.get("lam_f", 0.05)
        # Random component + some local minima
        base = 10.0 * lr + 5.0 * lam_f + random.uniform(0, 0.5)
        return base

    def _save_results(self, results, filename):
        path = os.path.join(self.output_dir, filename)
        df = pd.DataFrame(results)
        df.sort_values("score", ascending=True, inplace=True)
        df.to_csv(path, index=False)
        print(f"Saved results to {path}")

if __name__ == "__main__":
    optimizer = HyperparameterOptimizer(model_type="pinn")
    best = optimizer.run_optimization(method="random", n_trials=10)
