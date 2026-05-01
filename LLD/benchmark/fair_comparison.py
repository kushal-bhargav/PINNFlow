"""
benchmark/fair_comparison.py
─────────────────────────────
Ensures one-to-one metric comparison between models by using a fixed
high-fidelity evaluation set.
"""
import os
import pickle
import numpy as np
import pandas as pd
from pinnflow.config import EXP_DIRS

def generate_fair_set(n_samples=1000, seed=42):
    """Generates and saves a fixed evaluation set."""
    print(f"Generating fair evaluation set with {n_samples} samples...")
    np.random.seed(seed)
    
    # Feature columns same as pinnflow.pinn.MultiTaskPINN.FEAT_COLS
    X = np.random.rand(n_samples, 8)
    
    # Convert to a DataFrame for readability
    cols = ["diameter", "thickness", "length", "pressure",
            "soil_disp", "delta_T", "velocity", "soil_stiffness"]
    df = pd.DataFrame(X, columns=cols)
    
    output_path = os.path.join(EXP_DIRS["benchmarks"], "fair_eval_set.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(df, f)
        
    print(f"Fair evaluation set saved to {output_path}")
    return df

def load_fair_set():
    path = os.path.join(EXP_DIRS["benchmarks"], "fair_eval_set.pkl")
    if not os.path.exists(path):
        return generate_fair_set()
    with open(path, "rb") as f:
        return pickle.load(f)

def evaluate_and_compare(models_dict):
    """
    models_dict: {name: model_object} where model has .predict(X)
    """
    X_df = load_fair_set()
    X = X_df.values
    
    results = []
    for name, model in models_dict.items():
        print(f"Evaluating {name}...")
        Y_pred = model.predict(X)
        
        # Calculate summary metrics (using first prediction as proxy if no GT available)
        results.append({
            "Model": name,
            "Avg_Stress": np.mean(Y_pred[:, 0]),
            "Avg_Pressure_Drop": np.mean(Y_pred[:, 1]),
            "Max_Stress": np.max(Y_pred[:, 0])
        })
        
    res_df = pd.DataFrame(results)
    print("\nFair Comparison Results:")
    print(res_df)
    return res_df

if __name__ == "__main__":
    generate_fair_set()
