"""
pinnflow/config.py
──────────────────
Global constants shared across all modules.
"""
import os
import numpy as np

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Experiment Settings ──────────────────────────────────────────────────────
# Default Search Spaces for Hyperopt (Grid + Random)
SEARCH_SPACE_PINN = {
    "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "lam_s": [0.001, 0.005, 0.01, 0.05],
    "lam_f": [0.005, 0.02, 0.05, 0.1],
    "mu_bc": [0.005, 0.01, 0.05],
    "nu_mono": [0.0005, 0.001, 0.005]
}

SEARCH_SPACE_RL = {
    "lr": [1e-4, 3e-4, 1e-3],
    "gamma": [0.95, 0.99],
    "clip_ratio": [0.1, 0.2, 0.3],
    "ent_coef": [0.0, 0.01, 0.05]
}

# ── Output directories ─────────────────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Sub-directories for v4 Experimentation Framework
EXP_DIRS = {
    "hyperopt": os.path.join(RESULTS_DIR, "hyperopt"),
    "ablation": os.path.join(RESULTS_DIR, "ablation"),
    "rl":       os.path.join(RESULTS_DIR, "rl"),
    "benchmarks": os.path.join(RESULTS_DIR, "benchmarks"),
    "v3_legacy": "pinnflow_v3_results"  # Legacy compatibility
}
for d in EXP_DIRS.values():
    os.makedirs(d, exist_ok=True)

# ── Colour palette (matplotlib hex codes) ────────────────────────────────────
BLUE   = "#1F4E79"
LBLUE  = "#2E75B6"
GREEN  = "#1E8449"
RED    = "#C0392B"
GOLD   = "#B7950B"
PURPLE = "#7D3C98"
TEAL   = "#117A65"
ORANGE = "#D35400"
GRAY   = "#566573"

# ── v1 baseline reference metrics (for comparison plots) ─────────────────────
V1_METRICS = {
    "stress_mae_pct":  7.55,
    "pressure_mae_pct": 12.52,
    "stress_r2":        0.8752,
    "pressure_r2":      0.1115,
    "gen_csr":          0.018,
    "ppo_impr":         11.95,
    "e2e_csr":          0.75,
}

ELBOW_CONFIG = {
    "moe_experts": 4,
    "elbow_expert_width": 256,
    "elbow_expert_depth": 3,
    "use_refinement": True,
    "refinement_alpha": 0.8,
    "fourier_n_base": 64,
    "fourier_n_curve": 32,
    "loss_balancer": "gradnorm",
    "curriculum_stages": [200, 150, 200, 150],
    "uq_method": "ensemble",
    "n_ensemble": 3,
    "mc_dropout_p": 0.1,
    "mc_samples": 50,
    "elbow_sample_ratio": 0.40,
    "grad_norm_alpha": 0.12,
    "uncertainty_penalty_weight": 0.5,
}
