"""Evaluation package for per-class and calibration diagnostics.

This package also preserves the legacy ``pinnflow.evaluation`` functions.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from pinnflow.evaluation.calibration import expected_calibration_error
from pinnflow.evaluation.ood import topology_ood_report
from pinnflow.evaluation.per_class import (
    GEOMETRY_LABELS,
    hotspot_localization_error,
    per_class_regression_metrics,
    robustness_under_noise,
)


def eval_pinn(pinn, X_test: np.ndarray, Y_test: np.ndarray):
    """Evaluate a PINN-like model on held-out test data."""
    Y_pred = pinn.predict(X_test)
    out = {}
    for i, nm in enumerate(["von_mises_stress", "pressure_drop_kPa"]):
        mae = mean_absolute_error(Y_test[:, i], Y_pred[:, i])
        r2 = r2_score(Y_test[:, i], Y_pred[:, i])
        rng = Y_test[:, i].max() - Y_test[:, i].min()
        out[nm] = {
            "MAE": round(float(mae), 4),
            "MAE_pct": round(float(mae / max(rng, 1e-8) * 100), 2),
            "R2": round(float(r2), 4),
        }
    return out, Y_pred


def eval_rl(agent, rand_rewards: list) -> dict:
    """Compare final PPO and random-policy reward windows."""
    ppo = agent.reward_hist
    imp = (
        (np.mean(ppo[-50:]) - np.mean(rand_rewards[-50:]))
        / (abs(np.mean(rand_rewards[-50:])) + 1e-8)
        * 100
    )
    return {
        "ppo_final": round(float(np.mean(ppo[-50:])), 4),
        "rand_final": round(float(np.mean(rand_rewards[-50:])), 4),
        "improvement_pct": round(float(imp), 2),
        "final_csr": round(float(np.mean(agent.csr_hist[-50:])), 4),
    }


def random_baseline(env, n: int = 400, steps: int = 25) -> list:
    """Roll out random actions and collect per-episode average rewards."""
    rewards = []
    for _ in range(n):
        env.reset()
        ep_r = 0.0
        for _ in range(steps):
            action_dim = min(10, len(env.state))
            a = np.random.uniform(-1, 1, size=action_dim)
            _, r, _ = env.step(a)
            ep_r += r
        rewards.append(ep_r / steps)
    return rewards


__all__ = [
    "GEOMETRY_LABELS",
    "eval_pinn",
    "eval_rl",
    "expected_calibration_error",
    "hotspot_localization_error",
    "per_class_regression_metrics",
    "random_baseline",
    "robustness_under_noise",
    "topology_ood_report",
]
