"""
pinnflow/evaluation.py
───────────────────────
MODULE 11 — Evaluation helpers

eval_pinn       : computes MAE %, R² for both PINN output channels.
eval_rl         : compares PPO vs random reward histories.
random_baseline : rolls out a purely random policy to establish baseline rewards.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Tuple

from pinnflow.pinn import MultiTaskPINN
from pinnflow.agent import PPOAgent
from pinnflow.environment import PipelineEnv


def eval_pinn(
    pinn: MultiTaskPINN,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> Tuple[dict, np.ndarray]:
    """
    Evaluate PINN on held-out test data.

    Returns
    -------
    metrics : dict keyed by output name, each containing MAE, MAE_pct, R².
    Y_pred  : predicted array of shape (N, 2).
    """
    Y_pred = pinn.predict(X_test)
    out    = {}
    for i, nm in enumerate(["von_mises_stress", "pressure_drop_kPa"]):
        mae = mean_absolute_error(Y_test[:, i], Y_pred[:, i])
        r2  = r2_score(Y_test[:, i], Y_pred[:, i])
        rng = Y_test[:, i].max() - Y_test[:, i].min()
        out[nm] = {
            "MAE":     round(float(mae), 4),
            "MAE_pct": round(float(mae / rng * 100), 2),
            "R2":      round(float(r2), 4),
        }
    return out, Y_pred


def eval_rl(agent: PPOAgent, rand_rewards: list) -> dict:
    """
    Compare final 50-episode windows of PPO vs random-policy reward.
    Returns improvement percentage and final CSR.
    """
    ppo = agent.reward_hist
    imp = (
        (np.mean(ppo[-50:]) - np.mean(rand_rewards[-50:]))
        / (abs(np.mean(rand_rewards[-50:])) + 1e-8)
        * 100
    )
    return {
        "ppo_final":       round(float(np.mean(ppo[-50:])), 4),
        "rand_final":      round(float(np.mean(rand_rewards[-50:])), 4),
        "improvement_pct": round(float(imp), 2),
        "final_csr":       round(float(np.mean(agent.csr_hist[-50:])), 4),
    }


def random_baseline(
    env: PipelineEnv,
    n: int = 400,
    steps: int = 25,
) -> list:
    """Roll out purely random actions and collect per-episode average rewards."""
    rewards = []
    for _ in range(n):
        env.reset(); ep_r = 0.0
        for _ in range(steps):
            a = np.random.uniform(-1, 1, size=8)
            _, r, _ = env.step(a)
            ep_r += r
        rewards.append(ep_r / steps)
    return rewards
