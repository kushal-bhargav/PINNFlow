"""
Closed-loop optimization engine.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from pinnflow.agent import LagrangianPPOAgent
from pinnflow.environment import PipelineEnv
from pinnflow.pinn import MultiTaskPINN


class ClosedLoopOptim:
    """
    Refines designs until convergence against meaningful stabilization criteria.
    """

    def __init__(
        self,
        env: PipelineEnv,
        agent: LagrangianPPOAgent,
        pinn: MultiTaskPINN,
        fea_callback: Optional[Callable] = None,
    ):
        self.env = env
        self.agent = agent
        self.pinn = pinn
        self.fea_callback = fea_callback

    def _seed_environment(self, state: np.ndarray) -> None:
        if hasattr(self.env, "set_state"):
            self.env.set_state(state)
            if len(np.asarray(state).reshape(-1)) < len(getattr(self.env, "state", state)):
                self.env.state = np.asarray(state, dtype=float).copy()
        elif hasattr(self.env, "env") and hasattr(self.env.env, "set_state"):
            self.env.env.set_state(state)
        else:
            self.env.state = state.copy()

    def _normalized_state_delta(self, prev_state: np.ndarray, next_state: np.ndarray) -> float:
        prev = np.asarray(prev_state, dtype=float).reshape(-1)
        nxt = np.asarray(next_state, dtype=float).reshape(-1)
        n = min(len(prev), len(nxt))
        bounds = getattr(self.env, "BOUNDS", None)
        if bounds is None:
            scale = np.ones(n)
        else:
            scale = np.maximum(bounds[:n, 1] - bounds[:n, 0], 1e-6)
        delta = (nxt[:n] - prev[:n]) / scale
        return float(np.linalg.norm(delta) / np.sqrt(len(delta)))

    def _apply_codal_guidance(self, state: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Blend the current design toward the most relevant codal recommendations.

        This mimics a human reviewer applying code-based design corrections
        before the next optimization iteration.
        """
        recommendations = info.get("codal_recommendations", []) or []
        guided = np.asarray(state, dtype=float).copy()
        bounds = getattr(self.env, "BOUNDS", None)
        if bounds is None:
            return guided

        for rec in recommendations:
            idx = rec.get("index")
            if idx is None or idx < 0 or idx >= len(guided):
                continue
            current = float(guided[idx])
            target = float(rec.get("target", current))
            direction = str(rec.get("direction", "hold")).lower()
            if direction == "hold":
                continue

            step_fraction = 0.25 if rec.get("parameter") in {"thickness", "diameter"} else 0.15
            updated = current + (target - current) * step_fraction
            guided[idx] = updated

        n = min(len(guided), len(bounds))
        guided[:n] = np.clip(guided[:n], bounds[:n, 0], bounds[:n, 1])
        guided[8] = float(np.clip(np.rint(guided[8]), bounds[8, 0], bounds[8, 1]))
        guided[9] = float(np.clip(guided[9], bounds[9, 0], bounds[9, 1]))
        return guided

    def run_refinement(
        self,
        initial_state: np.ndarray,
        max_iters: int = 10,
        min_iters: int = 3,
    ) -> Dict[str, Any]:
        print(f"[ClosedLoop] Starting refinement loop (max {max_iters} iterations)...")
        state = np.asarray(initial_state, dtype=float).copy()
        self._seed_environment(state)

        base_state = getattr(getattr(self.env, "env", self.env), "state", None)
        if base_state is not None:
            print(f"[ClosedLoop] Seeded state: wrapper={np.round(state, 3)} base={np.round(base_state, 3)}")

        history = []
        previous_reward = None
        convergence_reason = "max_iters_reached"
        last_external_error = None

        for i in range(max_iters):
            action, _, _ = self.agent.select_action(state)
            next_state, reward, info = self.env.step(action)

            state_delta = self._normalized_state_delta(state, next_state)
            reward_delta = abs(reward - previous_reward) if previous_reward is not None else np.inf

            if self.fea_callback is not None:
                external_metrics = self.fea_callback(next_state)
                external_sigma = float(external_metrics.get("sigma", info["sigma"]))
                external_error = abs(external_sigma - info["sigma"])
            else:
                external_sigma = None
                external_error = None
            last_external_error = external_error

            if self.fea_callback is not None:
                converged = (
                    i + 1 >= min_iters
                    and external_error is not None
                    and external_error < 1.0
                    and info["constraint_ok"]
                )
            else:
                converged = (
                    i + 1 >= min_iters
                    and state_delta < 0.02
                    and reward_delta < 0.05
                    and info["constraint_ok"]
                    and info.get("compliance_score", 1.0) >= 0.90
                )

            history.append(
                {
                    "iter": i,
                    "sigma": info["sigma"],
                    "reward": reward,
                    "state_delta": state_delta,
                    "reward_delta": reward_delta,
                    "external_sigma": external_sigma,
                    "external_error": external_error,
                    "compliance_score": info.get("compliance_score"),
                    "converged": converged,
                }
            )

            state = next_state[: len(state)] if len(next_state) != len(state) else next_state
            if info.get("codal_recommendations"):
                state = self._apply_codal_guidance(state, info)
                self._seed_environment(state)
            previous_reward = reward

            if converged:
                convergence_reason = "external_validation" if self.fea_callback is not None else "state_stabilized"
                print(f"  [OK] Converged at iteration {i} via {convergence_reason}")
                break

        info["convergence_reason"] = convergence_reason
        info["external_error"] = last_external_error
        return {
            "final_state": state,
            "iterations": i + 1,
            "convergence_history": history,
            "metrics": info,
        }
