"""
Codal reinforcement wrapper.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from pinnflow.codal_engine.agents.base import CritiqueAgent
from pinnflow.environment import PipelineEnv


class CodalEnvironmentWrapper:
    """
    Wraps the base environment and injects critique-agent penalties.
    """

    def __init__(self, env: PipelineEnv, agents: List[CritiqueAgent]):
        self.env = env
        self.agents = agents
        self.last_compliance_report: List[Dict[str, Any]] = []
        self.codal_penalty_weight = 1.0
        self.verbose = False

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def set_state(self, state: np.ndarray):
        return self.env.set_state(state)

    def set_codal_penalty_weight(self, weight: float) -> None:
        self.codal_penalty_weight = max(float(weight), 0.0)

    def step(self, action: np.ndarray):
        state, reward, info = self.env.step(action)
        info["verbose"] = self.verbose

        total_penalty = 0.0
        report = []
        recommendations = []
        for agent in self.agents:
            result = agent.evaluate(state, info)
            total_penalty += result["penalty"]
            report.append(result)
            recommendations.extend(result.get("recommendations", []))

        weighted_penalty = total_penalty * self.codal_penalty_weight
        modified_reward = reward - weighted_penalty

        self.last_compliance_report = report
        info["raw_codal_penalty"] = total_penalty
        info["codal_penalty"] = weighted_penalty
        info["codal_penalty_weight"] = self.codal_penalty_weight
        info["codal_report"] = report
        info["codal_recommendations"] = recommendations
        info["codal_rule_values"] = [item.get("rule_value", {}) for item in report]
        info["compliance_score"] = 1.0 / (1.0 + weighted_penalty)
        return state, modified_reward, info

    def get_action_space(self):
        return self.env.action_space

    def __getattr__(self, name):
        return getattr(self.env, name)
