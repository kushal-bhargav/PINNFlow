"""Pipeline environment for optimization and reward evaluation."""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from pinnflow.config import ELBOW_CONFIG
from pinnflow.geometry.features import ensure_geometry_state
from pinnflow.pinn import MultiTaskPINN


class PipelineEnv:
    """Curriculum RL environment with geometry-aware 16-D observations."""

    BOUNDS = np.array(
        [
            [114, 620],
            [4, 22],
            [5, 150],
            [1, 20],
            [0, 150],
            [-40, 80],
            [0.5, 12],
            [0.3, 0.8],
            [0, 3],
            [0.3, 5.0],
            [0, 3100],
            [0, 1.0],
            [-1, 1],
            [-1, 1],
            [0, 0.2],
            [0, 1.0e7],
        ],
        dtype=float,
    )
    NAMES = [
        "Diameter",
        "Thickness",
        "Length",
        "Pressure",
        "SoilDisp",
        "DeltaT",
        "Velocity",
        "SoilK",
        "ShapeID",
        "ShapeParam",
        "BendRadius",
        "CurvatureRatio",
        "SinElbowAngle",
        "CosElbowAngle",
        "ThicknessRatio",
        "DeanNumber",
    ]

    W_FULL = np.array([0.30, 0.30, 0.15, 0.25])
    ASME_LIMIT = 200.0
    HARD_PENALTY = -2.0

    def __init__(
        self,
        pinn: MultiTaskPINN,
        curriculum: bool = True,
        mode: str = "deterministic",
        noise_level: float = 0.05,
    ):
        self.pinn = pinn
        self.curriculum = curriculum
        self.mode = mode
        self.noise_level = noise_level
        self.uncertainty_penalty_weight = ELBOW_CONFIG["uncertainty_penalty_weight"]
        self.episode = 0
        self.rolling_csr = deque(maxlen=50)
        self.state = np.zeros(16, dtype=float)
        self.reset()

    def sample_geometry_aware_batch(self, n: int) -> np.ndarray:
        return self.pinn.sample_collocation_points(n)

    def sanitize_state(self, state: np.ndarray) -> np.ndarray:
        raw = np.asarray(state, dtype=float).reshape(-1)
        if raw.size < 10:
            raise ValueError("PipelineEnv state requires at least 10 values.")
        clean = ensure_geometry_state(raw[:16] if raw.size >= 16 else raw[:10]).reshape(-1)
        clean[:10] = np.clip(clean[:10], self.BOUNDS[:10, 0], self.BOUNDS[:10, 1])
        clean[8] = float(np.clip(np.rint(clean[8]), 0, 3))
        clean[9] = float(np.clip(clean[9], self.BOUNDS[9, 0], self.BOUNDS[9, 1]))
        return ensure_geometry_state(clean).reshape(-1)

    def set_state(self, state: np.ndarray) -> np.ndarray:
        self.state = self.sanitize_state(state)
        return self.state.copy()

    def build_state_from_scenario(
        self,
        inputs: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        lines = schema.get("lines", []) if schema else []
        primary_line = lines[0] if lines else {}

        diameter = float(primary_line.get("diameter", 273.0))
        thickness = float(primary_line.get("thickness", 9.27))
        length = float(primary_line.get("design_length_m", 100.0))
        pressure = float(inputs.get("max_p", 15.0)) / 10.0
        max_temp = float(inputs.get("max_t", 65.0))
        topology = str(inputs.get("topology", "GasLib-134")).lower()

        shape_id = 2.0 if "fsi" in topology else 1.0
        shape_param = 1.2 if "fsi" in topology else 3.0
        velocity = np.clip(1.5 + pressure * 0.25, self.BOUNDS[6, 0], self.BOUNDS[6, 1])
        soil_disp = 40.0 if "refinery" in topology else 15.0
        soil_k = 0.55 if "fsi" in topology else 0.45
        delta_t = np.clip(max_temp - 20.0, self.BOUNDS[5, 0], self.BOUNDS[5, 1])

        scenario_state = np.array(
            [diameter, thickness, length, pressure, soil_disp, delta_t, velocity, soil_k, shape_id, shape_param],
            dtype=float,
        )
        return self.sanitize_state(scenario_state)

    def reset(self) -> np.ndarray:
        state = np.array(
            [
                np.random.uniform(200, 500),
                np.random.uniform(10, 20),
                np.random.uniform(5, 150),
                np.random.uniform(1, 10),
                np.random.uniform(0, 100),
                np.random.uniform(-20, 40),
                np.random.uniform(0.5, 5),
                np.random.uniform(0.3, 0.8),
                np.random.randint(0, 4),
                np.random.uniform(1.0, 5.0),
            ],
            dtype=float,
        )
        return self.set_state(state)

    def _weights(self) -> np.ndarray:
        if not self.curriculum:
            return self.W_FULL
        csr_50 = float(np.mean(self.rolling_csr)) if len(self.rolling_csr) >= 10 else 0.0
        if csr_50 >= 0.70:
            return self.W_FULL
        if csr_50 >= 0.50:
            return np.array([0.5, 0.5, 0.0, 0.0])
        return np.array([1.0, 0.0, 0.0, 0.0])

    def step(self, action: np.ndarray):
        act = np.asarray(action, dtype=float).reshape(-1)
        ranges = self.BOUNDS[:, 1] - self.BOUNDS[:, 0]
        delta = np.zeros_like(self.state)
        n_act = min(len(act), 10)
        delta[:n_act] = act[:n_act] * ranges[:n_act] * 0.05
        self.state = self.sanitize_state(self.state + delta)
        reward, info = self._reward()
        return self.state.copy(), reward, info

    def _model_state(self) -> np.ndarray:
        n_in = int(getattr(self.pinn, "n_in", len(self.state)))
        return self.state[:n_in].reshape(1, -1)

    def _reward(self):
        model_state = self._model_state()
        pred = self.pinn.predict(model_state)[0]

        if self.mode == "noisy":
            pred[0] *= 1.0 + np.random.randn() * self.noise_level
            pred[1] *= 1.0 + np.random.randn() * self.noise_level

        d = self.state[0]
        t = self.state[1]
        shape_id = int(np.clip(np.rint(self.state[8]), 0, 3))
        shape_param = float(self.state[9])
        self.state[8] = float(shape_id)

        sigma = max(float(pred[0]), 1.0)
        delta_p = max(float(pred[1]), 0.1)

        k_factor = 1.0
        if shape_id == 1:
            k_factor = 0.5 + (1.0 / (shape_param + 1e-6))
            sigma *= 1.15
        elif shape_id == 2:
            k_factor = 1.5 + shape_param
            sigma *= 1.4

        delta_p *= k_factor
        constraint_violation = float(np.maximum(0.0, sigma - self.ASME_LIMIT)) / self.ASME_LIMIT
        self.rolling_csr.append(1.0 if constraint_violation <= 0 else 0.0)

        weights = self._weights()
        fatigue = max(0.0, (sigma - 150.0) / 150.0)
        cost = np.pi * (d / 1000.0) * (t / 1000.0) * self.state[2] * 7850.0 / 1e4

        reward = -(
            weights[0] * np.clip(sigma / 200.0, 0.0, 1.0)
            + weights[1] * np.clip(delta_p / 100.0, 0.0, 1.0)
            + weights[2] * np.clip(fatigue, 0.0, 1.0)
            + weights[3] * np.clip(cost / 10.0, 0.0, 1.0)
        )
        if constraint_violation > 0:
            reward -= 5.0 * constraint_violation

        uncertainty_score = 0.0
        if hasattr(self.pinn, "predict_with_uncertainty"):
            uq_result = self.pinn.predict_with_uncertainty(model_state)
            if "uncertainty_score" in uq_result:
                uncertainty_score = float(np.asarray(uq_result["uncertainty_score"]).reshape(-1)[0])
                reward -= self.uncertainty_penalty_weight * uncertainty_score

        return reward, {
            "sigma": sigma,
            "delta_P": delta_p,
            "fatigue": fatigue,
            "cost": cost,
            "constraint_ok": constraint_violation <= 0,
            "violation": constraint_violation,
            "shape_id": shape_id,
            "shape_param": shape_param,
            "uncertainty_score": uncertainty_score,
        }
