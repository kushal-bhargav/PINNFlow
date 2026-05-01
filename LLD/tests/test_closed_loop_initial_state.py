from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinnflow import ClosedLoopOptim, MultiTaskPINN, PipelineEnv, PPOAgent


def test_closed_loop_uses_provided_initial_state():
    pinn = MultiTaskPINN(n_in=10)
    env = PipelineEnv(pinn=pinn)
    agent = PPOAgent(sdim=10, adim=10)
    optimizer = ClosedLoopOptim(env, agent, pinn)

    initial_state = np.array([620.0, 22.0, 150.0, 20.0, 150.0, 80.0, 9.0, 0.8, 2.0, 1.5])
    captured_state = {}

    original_step = env.step

    def wrapped_step(action):
        captured_state["before_step"] = env.state.copy()
        return original_step(action)

    env.step = wrapped_step
    optimizer.run_refinement(initial_state, max_iters=1)

    assert np.allclose(captured_state["before_step"], initial_state)
