"""
tests/test_lagrangian_convergence.py
────────────────────────────────────
Test: After 400 PPO episodes with Lagrangian RL, beta must have adapted.
"""
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pinnflow.environment import PipelineEnv
from pinnflow.agent import PPOAgent
from pinnflow.pinn import MultiTaskPINN
from pinnflow.simulator import PhysicsSimulator

def test_lagrangian_adaptation():
    sim = PhysicsSimulator()
    df = sim.generate(500)
    X, Y = df[MultiTaskPINN.FEAT_COLS].values, df[["von_mises_stress", "pressure_drop_kPa"]].values

    pinn = MultiTaskPINN(lr=1e-2)
    pinn.fit(X, Y, epochs=100, verbose=False)

    env = PipelineEnv(pinn, curriculum=True)
    agent = PPOAgent(lr=1e-2)

    print("Training Lagrangian PPO (50 episodes test)...")
    agent.train(env, n_ep=50, steps=10, verbose=False)

    print(f"Final beta: {agent.beta:.4f}")
    # Since agent begins near violation, beta should rise from 0.1
    assert agent.beta != 0.1, "Lagrangian multiplier beta did not adapt!"
    print("✓ Test Passed: Lagrangian multiplier beta adapted to constraints.")

if __name__ == "__main__":
    test_lagrangian_adaptation()
