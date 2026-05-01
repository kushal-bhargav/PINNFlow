import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinnflow import PipelineEnv, PPOAgent, MultiTaskPINN, ClosedLoopOptim, DeliverableGenerator


def test_phase3():
    print("Testing Closed-Loop Optimization & Deliverables...")

    pinn = MultiTaskPINN(n_in=10)
    env = PipelineEnv(pinn=pinn)
    agent = PPOAgent(sdim=10, adim=10)

    optimizer = ClosedLoopOptim(env, agent, pinn)
    initial_state = env.reset()
    result = optimizer.run_refinement(initial_state, max_iters=3)
    print(f"[OK] Closed-Loop OK. Iterations: {result['iterations']}")
    print(f"   Final Sigma: {result['metrics']['sigma']:.2f} MPa")

    gen = DeliverableGenerator("results/test_deliverables")
    paths = gen.generate_all("TEST-001", result["final_state"], result["metrics"])
    print("[OK] Deliverables OK. Paths generated:")
    for key, value in paths.items():
        print(f"   {key}: {value}")
        assert os.path.exists(value)
    assert "TEST-001" in paths["bom"]


if __name__ == "__main__":
    test_phase3()
