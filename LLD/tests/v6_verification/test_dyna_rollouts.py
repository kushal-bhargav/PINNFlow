import numpy as np
from pinnflow.agent import LagrangianPPOAgent
from pinnflow.environment import PipelineEnv
from pinnflow.pinn import MultiTaskPINN

def test_dyna_mechanism():
    pinn = MultiTaskPINN()
    env = PipelineEnv(pinn=pinn)
    agent = LagrangianPPOAgent(sdim=10, adim=10)
    
    # Verify that synthetic rollouts can be generated
    s, a, r, v = agent.generate_pinn_rollouts(pinn, env, n_rollouts=10)
    assert len(s) == 10
    assert len(a) == 10
    assert len(r) == 10
    print("✓ Dyna-style synthetic rollout generation passed.")

if __name__ == "__main__":
    test_dyna_mechanism()
