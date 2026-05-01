"""
tests/test_diffusion_csr.py
───────────────────────────
Test: DDPM with PINN guidance must produce higher CSR than DDPM without guidance.
"""
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pinnflow.diffusion import PipelineDDPM
from pinnflow.pinn import MultiTaskPINN
from pinnflow.simulator import PhysicsSimulator

def test_diffusion_guidance():
    sim = PhysicsSimulator()
    df = sim.generate(500)
    X, Y = df[MultiTaskPINN.FEAT_COLS].values, df[["von_mises_stress", "pressure_drop_kPa"]].values

    pinn = MultiTaskPINN(lr=1e-2)
    pinn.fit(X, Y, epochs=100, verbose=False)

    print("Training DDPM...")
    diff = PipelineDDPM(dim=8, steps=50, pinn=pinn)
    diff.fit(X, epochs=100)

    print("Sampling with vs without guidance...")
    s_unguided = diff.sample(50, guidance_scale=0.0)
    s_guided   = diff.sample(50, guidance_scale=5.0)

    csr_u = np.mean(pinn.predict(s_unguided)[:, 0] < 200)
    csr_g = np.mean(pinn.predict(s_guided)[:, 0] < 200)

    print(f"CSR Unguided: {csr_u:.2%} | CSR Guided: {csr_g:.2%}")
    assert csr_g >= csr_u, "Guidance failed to improve CSR!"
    print("✓ Test Passed: PINN guidance improved diffusion feasibility.")

if __name__ == "__main__":
    test_diffusion_guidance()
