"""
tests/test_asme_real_data.py
────────────────────────────
Test: Evaluate PINN on real-world ASME B31.3 published cases.
      Expect Mean Pct Error < 15%.
"""
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pinnflow.pinn import MultiTaskPINN
from pinnflow.simulator import PhysicsSimulator
from data.real_data_adapter import ASMECaseAdapter

def test_asme_validation():
    sim = PhysicsSimulator()
    df = sim.generate(1000)
    X, Y = df[MultiTaskPINN.FEAT_COLS].values, df[["von_mises_stress", "pressure_drop_kPa"]].values

    pinn = MultiTaskPINN(lr=1e-2)
    pinn.fit(X, Y, epochs=200, verbose=False)

    adapter = ASMECaseAdapter()
    results = adapter.evaluate_pinn(pinn)
    
    m_err = results["Pct Error (%)"].mean()
    print(f"Mean Pct Error on ASME Cases: {m_err:.2f}%")
    
    assert m_err < 15.0, f"Real-world OOD error too high! ({m_err:.2f}%)"
    print("✓ Test Passed: PINN validated against ASME B31.3 published cases.")

if __name__ == "__main__":
    test_asme_validation()
