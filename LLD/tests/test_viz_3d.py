"""
tests/test_viz_3d.py
────────────────────
Verification: Generate the 3D pipe model and check if HTML file exists.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pinnflow.pinn import MultiTaskPINN
from pinnflow.visualization_3d import render_pipe_3d

def test_3d_viz():
    # Mock PINN
    class MockPINN:
        def predict(self, x):
            return np.array([[180.0, 50.0]]) # Stress, dP
    
    pinn = MockPINN()
    
    # Random design: D=400, t=10, L=20, P=5, SoilDisp=10, ...
    params = np.array([400, 10, 20, 5, 10, 20, 2.0, 0.5])
    
    print("Testing 3D Visualization...")
    path = render_pipe_3d(params, pinn, filename="test_pipe_3d.html")
    
    assert os.path.exists(path), f"Failed to generate 3D HTML at {path}"
    print(f"✓ Test Passed: 3D Visualization generated at {path}")

if __name__ == "__main__":
    test_3d_viz()
