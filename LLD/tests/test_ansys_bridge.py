"""
tests/test_ansys_bridge.py
──────────────────────────
Verification: Generate the ANSYS APDL script and verify content structure.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pinnflow.ansys_bridge import generate_apdl_script

def test_ansys_export():
    # Random design: D=500, t=15, L=30, P=8, ...
    params = np.array([500, 15, 30, 8, 0, 20, 2.0, 0.5])
    
    print("Testing ANSYS APDL Export...")
    path = generate_apdl_script(params, filename="test_ansys_script.txt")
    
    assert os.path.exists(path), f"Failed to generate APDL script at {path}"
    
    with open(path, "r") as f:
        content = f.read()
        
    assert "/PREP7" in content, "Missing /PREP7 command"
    assert "SOLID185" in content, "Missing SOLID185 element definition"
    assert "CYLIND" in content, "Missing CYLIND geometry definition"
    assert "SOLVE" in content, "Missing SOLVE command"
    
    print(f"✓ Test Passed: ANSYS APDL Script valid and generated at {path}")

if __name__ == "__main__":
    test_ansys_export()
