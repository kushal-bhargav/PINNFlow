"""
tests/test_sparse_pinn.py
─────────────────────────
Test: PINN trained on N_obs=50 + N_coll=5000 must beat VanillaMLPBaseline
      on stress MAE% when both evaluated on same 100-point test set.
"""
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pinnflow.pinn import MultiTaskPINN
from pinnflow.metrics import VanillaMLPBaseline
from pinnflow.simulator import PhysicsSimulator

def test_pinn_vs_mlp():
    sim = PhysicsSimulator()
    df = sim.generate(1000)
    X = df[MultiTaskPINN.FEAT_COLS].values
    Y = df[["von_mises_stress", "pressure_drop_kPa"]].values

    # Sparse Label set (50)
    idx_l = np.random.choice(len(X), 50, replace=False)
    Xtr, Ytr = X[idx_l], Y[idx_l]
    
    # Dense Collocation (5000)
    X_coll = sim.generate(5000)[MultiTaskPINN.FEAT_COLS].values
    
    # Test set
    Xte, Yte = X[:100], Y[:100]

    print("Training Sparse PINN...")
    pinn = MultiTaskPINN(lr=1e-2)
    pinn.fit(Xtr, Ytr, X_coll=X_coll, epochs=300, verbose=False)
    
    print("Training Vanilla MLP...")
    mlp = VanillaMLPBaseline(lr=1e-2)
    mlp.fit(Xtr, Ytr[:, 0], epochs=300)

    p_preds = pinn.predict(Xte)[:, 0]
    m_preds = mlp.predict(Xte).flatten()

    p_mae = np.mean(np.abs(p_preds - Yte[:, 0]))
    m_mae = np.mean(np.abs(m_preds - Yte[:, 0]))

    print(f"PINN MAE: {p_mae:.2f} | MLP MAE: {m_mae:.2f}")
    assert p_mae < m_mae, "PINN failed to beat MLP in sparse data regime!"
    print("✓ Test Passed: PINN outperformed MLP on sparse data.")

if __name__ == "__main__":
    test_pinn_vs_mlp()
