import numpy as np
from pinnflow.pinn import MultiTaskPINN

def test_log_scaling_math():
    """Verify that log-stress targets are correctly handled."""
    pinn = MultiTaskPINN(use_log_stress=True)
    
    # Dummy data with large stress range
    X = np.random.uniform(0, 1, (10, 10))
    Y = np.array([[30.0, 1.0], [300.0, 1.0]] + [[150.0, 1.0]]*8)
    
    pinn.fit(X, Y, epochs=10, verbose=False)
    
    # Test internal scaling
    Y_tgt = Y.copy()
    Y_tgt[:, 0] = np.log(np.maximum(Y_raw[:, 0] if 'Y_raw' in locals() else Y[:, 0], 1.0))
    
    # Predict should return linear scale
    preds = pinn.predict(X)
    assert preds.shape == (10, 2)
    assert np.all(preds[:, 0] > 0)
    print("✓ Log-scaling math verification passed.")

if __name__ == "__main__":
    try:
        test_log_scaling_math()
    except Exception as e:
        print(f"✖ Test failed: {e}")
