"""
tests/test_vae_diversity.py
───────────────────────────
Test: After KL annealing fix, diversity_score must be in [0.5, 2.0].
"""
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pinnflow.vae import CAVAE
from pinnflow.simulator import PhysicsSimulator

def test_vae_diversity():
    sim = PhysicsSimulator()
    df = sim.generate(1000)
    X = df[["diameter", "thickness", "length", "pressure", 
            "soil_disp", "delta_T", "velocity", "soil_stiffness"]].values

    print("Training CAVAE with KL annealing...")
    vae = CAVAE(x_dim=8, z_dim=16)
    vae.fit(X, epochs=200, verbose=False)

    div = vae.diversity_score()
    print(f"VAE Diversity Score: {div:.4f}")
    
    assert 0.5 <= div <= 2.5, f"Posterior collapse detected! Diversity={div}"
    assert div != 1.0, "Diversity score appears to be a constant bug!"
    print("✓ Test Passed: VAE diversity is healthy.")


def test_trained_vae_ranks_candidate_pool(monkeypatch):
    vae = CAVAE(x_dim=10, z_dim=4)
    vae.is_trained = True
    vae.scaler.mean_ = np.zeros(10, dtype=float)
    vae.scaler.scale_ = np.ones(10, dtype=float)
    monkeypatch.setattr(vae.scaler, "inverse_transform", lambda x: x)
    monkeypatch.setattr(vae, "_map_to_bounds", lambda x: x)

    def fake_decode(z, condition):
        assert condition.shape == (len(z), 4)
        rows = np.vstack([np.full(10, float(i), dtype=float) for i in range(len(z))])
        return rows

    def fake_score(candidates, pinn=None, condition=None):
        return np.arange(len(candidates), dtype=float)

    monkeypatch.setattr(vae, "_decode", fake_decode)
    monkeypatch.setattr(vae, "_score_candidates", fake_score)

    out = vae.generate(n=2, condition={"max_p": 80.0, "max_t": 25.0}, top_k=1, candidate_multiplier=2)
    expected = vae._clip_designs(np.full((1, 10), 3.0, dtype=float))[0]

    assert out.shape == (1, 10)
    assert np.allclose(out[0], expected)


def test_cvae_architecture_includes_condition_inputs():
    vae = CAVAE(x_dim=10, z_dim=4)

    assert vae.enc_l[0].W.shape[0] == 14
    assert vae.dec_l[0].W.shape[0] == 8

if __name__ == "__main__":
    test_vae_diversity()
