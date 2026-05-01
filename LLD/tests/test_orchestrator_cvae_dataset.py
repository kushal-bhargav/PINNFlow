from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinnflow.orchestrator_v2 import UnifiedOrchestrator


def _fake_simulator_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "diameter": rng.uniform(114.3, 610.0, n),
            "thickness": rng.uniform(4.0, 18.0, n),
            "length": rng.uniform(5.0, 150.0, n),
            "pressure": rng.uniform(1.5, 20.0, n),
            "soil_disp": rng.uniform(0.0, 150.0, n),
            "delta_T": rng.uniform(-40.0, 80.0, n),
            "velocity": rng.uniform(0.5, 9.0, n),
            "soil_stiffness": rng.uniform(0.3, 0.8, n),
            "von_mises_stress": rng.uniform(50.0, 220.0, n),
            "pressure_drop_kPa": rng.uniform(0.1, 20.0, n),
            "failure_prob": rng.uniform(0.01, 0.2, n),
            "fatigue_life": rng.uniform(1e5, 1e7, n),
        }
    )


def test_orchestrator_builds_conditioned_cvae_dataset(monkeypatch):
    orch = UnifiedOrchestrator(train_cvae=False)

    fake_frame = _fake_simulator_frame()
    monkeypatch.setattr(orch.physics, "generate", lambda n: fake_frame)

    captured: dict[str, object] = {}

    def fake_fit(X_raw, conditions=None, epochs=None, batch=None, verbose=None):
        captured["X"] = np.asarray(X_raw, dtype=float)
        captured["C"] = np.asarray(conditions, dtype=float)
        captured["epochs"] = epochs
        captured["batch"] = batch
        captured["verbose"] = verbose

    monkeypatch.setattr(orch.vae, "fit", fake_fit)

    orch._train_cvae_on_scenarios()

    assert captured["X"].shape[1] == 10
    assert captured["C"].shape[1] == 4
    assert captured["X"].shape[0] == captured["C"].shape[0]
    assert orch.cvae_training_summary["trained"] is True
    assert orch.cvae_training_summary["condition_dim"] == 4
    assert captured["epochs"] == orch.cvae_epochs
    assert captured["batch"] == 32
