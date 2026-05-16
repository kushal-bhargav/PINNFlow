"""Out-of-distribution topology evaluation helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error


def topology_ood_report(model, datasets: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    report = {}
    for name, (X, Y) in datasets.items():
        pred = model.predict(X)
        report[name] = {
            "stress_mae": float(mean_absolute_error(Y[:, 0], pred[:, 0])),
            "pressure_mae": float(mean_absolute_error(Y[:, 1], pred[:, 1])),
            "n": int(len(X)),
        }
    return report
