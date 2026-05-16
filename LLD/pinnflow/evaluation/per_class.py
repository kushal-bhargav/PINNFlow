"""Per-geometry regression diagnostics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

GEOMETRY_LABELS = {0: "straight", 1: "elbow", 2: "t-junction", 3: "reducer"}


def per_class_regression_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, shape_id: np.ndarray) -> dict:
    labels = np.clip(np.rint(shape_id), 0, 3).astype(int)
    out = {}
    for gid, name in GEOMETRY_LABELS.items():
        mask = labels == gid
        if not np.any(mask):
            continue
        class_out = {}
        for idx, metric_name in enumerate(["von_mises_stress", "pressure_drop_kPa"]):
            yt = Y_true[mask, idx]
            yp = Y_pred[mask, idx]
            class_out[metric_name] = {
                "n": int(mask.sum()),
                "MAE": round(float(mean_absolute_error(yt, yp)), 4),
                "R2": round(float(r2_score(yt, yp)) if len(yt) > 1 else 0.0, 4),
            }
        out[name] = class_out
    return out


def hotspot_localization_error(pred_peak_xy: np.ndarray, fem_peak_xy: np.ndarray) -> float:
    pred = np.asarray(pred_peak_xy, dtype=float)
    fem = np.asarray(fem_peak_xy, dtype=float)
    return float(np.linalg.norm(pred - fem, axis=-1).mean())


def robustness_under_noise(model, X: np.ndarray, Y: np.ndarray, noise_levels=(0.05, 0.10, 0.15)) -> dict:
    out = {}
    scale = np.maximum(np.std(X, axis=0, keepdims=True), 1e-6)
    for level in noise_levels:
        Xn = X + np.random.randn(*X.shape) * scale * level
        pred = model.predict(Xn)
        out[f"{int(level * 100)}pct"] = float(mean_absolute_error(Y[:, 0], pred[:, 0]))
    return out
