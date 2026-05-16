"""Per-class residual histograms."""
from __future__ import annotations

import numpy as np

from pinnflow.evaluation.per_class import GEOMETRY_LABELS


def plot_per_class_error_histogram(ax, Y_true: np.ndarray, Y_pred: np.ndarray, shape_id: np.ndarray):
    labels = np.clip(np.rint(shape_id), 0, 3).astype(int)
    residual = Y_pred[:, 0] - Y_true[:, 0]
    for gid, name in GEOMETRY_LABELS.items():
        mask = labels == gid
        if np.any(mask):
            ax.hist(residual[mask], bins=20, alpha=0.45, label=name)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_title("Stress residuals by class")
    ax.set_xlabel("Predicted - actual MPa")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    return ax
