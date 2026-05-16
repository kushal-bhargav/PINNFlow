"""Uncertainty calibration diagnostics."""
from __future__ import annotations

import numpy as np


def expected_calibration_error(Y_true: np.ndarray, ci_lo: np.ndarray, ci_hi: np.ndarray, confidence: float = 0.95) -> float:
    y = np.asarray(Y_true, dtype=float).reshape(-1)
    lo = np.asarray(ci_lo, dtype=float).reshape(-1)
    hi = np.asarray(ci_hi, dtype=float).reshape(-1)
    covered = np.mean((y >= lo) & (y <= hi))
    return float(abs(confidence - covered))
