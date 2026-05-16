"""Reproducible experiment runner for MoE-PINN studies."""
from __future__ import annotations

import numpy as np

from pinnflow.evaluation.per_class import per_class_regression_metrics
from pinnflow.models.moe_pinn import MoEPINN
from pinnflow.trainers.curriculum import MoECurriculumTrainer


class ExperimentRunner:
    """Minimal train/evaluate loop used by redesign experiments."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def run(self, X_train, Y_train, X_test, Y_test, use_curriculum: bool = True) -> dict:
        np.random.seed(self.seed)
        model = MoEPINN()
        if use_curriculum:
            MoECurriculumTrainer().fit(model, X_train, Y_train, verbose=False)
        else:
            model.fit(X_train, Y_train, verbose=False)
        pred = model.predict(X_test)
        return {"model": model, "pred": pred, "per_class": per_class_regression_metrics(Y_test, pred, X_test[:, 8])}
