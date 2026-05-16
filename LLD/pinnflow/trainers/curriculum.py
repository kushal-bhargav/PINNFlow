"""Four-stage curriculum scheduler for MoE-PINN experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CurriculumStage:
    name: str
    geometry_ids: tuple[int, ...]
    epochs: int
    loss_balancer: str
    min_r_over_d: float | None = None
    use_refinement: bool = False


class MoECurriculumTrainer:
    """Run staged class-filtered training against a MoEPINN-like model."""

    DEFAULT_STAGES = (
        CurriculumStage("straight", (0,), 200, "gradnorm"),
        CurriculumStage("mild_elbows", (0, 1), 150, "gradnorm", min_r_over_d=3.0),
        CurriculumStage("all_elbows", (1,), 200, "softadapt", min_r_over_d=1.0, use_refinement=True),
        CurriculumStage("mixed_topology", (0, 1, 2, 3), 150, "uncertainty", use_refinement=True),
    )

    def __init__(self, stages: tuple[CurriculumStage, ...] | None = None):
        self.stages = stages or self.DEFAULT_STAGES
        self.history: list[dict] = []

    def _mask(self, X: np.ndarray, stage: CurriculumStage) -> np.ndarray:
        shape = np.clip(np.rint(X[:, 8]), 0, 3).astype(int)
        mask = np.isin(shape, stage.geometry_ids)
        if stage.min_r_over_d is not None:
            mask &= X[:, 9] >= stage.min_r_over_d
        return mask

    def fit(self, model, X: np.ndarray, Y: np.ndarray, batch: int = 128, verbose: bool = True):
        for stage in self.stages:
            mask = self._mask(X, stage)
            if not np.any(mask):
                continue
            if hasattr(model, "use_refinement"):
                model.use_refinement = stage.use_refinement
            if verbose:
                print(f"  Curriculum stage {stage.name}: {int(mask.sum())} samples, {stage.epochs} epochs")
            model.fit(X[mask], Y[mask], epochs=stage.epochs, batch=batch, verbose=False)
            self.history.append({"stage": stage.name, "n": int(mask.sum()), "epochs": stage.epochs})
        return model
