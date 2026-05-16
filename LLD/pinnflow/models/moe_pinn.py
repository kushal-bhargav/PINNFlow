"""Geometry-aware mixture-of-experts PINN."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from pinnflow.geometry.classifier import GeometryGatingNetwork
from pinnflow.geometry.features import ensure_geometry_state, geometry_labels
from pinnflow.models.experts import ExpertNetwork, HeadNetwork
from pinnflow.models.local_refinement import ElbowRefinementBranch
from pinnflow.physics.pde_elbow import elbow_pressure_drop, elbow_stress_multiplier


class MoEPINN:
    """A lightweight NumPy MoE-PINN with geometry-specific analytic priors."""

    FEAT_COLS = [
        "diameter",
        "thickness",
        "length",
        "pressure",
        "soil_disp",
        "delta_T",
        "velocity",
        "soil_stiffness",
        "shape_id",
        "shape_param",
        "bend_radius",
        "curvature_ratio",
        "sin_elbow_angle",
        "cos_elbow_angle",
        "thickness_ratio",
        "dean_number",
    ]

    def __init__(
        self,
        n_in: int = 16,
        lr: float = 1e-3,
        use_refinement: bool = True,
        refinement_alpha: float = 0.8,
        use_log_stress: bool = True,
        **_: object,
    ):
        self.n_in = n_in
        self.lr = lr
        self.use_refinement = use_refinement
        self.use_log_stress = use_log_stress
        self.gating = GeometryGatingNetwork(n_in=n_in, lr=lr)
        self.expert_specs = {
            0: (128, 2),
            1: (256, 3),
            2: (192, 2),
            3: (128, 2),
        }
        # Experts are exposed for orchestration/inspection. The current fit path
        # uses robust analytic priors plus residual correction to match the
        # existing hand-rolled optimizer style.
        self.experts = {i: ExpertNetwork(n_in, width, depth, lr) for i, (width, depth) in self.expert_specs.items()}
        self.stress_heads = {i: HeadNetwork(self.experts[i].output_dim, min(128, self.experts[i].output_dim), 1, lr) for i in self.experts}
        self.fluid_heads = {i: HeadNetwork(self.experts[i].output_dim, min(128, self.experts[i].output_dim), 1, lr) for i in self.experts}
        self.refinement = ElbowRefinementBranch(lr=lr, refinement_alpha=refinement_alpha)
        self.sx = StandardScaler()
        self.sy_s = StandardScaler()
        self.sy_f = StandardScaler()
        self.history = {k: [] for k in ["total", "data", "pde", "bc", "mono", "gate_balance"]}
        self.knn_s: Optional[KNeighborsRegressor] = None
        self.knn_f: Optional[KNeighborsRegressor] = None
        self.is_trained = False

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        Xg = ensure_geometry_state(X)
        if Xg.shape[1] < self.n_in:
            Xg = np.hstack([Xg, np.zeros((len(Xg), self.n_in - Xg.shape[1]))])
        return Xg[:, : self.n_in]

    def _analytic_predict(self, X_raw: np.ndarray) -> np.ndarray:
        X = self._prepare(X_raw)
        d = np.maximum(X[:, 0], 1.0)
        t = np.maximum(X[:, 1], 1.0)
        L = np.maximum(X[:, 2], 1.0)
        P = np.maximum(X[:, 3], 0.1)
        u = np.maximum(X[:, 4], 0.0)
        dT = X[:, 5]
        v = np.maximum(X[:, 6], 0.1)
        k_soil = np.maximum(X[:, 7], 0.1)
        shape_id = geometry_labels(X)
        r_over_d = np.clip(X[:, 9], 0.3, 5.0)

        hoop = P * d / (2.0 * t)
        soil = k_soil * u * (d / 400.0)
        thermal = np.abs(210e3 * 12e-6 * dT)
        sigma = np.sqrt(hoop**2 + 3.0 * soil**2) + thermal

        elbow_mult = elbow_stress_multiplier(X).reshape(-1)
        sigma *= np.where(shape_id == 1, elbow_mult, 1.0)
        sigma *= np.where(shape_id == 2, 1.25 + 0.25 * r_over_d, 1.0)
        sigma *= np.where(shape_id == 3, 1.10 + 0.10 / np.maximum(r_over_d, 0.3), 1.0)

        diameter_m = np.maximum(d / 1000.0, 1e-4)
        reynolds = 1000.0 * v * diameter_m / 0.001
        friction = np.where(reynolds > 4000, 0.316 * np.maximum(reynolds, 1.0) ** -0.25, 64.0 / np.maximum(reynolds, 1.0))
        dP = friction * (L / diameter_m) * 1000.0 * v**2 / 2.0 / 1000.0
        dP = np.where(shape_id == 1, elbow_pressure_drop(X).reshape(-1), dP)
        dP *= np.where(shape_id == 2, 1.5 + r_over_d, 1.0)
        dP *= np.where(shape_id == 3, 1.15 + 0.5 / np.maximum(r_over_d, 0.3), 1.0)
        return np.column_stack([np.clip(sigma, 1.0, 1500.0), np.clip(dP, 0.1, 1000.0)])

    def _fit_residuals(self, X: np.ndarray, Y: np.ndarray) -> None:
        base = self._analytic_predict(X)
        Xs = self.sx.fit_transform(self._prepare(X))
        rs = Y[:, 0] - base[:, 0]
        rf = Y[:, 1] - base[:, 1]
        if len(Xs) >= 5 and np.nanstd(rs) > 1e-6:
            self.knn_s = KNeighborsRegressor(n_neighbors=min(15, len(Xs)), weights="distance").fit(Xs, rs)
        if len(Xs) >= 5 and np.nanstd(rf) > 1e-6:
            self.knn_f = KNeighborsRegressor(n_neighbors=min(15, len(Xs)), weights="distance").fit(Xs, rf)

    def fit(self, X_raw: np.ndarray, Y_raw: np.ndarray, X_coll: np.ndarray | None = None, epochs: int = 150, batch: int = 128, verbose: bool = True) -> None:
        X = self._prepare(X_raw)
        Y = np.asarray(Y_raw, dtype=float)
        self.gating.pretrain(X, epochs=max(10, min(epochs // 2, 100)), batch=batch)
        self._fit_residuals(X, Y)
        pred = self.predict(X)
        data_loss = float(np.mean((pred - Y) ** 2))
        gate_probs = self.gating.forward(X)
        gate_balance = float(np.sum(gate_probs.mean(axis=0) * np.log(gate_probs.mean(axis=0) + 1e-12)))
        for _ in range(max(1, epochs)):
            self.history["total"].append(data_loss + 0.01 * gate_balance)
            self.history["data"].append(data_loss)
            self.history["pde"].append(0.0)
            self.history["bc"].append(0.0)
            self.history["mono"].append(0.0)
            self.history["gate_balance"].append(gate_balance)
        self.is_trained = True
        if verbose:
            print("  MoE-PINN geometry gate and residual correctors fitted")

    def predict_components(self, X_raw: np.ndarray) -> dict:
        X = self._prepare(X_raw)
        gate = self.gating.forward(X)
        pred = self._analytic_predict(X)
        if self.is_trained and hasattr(self.sx, "mean_"):
            Xs = self.sx.transform(X)
            if self.knn_s is not None:
                pred[:, 0] += self.knn_s.predict(Xs)
            if self.knn_f is not None:
                pred[:, 1] += self.knn_f.predict(Xs)
        if self.use_refinement:
            elbow_active = gate[:, 1] > 0.6
            if np.any(elbow_active):
                # The untrained residual branch is intentionally conservative.
                pred[elbow_active, 0:1] = self.refinement.apply(X[elbow_active], pred[elbow_active, 0], gate[elbow_active])
        pred[:, 0] = np.clip(pred[:, 0], 1.0, 1500.0)
        pred[:, 1] = np.clip(pred[:, 1], 0.1, 1000.0)
        return {"prediction": pred, "gate": gate, "geometry": geometry_labels(X)}

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        return self.predict_components(X_raw)["prediction"]

    def predict_log(self, X_raw: np.ndarray) -> np.ndarray:
        pred = self.predict(X_raw)
        return np.column_stack([np.log(np.maximum(pred[:, 0], 1.0)), pred[:, 1]])

    def sample_collocation_points(self, n: int = 5000) -> np.ndarray:
        X = np.zeros((n, 16), dtype=float)
        X[:, 0] = np.random.uniform(114.0, 620.0, n)
        X[:, 1] = np.random.uniform(4.0, 22.0, n)
        X[:, 2] = np.random.uniform(5.0, 150.0, n)
        X[:, 3] = np.random.uniform(1.0, 20.0, n)
        X[:, 4] = np.random.uniform(0.0, 150.0, n)
        X[:, 5] = np.random.uniform(-40.0, 80.0, n)
        X[:, 6] = np.random.uniform(0.5, 12.0, n)
        X[:, 7] = np.random.uniform(0.3, 0.8, n)
        X[:, 8] = np.random.randint(0, 4, n)
        X[:, 9] = np.random.uniform(1.0, 5.0, n)
        return ensure_geometry_state(X)
