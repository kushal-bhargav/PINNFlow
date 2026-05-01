"""
Feed-Forward Neural Network (FFNN) for ASME B31.3 code stress prediction.

One FFNN per load case (expansion SE and sustained SL trained separately).
Architecture: 2-4 hidden layers, sigmoid activation, Levenberg-Marquardt or Adam.

References:
    Caponetto / Giudice et al. (2022), ANN-Based Optimization of Pressure Piping
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple


class StressPredictorANN(nn.Module):
    """
    Fully-connected ANN for predicting a single ASME B31.3 stress value.

    Inputs:  [H, W, D_m, t_m, delta_T, P_Pa]  (6 features, normalised externally)
    Output:  scalar stress (Pa) — either SE or SL depending on training data

    Architecture:
        Input(6) → [Linear → Sigmoid] × n_hidden → Linear(1)
    """

    def __init__(
        self,
        n_inputs:  int = 6,
        n_hidden:  int = 3,
        n_neurons: int = 50,
        activation: str = "sigmoid",
    ):
        """
        Args:
            n_inputs  : number of input features (default 6)
            n_hidden  : number of hidden layers (2–4 recommended)
            n_neurons : neurons per hidden layer (20–100)
            activation: 'sigmoid' | 'tanh' | 'relu'
        """
        super().__init__()

        act_map = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}
        Act = act_map.get(activation, nn.Sigmoid)

        layers = []
        in_dim = n_inputs
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, n_neurons), Act()]
            in_dim  = n_neurons
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

        # Stores normalisation stats set during training (for inference)
        self.x_mean: Optional[np.ndarray] = None
        self.x_std:  Optional[np.ndarray] = None
        self.y_mean: float = 0.0
        self.y_std:  float = 1.0

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X : (N, n_inputs) normalised input tensor

        Returns:
            y_pred : (N,) predicted normalised stress
        """
        return self.net(X).squeeze(-1)

    def predict_raw(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Predict stress in physical units from un-normalised input.

        Args:
            X_raw : (N, n_inputs) raw (unnormalised) features

        Returns:
            stress_Pa : (N,) predicted stress in Pa
        """
        if self.x_mean is None:
            raise RuntimeError("Call set_normalisation_params() before predict_raw().")
        X_norm = (X_raw - self.x_mean) / (self.x_std + 1e-8)
        X_t    = torch.tensor(X_norm.astype(np.float32))
        with torch.no_grad():
            y_norm = self.forward(X_t).numpy()
        return y_norm * self.y_std + self.y_mean

    def set_normalisation_params(
        self,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: float = 0.0,
        y_std:  float = 1.0,
    ):
        """Store normalisation parameters for predict_raw()."""
        self.x_mean = x_mean.copy()
        self.x_std  = x_std.copy()
        self.y_mean = y_mean
        self.y_std  = y_std

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Levenberg-Marquardt (approximated via Adam + schedule)
# For LM, use scipy or a custom implementation; here we use Adam as drop-in
# ─────────────────────────────────────────────────────────────────────────────

def build_expansion_ann(n_neurons: int = 50, n_hidden: int = 3) -> StressPredictorANN:
    """Build ANN for expansion stress SE prediction."""
    return StressPredictorANN(n_inputs=6, n_hidden=n_hidden,
                               n_neurons=n_neurons, activation="sigmoid")


def build_sustained_ann(n_neurons: int = 50, n_hidden: int = 3) -> StressPredictorANN:
    """Build ANN for sustained stress SL prediction."""
    return StressPredictorANN(n_inputs=6, n_hidden=n_hidden,
                               n_neurons=n_neurons, activation="sigmoid")


# ─────────────────────────────────────────────────────────────────────────────
# Dual-ANN wrapper (expansion + sustained case)
# ─────────────────────────────────────────────────────────────────────────────

class DualStressANN:
    """
    Wrapper holding two ANNs — one for SE prediction, one for SL prediction.
    Both are trained independently (separate load case datasets).

    Usage:
        dual = DualStressANN()
        dual.train(df_train, df_val, ...)
        SE, SL = dual.predict(H=2.0, W=1.5, D=0.219, t=0.0095, dT=150, P=2e6)
    """

    def __init__(self, n_neurons: int = 50, n_hidden: int = 3):
        self.ann_SE = build_expansion_ann(n_neurons, n_hidden)
        self.ann_SL = build_sustained_ann(n_neurons, n_hidden)

    def predict(self, X_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict SE and SL for raw input array.

        Args:
            X_raw : (N, 6) array [H, W, D_m, t_m, delta_T, P_Pa]

        Returns:
            SE_pred, SL_pred : (N,) arrays in Pa
        """
        return self.ann_SE.predict_raw(X_raw), self.ann_SL.predict_raw(X_raw)

    def predict_single(
        self,
        H: float, W: float, D: float, t: float,
        dT: float, P: float,
    ) -> Tuple[float, float]:
        """Convenience wrapper for a single geometry point."""
        X_raw = np.array([[H, W, D, t, dT, P]], dtype=np.float32)
        SE, SL = self.predict(X_raw)
        return float(SE[0]), float(SL[0])
