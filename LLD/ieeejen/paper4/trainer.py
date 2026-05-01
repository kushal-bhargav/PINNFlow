"""
Training utilities for the ASME B31.3 stress prediction ANN (Paper 4).

Uses Adam optimiser with:
  - MSE loss
  - L2 weight decay regularisation
  - Early stopping on validation loss
  - Learning rate scheduling

References:
    Caponetto / Giudice et al. (2022), ANN-Based Optimization of Pressure Piping
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Optional, Tuple

from .ffnn import StressPredictorANN


# ─────────────────────────────────────────────────────────────────────────────
# Regression metrics
# ─────────────────────────────────────────────────────────────────────────────

def r_squared(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def mean_abs_pct_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean absolute percentage error (%)."""
    return float(np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + 1e-12))) * 100.0)


def max_abs_pct_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Maximum absolute percentage error (%)."""
    return float(np.max(np.abs((y_pred - y_true) / (np.abs(y_true) + 1e-12))) * 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ANNTrainer:
    """
    Trains a StressPredictorANN with Adam optimiser, L2 regularisation,
    early stopping, and LR scheduling.

    Args:
        model          : StressPredictorANN instance
        X_train, y_train : normalised training tensors
        X_val,   y_val   : normalised validation tensors
        lr             : learning rate
        weight_decay   : L2 regularisation coefficient
        max_epochs     : maximum training epochs
        patience       : early stopping patience (epochs)
        batch_size     : mini-batch size (None = full-batch)
        log_every      : logging frequency
        device         : torch device
    """

    def __init__(
        self,
        model: StressPredictorANN,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val:   torch.Tensor,
        y_val:   torch.Tensor,
        lr:            float = 1e-3,
        weight_decay:  float = 1e-4,
        max_epochs:    int   = 5_000,
        patience:      int   = 300,
        batch_size:    Optional[int] = 256,
        log_every:     int   = 200,
        device:        str   = "cpu",
    ):
        self.model    = model.to(device)
        self.X_train  = X_train.to(device)
        self.y_train  = y_train.to(device)
        self.X_val    = X_val.to(device)
        self.y_val    = y_val.to(device)
        self.max_epochs = max_epochs
        self.patience   = patience
        self.batch_size = batch_size
        self.log_every  = log_every
        self.device     = device

        self.optim = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, patience=100, factor=0.5, min_lr=1e-6
        )
        self.criterion = nn.MSELoss()

        self.history: Dict = {
            "epoch": [], "train_loss": [], "val_loss": [], "lr": []
        }
        self.best_val_loss = float("inf")
        self.best_state    = None
        self.early_stop_count = 0

    def _get_batches(self):
        """Yield mini-batches of (X, y) from training set."""
        N  = self.X_train.shape[0]
        if self.batch_size is None or self.batch_size >= N:
            yield self.X_train, self.y_train
            return
        perm = torch.randperm(N, device=self.device)
        for i in range(0, N, self.batch_size):
            idx = perm[i : i + self.batch_size]
            yield self.X_train[idx], self.y_train[idx]

    def train(self) -> Dict:
        """
        Run full training loop.

        Returns:
            history : dict with epoch, train_loss, val_loss, lr lists
        """
        print(f"\nANN Training  |  {self.model.count_parameters()} params  |  "
              f"max_epochs={self.max_epochs}  |  patience={self.patience}")
        print("─" * 60)
        t0 = time.time()

        for epoch in range(1, self.max_epochs + 1):
            # ── Train ─────────────────────────────────────────────────────────
            self.model.train()
            batch_losses = []
            for Xb, yb in self._get_batches():
                self.optim.zero_grad()
                pred  = self.model(Xb)
                loss  = self.criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optim.step()
                batch_losses.append(loss.item())
            train_loss = float(np.mean(batch_losses))

            # ── Validate ──────────────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_pred  = self.model(self.X_val)
                val_loss  = self.criterion(val_pred, self.y_val).item()

            self.scheduler.step(val_loss)
            lr_now = self.optim.param_groups[0]["lr"]

            # ── Early stopping ─────────────────────────────────────────────────
            if val_loss < self.best_val_loss - 1e-8:
                self.best_val_loss   = val_loss
                self.best_state      = {k: v.clone() for k, v in
                                        self.model.state_dict().items()}
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1

            # ── Logging ───────────────────────────────────────────────────────
            if epoch % self.log_every == 0 or epoch == 1:
                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["lr"].append(lr_now)
                print(f"  [Epoch {epoch:>5d}] train={train_loss:.4e}  "
                      f"val={val_loss:.4e}  lr={lr_now:.2e}  "
                      f"patience={self.early_stop_count}/{self.patience}")

            if self.early_stop_count >= self.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val={self.best_val_loss:.4e})")
                break

        # Restore best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        elapsed = time.time() - t0
        print(f"Training complete in {elapsed:.1f} s | "
              f"best val_loss={self.best_val_loss:.4e}")
        return self.history

    def evaluate(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        y_std: float = 1.0,
        y_mean: float = 0.0,
    ) -> Dict:
        """
        Evaluate trained model on test set and return regression metrics.

        Args:
            X_test, y_test : test tensors (normalised)
            y_std, y_mean  : de-normalisation parameters for physical-unit errors

        Returns:
            metrics dict with R², mean%, max%, MSE
        """
        self.model.eval()
        with torch.no_grad():
            y_pred_norm = self.model(X_test.to(self.device)).cpu().numpy()
        y_test_norm = y_test.cpu().numpy()

        # De-normalise to physical units (Pa)
        y_pred_phys = y_pred_norm * y_std + y_mean
        y_test_phys = y_test_norm * y_std + y_mean

        mse  = float(np.mean((y_pred_phys - y_test_phys) ** 2))
        rmse = float(np.sqrt(mse))
        R2   = r_squared(y_pred_phys, y_test_phys)
        mape = mean_abs_pct_error(y_pred_phys, y_test_phys)
        maxe = max_abs_pct_error(y_pred_phys, y_test_phys)

        metrics = {"R2": R2, "MSE": mse, "RMSE": rmse,
                   "MAPE_%": mape, "MaxAPE_%": maxe}
        print(f"\nTest metrics:  R²={R2:.5f}  MAPE={mape:.3f}%  MaxAPE={maxe:.3f}%")
        return metrics
