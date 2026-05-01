"""
Training loop for the PIRN gas pipeline parameter identification model.

Key training techniques:
  1. Warm start  : initialise θ from engineering estimates before gradient descent
  2. Normalisation: normalise pressures and flows to [0, 1]
  3. Gradient clipping: clip gradient norms to prevent exploding gradients in RNN
  4. TBPTT: truncated backpropagation through time (window τ)
  5. Cosine annealing LR schedule

References:
    Wang / Liu et al. (2025), PIRN (arXiv:2502.07230)
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Optional, List

from .model import PIRNModel


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

class SeriesNormalizer:
    """
    Min-max normaliser for pressure/flow time series.
    Fitted on training data; applied identically to test data.
    """
    def __init__(self):
        self.mins = None
        self.maxs = None

    def fit(self, data: np.ndarray) -> "SeriesNormalizer":
        """data : (T, n_channels)"""
        self.mins = data.min(axis=0, keepdims=True)
        self.maxs = data.max(axis=0, keepdims=True)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mins) / (self.maxs - self.mins + 1e-12)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * (self.maxs - self.mins + 1e-12) + self.mins


# ─────────────────────────────────────────────────────────────────────────────
# PIRN Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PIRNTrainer:
    """
    Trains a PIRNModel to identify physical parameters {λ_i} from sparse
    terminal pressure/flow measurements.

    Args:
        model           : PIRNModel instance
        y_measured_train: (T_train, n_sensors) normalised measurement array
        u_train         : (T_train, n_comp) control inputs (normalised)
        x0_train        : (n_nodes,) initial state estimate
        y_measured_test : (T_test, n_sensors) test measurements
        u_test          : (T_test, n_comp) test control inputs
        x0_test         : (n_nodes,) test initial state
        theta_true      : dict of true parameter values (for evaluation only)
        n_epochs        : number of training epochs
        lr              : initial Adam learning rate
        grad_clip       : gradient norm clipping threshold
        tau             : TBPTT window length
        log_every       : logging frequency (epochs)
        device          : torch device
    """

    def __init__(
        self,
        model: PIRNModel,
        y_measured_train: torch.Tensor,
        u_train:          torch.Tensor,
        x0_train:         torch.Tensor,
        y_measured_test:  torch.Tensor,
        u_test:           torch.Tensor,
        x0_test:          torch.Tensor,
        theta_true:       Optional[Dict[str, float]] = None,
        n_epochs:         int   = 500,
        lr:               float = 1e-3,
        grad_clip:        float = 1.0,
        tau:              int   = 20,
        log_every:        int   = 50,
        device:           str   = "cpu",
    ):
        self.model       = model.to(device)
        self.y_train     = y_measured_train.to(device)
        self.u_train     = u_train.to(device)
        self.x0_train    = x0_train.to(device)
        self.y_test      = y_measured_test.to(device)
        self.u_test      = u_test.to(device)
        self.x0_test     = x0_test.to(device)
        self.theta_true  = theta_true or {}
        self.n_epochs    = n_epochs
        self.lr          = lr
        self.grad_clip   = grad_clip
        self.tau         = tau
        self.log_every   = log_every
        self.device      = device

        self.optim = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=n_epochs, eta_min=lr * 0.01
        )

        self.history: Dict = {
            "epoch": [], "train_loss": [], "test_loss": [],
            "lambda_errors": [], "lr": [],
        }

    @staticmethod
    def _mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_pred - y_true) ** 2)

    def _parameter_errors(self) -> Dict[str, float]:
        """Compare current parameter estimates to ground truth."""
        errors = {}
        for name, true_val in self.theta_true.items():
            identified = self.model.get_identified_parameters()
            if name in identified:
                err = abs(identified[name] - true_val) / (abs(true_val) + 1e-12)
                errors[name] = err
        return errors

    def warm_start_check(self) -> None:
        """
        Verify that warm-start initialisation gives reasonable initial loss.
        Log initial parameter values and test loss before any gradient steps.
        """
        print("\nWarm-start check (before training):")
        params = self.model.get_identified_parameters()
        for name, val in list(params.items())[:5]:
            true_val = self.theta_true.get(name, None)
            true_str = f"  (true={true_val:.5f})" if true_val else ""
            print(f"  {name}: {val:.5f}{true_str}")

        T_test = self.u_test.shape[0]
        with torch.no_grad():
            _, y_pred = self.model(self.x0_test.double(),
                                   self.u_test.double(), T_test)
        init_loss = self._mse_loss(y_pred.float(), self.y_test).item()
        print(f"  Initial test MSE: {init_loss:.4e}")

    def train(self) -> Dict:
        """
        Run full PIRN training loop.

        Returns:
            history dict with training curves
        """
        self.warm_start_check()
        t0 = time.time()

        print(f"\nPIRN Training  |  {self.n_epochs} epochs  |  "
              f"TBPTT tau={self.tau}  |  device={self.device}")
        print("-" * 65)

        T_train = self.u_train.shape[0]

        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            self.optim.zero_grad()

            # TBPTT forward + backward
            train_loss = self.model.rollout_tbptt(
                self.x0_train.double(),
                self.u_train.double(),
                self.y_train.double(),
                self._mse_loss,
            )

            train_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
            self.optim.step()
            self.scheduler.step()

            # ── Enforce physical constraints on learned parameters ────────────
            # Friction factors must be positive
            with torch.no_grad():
                for pipe in self.model.cell.network_ss.pipe_models:
                    pipe.lam.clamp_(min=1e-4, max=0.1)
                    pipe.Z.clamp_(min=0.5, max=1.2)

            lr_now = self.optim.param_groups[0]["lr"]

            if epoch % self.log_every == 0 or epoch == 1:
                # Compute test loss
                self.model.eval()
                T_test = self.u_test.shape[0]
                with torch.no_grad():
                    _, y_pred_test = self.model(
                        self.x0_test.double(), self.u_test.double(), T_test
                    )
                test_loss = self._mse_loss(
                    y_pred_test.float(), self.y_test
                ).item()

                param_errs = self._parameter_errors()
                mean_lam_err = (np.mean(list(param_errs.values()))
                                if param_errs else float("nan"))

                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(train_loss.item())
                self.history["test_loss"].append(test_loss)
                self.history["lambda_errors"].append(mean_lam_err)
                self.history["lr"].append(lr_now)

                print(f"  [Epoch {epoch:>5d}] train={train_loss.item():.4e}  "
                      f"test={test_loss:.4e}  "
                      f"mean_lam_err={mean_lam_err*100:.2f}%  lr={lr_now:.2e}")

        elapsed = time.time() - t0
        print(f"\nTraining complete in {elapsed:.1f} s ({elapsed/60:.1f} min)")
        self.history["elapsed_s"] = elapsed

        # Final parameter summary
        print("\nIdentified parameters:")
        for name, val in self.model.get_identified_parameters().items():
            true_val = self.theta_true.get(name, None)
            if true_val:
                err = abs(val - true_val) / abs(true_val) * 100
                print(f"  {name}: {val:.5f}  (true={true_val:.5f}, err={err:.2f}%)")
            else:
                print(f"  {name}: {val:.5f}")

        return self.history
