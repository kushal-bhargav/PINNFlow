"""
PINN training loop for the buried pipeline parametric PINN.

Training strategy (two-phase):
  Phase 1 — Adam optimiser   : fast convergence to a reasonable basin
  Phase 2 — L-BFGS optimiser : precise local minimisation

References:
    Haghighat et al. (2025), PINN-RA for Buried Pipeline Reliability Analysis
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
from typing import Dict, List, Optional, Callable, Tuple

from .collocation import DomainBounds, resample_collocation, sample_boundary_points


# ─────────────────────────────────────────────────────────────────────────────
# Training history container
# ─────────────────────────────────────────────────────────────────────────────

class TrainingHistory:
    """Records loss values and timing across training epochs."""

    def __init__(self):
        self.epochs:   List[int]   = []
        self.L_total:  List[float] = []
        self.L_axial:  List[float] = []
        self.L_lateral: List[float] = []
        self.L_BC:     List[float] = []
        self.wall_time: List[float] = []
        self._t0 = time.time()

    def log(self, epoch: int, losses: Dict[str, float]):
        self.epochs.append(epoch)
        self.L_total.append(losses.get("L_total", float("nan")))
        self.L_axial.append(losses.get("L_axial", float("nan")))
        self.L_lateral.append(losses.get("L_lateral", float("nan")))
        self.L_BC.append(losses.get("L_BC", float("nan")))
        self.wall_time.append(time.time() - self._t0)

    def to_dict(self) -> Dict:
        return {
            "epochs":     self.epochs,
            "L_total":    self.L_total,
            "L_axial":    self.L_axial,
            "L_lateral":  self.L_lateral,
            "L_BC":       self.L_BC,
            "wall_time":  self.wall_time,
        }

    def save_csv(self, path: str):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch","L_total","L_axial","L_lateral","L_BC","wall_time"])
            w.writeheader()
            for i, ep in enumerate(self.epochs):
                w.writerow({
                    "epoch": ep,
                    "L_total": self.L_total[i],
                    "L_axial": self.L_axial[i],
                    "L_lateral": self.L_lateral[i],
                    "L_BC": self.L_BC[i],
                    "wall_time": self.wall_time[i],
                })


# ─────────────────────────────────────────────────────────────────────────────
# Main Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PINNTrainer:
    """
    Two-phase PINN trainer: Adam → L-BFGS.

    Args:
        network          : PINNNetwork model
        residual_fn      : callable(network, X_col) → (R1, R2)
        bounds           : DomainBounds for collocation sampling
        n_col            : number of interior collocation points
        n_bc             : number of boundary collocation points
        sampling_method  : 'lhs' or 'uniform'
        resample_every   : resample collocation points every N epochs (0 = no resample)
        adam_lr          : Adam learning rate
        adam_epochs      : number of Adam epochs (Phase 1)
        lbfgs_max_iter   : max L-BFGS iterations (Phase 2)
        loss_weights     : dict {'w1', 'w2', 'w_bc'} — loss term weights
        log_every        : print frequency (epochs)
        device           : 'cpu' or 'cuda'
        checkpoint_dir   : directory to save model checkpoints (None = no saving)
        pipe_params      : dict passed to residual_fn (pressure, temp, geometry)
    """

    def __init__(
        self,
        network: nn.Module,
        residual_fn: Callable,
        bounds: DomainBounds,
        n_col:      int   = 10_000,
        n_bc:       int   = 500,
        sampling_method: str = "lhs",
        resample_every: int  = 0,
        adam_lr:    float = 1e-3,
        adam_epochs: int  = 20_000,
        lbfgs_max_iter: int = 5_000,
        loss_weights: Optional[Dict[str, float]] = None,
        log_every:  int   = 500,
        device:     str   = "cpu",
        checkpoint_dir: Optional[str] = None,
        pipe_params: Optional[Dict] = None,
    ):
        self.network       = network.to(device)
        self.residual_fn   = residual_fn
        self.bounds        = bounds
        self.n_col         = n_col
        self.n_bc          = n_bc
        self.sampling_method = sampling_method
        self.resample_every = resample_every
        self.adam_epochs   = adam_epochs
        self.lbfgs_max_iter = lbfgs_max_iter
        self.log_every     = log_every
        self.device        = device
        self.checkpoint_dir = checkpoint_dir
        self.pipe_params   = pipe_params or {}
        self.history       = TrainingHistory()

        lw = loss_weights or {}
        self.w1   = lw.get("w1",   1.0)
        self.w2   = lw.get("w2",   1.0)
        self.w_bc = lw.get("w_bc", 10.0)

        self.adam   = torch.optim.Adam(network.parameters(), lr=adam_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.adam, gamma=0.9999)

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Sampling helpers ─────────────────────────────────────────────────────

    def _sample_col(self, seed=None):
        return resample_collocation(
            self.bounds, self.n_col, method=self.sampling_method,
            device=self.device, seed=seed
        )

    def _sample_bc(self, seed=None):
        return sample_boundary_points(
            self.bounds, self.n_bc, device=self.device, seed=seed
        )

    # ── Single forward + loss computation ────────────────────────────────────

    def _compute_loss(
        self,
        X_col: torch.Tensor,
        X_bc: torch.Tensor,
        u_bc_t: torch.Tensor,
        w_bc_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        from .loss import total_loss

        R1, R2 = self.residual_fn(self.network, X_col, self.pipe_params)
        loss, losses = total_loss(
            R1, R2,
            self.network, X_bc, u_bc_t, w_bc_t,
            w1=self.w1, w2=self.w2, w_bc=self.w_bc,
        )
        return loss, losses

    # ── Phase 1: Adam ─────────────────────────────────────────────────────────

    def train_adam(self) -> None:
        """Run Adam optimisation phase."""
        print(f"\n{'='*60}")
        print(f"Phase 1: Adam  |  {self.adam_epochs} epochs  |  device={self.device}")
        print(f"{'='*60}")

        X_col, (X_bc, u_t, w_t) = self._sample_col(seed=0), self._sample_bc(seed=1)

        for epoch in range(1, self.adam_epochs + 1):
            # Optionally resample collocation points
            if self.resample_every > 0 and epoch % self.resample_every == 0:
                X_col = self._sample_col()
                X_bc, u_t, w_t = self._sample_bc()

            self.adam.zero_grad()
            loss, losses = self._compute_loss(X_col, X_bc, u_t, w_t)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.adam.step()
            self.scheduler.step()

            if epoch % self.log_every == 0 or epoch == 1:
                self.history.log(epoch, losses)
                lr = self.adam.param_groups[0]["lr"]
                print(f"  [Adam {epoch:>6d}] total={losses['L_total']:.3e}  "
                      f"axial={losses['L_axial']:.3e}  "
                      f"lat={losses['L_lateral']:.3e}  "
                      f"BC={losses['L_BC']:.3e}  lr={lr:.2e}")

            if self.checkpoint_dir and epoch % (self.adam_epochs // 5) == 0:
                self._save_checkpoint(f"adam_epoch_{epoch}.pt")

        print("Phase 1 complete.")

    # ── Phase 2: L-BFGS ──────────────────────────────────────────────────────

    def train_lbfgs(self) -> None:
        """Run L-BFGS fine-tuning phase (full-batch)."""
        print(f"\n{'='*60}")
        print(f"Phase 2: L-BFGS  |  max_iter={self.lbfgs_max_iter}  |  device={self.device}")
        print(f"{'='*60}")

        # Fixed collocation points for L-BFGS (full-batch, no resampling)
        X_col           = self._sample_col(seed=99)
        X_bc, u_t, w_t  = self._sample_bc(seed=100)

        lbfgs = torch.optim.LBFGS(
            self.network.parameters(),
            max_iter=self.lbfgs_max_iter,
            max_eval=self.lbfgs_max_iter * 2,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        iter_count = [0]

        def closure():
            lbfgs.zero_grad()
            loss, losses = self._compute_loss(X_col, X_bc, u_t, w_t)
            loss.backward()
            iter_count[0] += 1
            if iter_count[0] % max(1, self.lbfgs_max_iter // 20) == 0:
                offset = self.adam_epochs + iter_count[0]
                self.history.log(offset, losses)
                print(f"  [LBFGS {iter_count[0]:>5d}] total={losses['L_total']:.3e}  "
                      f"axial={losses['L_axial']:.3e}  "
                      f"lat={losses['L_lateral']:.3e}  "
                      f"BC={losses['L_BC']:.3e}")
            return loss

        lbfgs.step(closure)
        print("Phase 2 complete.")

    # ── Full training pipeline ────────────────────────────────────────────────

    def train(self) -> TrainingHistory:
        """Run full two-phase training (Adam → L-BFGS)."""
        t_start = time.time()
        self.train_adam()
        self.train_lbfgs()
        elapsed = time.time() - t_start
        print(f"\nTotal training time: {elapsed:.1f} s ({elapsed/60:.1f} min)")
        if self.checkpoint_dir:
            self._save_checkpoint("final_model.pt")
        return self.history

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "model_state": self.network.state_dict(),
            "history":     self.history.to_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {path}")
