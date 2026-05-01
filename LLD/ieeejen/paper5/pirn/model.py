"""
Physics-Informed Recurrent Network (PIRN) for gas pipeline state estimation.

Core idea: replace learnable RNN weight matrices with physics-structured
transition matrices A(θ) and B(θ) parameterised by physical parameters θ.

    Standard RNN:   h_k = tanh(W_h * h_{k-1} + W_x * x_k + b)
    PIRN cell:      x_{k+1} = A(θ) * x_k + B(θ) * u_k

The matrices A(θ) and B(θ) are assembled from physical parameters θ = {λ_i, Z_i}
and are differentiable w.r.t. θ (enabling gradient-based identification).

References:
    Wang / Liu et al. (2025), PIRN (arXiv:2502.07230)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# PIRN Cell (single time-step physics update)
# ─────────────────────────────────────────────────────────────────────────────

class PIRNCell(nn.Module):
    """
    Single-step PIRN recurrent cell for one gas pipeline network.

    Implements:
        x_{k+1} = A(θ) * x_k + B(θ) * u_k + ε_k
        y_k     = C * x_k + η_k

    Where A(θ), B(θ) come from the GasNetworkStateSpace module.

    Args:
        network_ss : GasNetworkStateSpace instance (provides A, B, C)
    """

    def __init__(self, network_ss):
        super().__init__()
        self.network_ss = network_ss

    def forward(
        self,
        x_k: torch.Tensor,
        u_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One-step state transition.

        Args:
            x_k : (n_nodes,) or (batch, n_nodes) state at time k
            u_k : (n_comp,) or (batch, n_comp) control input

        Returns:
            x_next : next state
            y_k    : measurement prediction
        """
        if x_k.dim() == 1:
            x_next, y_k = self.network_ss(x_k, u_k)
        else:
            # Batch processing
            x_next_list, y_list = [], []
            for b in range(x_k.shape[0]):
                xn, yn = self.network_ss(x_k[b], u_k[b])
                x_next_list.append(xn)
                y_list.append(yn)
            x_next = torch.stack(x_next_list)
            y_k    = torch.stack(y_list)
        return x_next, y_k

    def get_friction_factors(self) -> Dict[int, torch.Tensor]:
        """Return current friction factor estimates for all pipes."""
        return self.network_ss.get_learnable_friction_factors()


# ─────────────────────────────────────────────────────────────────────────────
# Full PIRN (unrolled over time)
# ─────────────────────────────────────────────────────────────────────────────

class PIRNModel(nn.Module):
    """
    Full Physics-Informed Recurrent Network: unrolls the PIRN cell over T time steps.

    Args:
        pirn_cell    : PIRNCell instance
        tau          : TBPTT (Truncated Backprop Through Time) window length
                       (how many steps to backpropagate; 0 = full unrolling)
    """

    def __init__(self, pirn_cell: PIRNCell, tau: int = 20):
        super().__init__()
        self.cell = pirn_cell
        self.tau  = tau

    def forward(
        self,
        x0: torch.Tensor,
        u_seq: torch.Tensor,
        T_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rollout the PIRN over T_steps time steps.

        Args:
            x0      : (n_nodes,) initial state
            u_seq   : (T_steps, n_comp) control input sequence
            T_steps : number of rollout steps

        Returns:
            x_seq : (T_steps+1, n_nodes) state trajectory (including x0)
            y_seq : (T_steps, n_sensors) measurement predictions
        """
        x_seq = [x0]
        y_seq = []

        x_k = x0
        for k in range(T_steps):
            u_k   = u_seq[k]
            x_next, y_k = self.cell(x_k, u_k)

            x_seq.append(x_next)
            y_seq.append(y_k)
            x_k = x_next.detach() if (self.tau > 0 and k % self.tau == 0) else x_next

        x_seq_t = torch.stack(x_seq, dim=0)   # (T+1, n_nodes)
        y_seq_t = torch.stack(y_seq, dim=0)   # (T, n_sensors)

        return x_seq_t, y_seq_t

    def rollout_tbptt(
        self,
        x0: torch.Tensor,
        u_seq: torch.Tensor,
        y_measured: torch.Tensor,
        loss_fn: callable,
    ) -> torch.Tensor:
        """
        TBPTT (Truncated Backpropagation Through Time) rollout with loss accumulation.

        Divides the time series into windows of length tau, performs forward pass
        and backward pass within each window, accumulating gradients.

        Args:
            x0          : (n_nodes,) initial state
            u_seq       : (T, n_comp) control inputs
            y_measured  : (T, n_sensors) measurements
            loss_fn     : callable(y_pred, y_meas) → scalar loss

        Returns:
            total_loss : accumulated loss over all windows (scalar)
        """
        T_total = u_seq.shape[0]
        x_k     = x0.detach()    # detach initial state (no grad from outside window)
        total_loss = torch.tensor(0.0, dtype=x0.dtype, device=x0.device)

        for window_start in range(0, T_total, self.tau):
            window_end = min(window_start + self.tau, T_total)

            y_preds = []
            for k in range(window_start, window_end):
                u_k   = u_seq[k]
                x_next, y_k = self.cell(x_k, u_k)
                y_preds.append(y_k)
                x_k = x_next

            y_pred_window = torch.stack(y_preds, dim=0)
            y_meas_window = y_measured[window_start:window_end]

            window_loss = loss_fn(y_pred_window, y_meas_window)
            total_loss  = total_loss + window_loss

            x_k = x_k.detach()   # detach at window boundary → truncated BPTT

        return total_loss

    def get_identified_parameters(self) -> Dict[str, float]:
        """
        Return dictionary of identified physical parameters after training.
        """
        params = {}
        for i, (idx, lam) in enumerate(self.cell.get_friction_factors().items()):
            params[f"lambda_pipe_{idx}"] = lam.item()
        return params
