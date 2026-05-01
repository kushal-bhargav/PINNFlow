"""
Dual-network architecture for dynamic load estimation (inverse PINN).

Two networks trained simultaneously:
  Network 1 (Response net): (x, t) → ŵ(x,t) [and optionally ψ̂(x,t)]
  Network 2 (Load net):     (x, t) → f̂(x,t)

The response network is constrained by:
  - PDE residual loss (beam dynamics)
  - Data loss at sparse sensor locations
  - Initial and boundary condition losses

The load network is implicitly constrained through the PDE residual,
which relates f to the space-time derivatives of w.

References:
    Patel et al. (2025), PINNs for Dynamic Load Estimation in Pipework (IJFMR)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Base FC network (shared architecture for both networks)
# ─────────────────────────────────────────────────────────────────────────────

class SpaceTimeNet(nn.Module):
    """
    Fully-connected network mapping (x, t) → scalar output.

    Args:
        n_hidden  : number of hidden layers
        n_neurons : neurons per hidden layer
        n_outputs : number of outputs (1 for w or f; 2 for Timoshenko w+ψ)
        activation: 'tanh' (default) or 'sin' (for periodic signals)
    """

    def __init__(
        self,
        n_hidden:  int = 5,
        n_neurons: int = 50,
        n_outputs: int = 1,
        activation: str = "tanh",
        t_lb: float = 0.0,
        t_ub: float = 1.0,
        x_lb: float = 0.0,
        x_ub: float = 5.0,
    ):
        super().__init__()

        lb = torch.tensor([x_lb, t_lb], dtype=torch.float32)
        ub = torch.tensor([x_ub, t_ub], dtype=torch.float32)
        self.register_buffer("lb", lb)
        self.register_buffer("ub", ub)

        act_cls = nn.Tanh if activation == "tanh" else nn.Sigmoid
        layers  = []
        in_dim  = 2
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, n_neurons), act_cls()]
            in_dim  = n_neurons
        layers.append(nn.Linear(in_dim, n_outputs))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X : (N, 2) tensor [x, t]
        Returns:
            out : (N, n_outputs)
        """
        X_n = 2.0 * (X - self.lb) / (self.ub - self.lb + 1e-8) - 1.0
        return self.net(X_n)


# ─────────────────────────────────────────────────────────────────────────────
# Dual-network wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DualNetwork(nn.Module):
    """
    Combines response network and load network into a single module.

    Attributes:
        response_net : SpaceTimeNet predicting w(x,t)
        load_net     : SpaceTimeNet predicting f(x,t)
    """

    def __init__(
        self,
        pipe_length: float = 5.0,
        T_total: float     = 1.0,
        n_hidden_r: int    = 5,
        n_neurons_r: int   = 50,
        n_hidden_f: int    = 4,
        n_neurons_f: int   = 30,
        activation: str    = "tanh",
    ):
        super().__init__()

        self.response_net = SpaceTimeNet(
            n_hidden=n_hidden_r, n_neurons=n_neurons_r, n_outputs=1,
            activation=activation, x_lb=0.0, x_ub=pipe_length,
            t_lb=0.0, t_ub=T_total,
        )
        self.load_net = SpaceTimeNet(
            n_hidden=n_hidden_f, n_neurons=n_neurons_f, n_outputs=1,
            activation=activation, x_lb=0.0, x_ub=pipe_length,
            t_lb=0.0, t_ub=T_total,
        )

    def predict_response(self, X: torch.Tensor) -> torch.Tensor:
        """ŵ(x,t) prediction, shape (N,)"""
        return self.response_net(X)[:, 0]

    def predict_load(self, X: torch.Tensor) -> torch.Tensor:
        """f̂(x,t) prediction, shape (N,)"""
        return self.load_net(X)[:, 0]

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (ŵ, f̂) both shape (N,)"""
        return self.predict_response(X), self.predict_load(X)

    def count_parameters(self) -> Dict[str, int]:
        n_r = sum(p.numel() for p in self.response_net.parameters() if p.requires_grad)
        n_f = sum(p.numel() for p in self.load_net.parameters() if p.requires_grad)
        return {"response_net": n_r, "load_net": n_f, "total": n_r + n_f}


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions for inverse problem
# ─────────────────────────────────────────────────────────────────────────────

def data_loss(
    model: DualNetwork,
    X_sensor: torch.Tensor,
    w_measured: torch.Tensor,
) -> torch.Tensor:
    """
    ℒ_data = (1/N_s) * Σ ||ŵ(x_i, t_i) - w_meas(x_i, t_i)||²

    Args:
        model      : DualNetwork
        X_sensor   : (N_s, 2) sensor locations [x, t]
        w_measured : (N_s,) measured displacement values

    Returns:
        scalar data loss
    """
    w_pred = model.predict_response(X_sensor)
    return torch.mean((w_pred - w_measured) ** 2)


def pde_loss_eb(
    model: DualNetwork,
    X_col: torch.Tensor,
    E: float, I: float, rho: float, A: float,
) -> torch.Tensor:
    """
    ℒ_PDE for Euler-Bernoulli beam (see beam_pde.py for derivation).

    Inline computation to avoid circular imports.
    """
    from .beam_pde import eb_pde_residual
    R = eb_pde_residual(model.response_net, model.load_net, X_col, E, I, rho, A)
    return torch.mean(R ** 2)


def ic_loss(
    model: DualNetwork,
    X_ic: torch.Tensor,
    w0: torch.Tensor,
    wdot0: torch.Tensor,
) -> torch.Tensor:
    """
    ℒ_IC = (1/N_IC) * Σ ||ŵ(x_k, 0) - w₀(x_k)||² + ||∂ŵ/∂t(x_k, 0) - ẇ₀(x_k)||²

    Args:
        X_ic  : (N_IC, 2) points at t=0, requires_grad=True
        w0    : (N_IC,) initial displacement
        wdot0 : (N_IC,) initial velocity
    """
    if not X_ic.requires_grad:
        X_ic = X_ic.detach().requires_grad_(True)

    w_pred = model.predict_response(X_ic)
    wt     = torch.autograd.grad(w_pred, X_ic,
                                  grad_outputs=torch.ones_like(w_pred),
                                  create_graph=True)[0][:, 1]
    return torch.mean((w_pred - w0) ** 2) + torch.mean((wt - wdot0) ** 2)


def bc_loss_dynamic(
    model: DualNetwork,
    X_bc: torch.Tensor,
    bc_type: str = "simply_supported",
) -> torch.Tensor:
    """
    ℒ_BC at pipe end boundaries.

    Supported types:
      'simply_supported' : w=0, M=EI*w_xx=0 at both ends
      'clamped'          : w=0, w_x=0 at both ends
      'free_free'        : M=0, V=0 at both ends
    """
    if not X_bc.requires_grad:
        X_bc = X_bc.detach().requires_grad_(True)

    w_pred = model.predict_response(X_bc)
    loss   = torch.mean(w_pred ** 2)

    if bc_type in ("simply_supported", "clamped"):
        # Penalise w=0
        pass  # already included above

    if bc_type == "clamped":
        # Also penalise ∂w/∂x = 0
        grad_w = torch.autograd.grad(w_pred, X_bc,
                                      grad_outputs=torch.ones_like(w_pred),
                                      create_graph=True)[0]
        loss   = loss + torch.mean(grad_w[:, 0] ** 2)

    return loss


def total_loss_inverse(
    model: DualNetwork,
    X_col: torch.Tensor,
    X_sensor: torch.Tensor,
    w_measured: torch.Tensor,
    X_ic: torch.Tensor,
    w0: torch.Tensor,
    wdot0: torch.Tensor,
    X_bc: torch.Tensor,
    beam_params: dict,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Weighted total inverse loss for dynamic load identification.

        ℒ = λ_data*ℒ_data + λ_PDE*ℒ_PDE + λ_IC*ℒ_IC + λ_BC*ℒ_BC

    Args:
        weights : dict with keys 'lambda_data', 'lambda_pde', 'lambda_ic', 'lambda_bc'
        beam_params : dict with E, I, rho, A, (G, k_s for Timoshenko)

    Returns:
        total, losses_dict
    """
    w = weights or {"lambda_data": 1.0, "lambda_pde": 1.0,
                    "lambda_ic": 1.0, "lambda_bc": 1.0}

    E, I, rho, A = (beam_params["E"], beam_params["I"],
                    beam_params["rho"], beam_params["A"])

    L_data = data_loss(model, X_sensor, w_measured)
    L_pde  = pde_loss_eb(model, X_col, E, I, rho, A)
    L_ic   = ic_loss(model, X_ic, w0, wdot0)
    L_bc   = bc_loss_dynamic(model, X_bc)

    total  = (w["lambda_data"] * L_data +
              w["lambda_pde"]  * L_pde  +
              w["lambda_ic"]   * L_ic   +
              w["lambda_bc"]   * L_bc)

    losses = {
        "L_data":  L_data.item(),
        "L_PDE":   L_pde.item(),
        "L_IC":    L_ic.item(),
        "L_BC":    L_bc.item(),
        "L_total": total.item(),
    }
    return total, losses
