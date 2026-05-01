"""
Loss function components for the parametric PINN (Paper 1).

Three loss components:
    ℒ₁   — axial PDE residual:    dN/dx + h(Ug - u) = 0
    ℒ₂   — lateral PDE residual:  d²M/dx² - d/dx[N·dw/dx] - q(Wg - w) = 0
    ℒ_BC — boundary condition loss (zero disp/rotation at pipe ends)

Total:  ℒ = ℒ₁ + ℒ₂ + ℒ_BC

References:
    Haghighat et al. (2025), PINN-RA for Buried Pipeline Reliability Analysis
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss terms
# ─────────────────────────────────────────────────────────────────────────────

def pde_loss_axial(R1: torch.Tensor) -> torch.Tensor:
    """
    ℒ₁ = (1/N_f) * Σ ||dN/dx + h(Ug - u)||²

    Args:
        R1 : axial PDE residual (N_col,)

    Returns:
        scalar loss
    """
    return torch.mean(R1 ** 2)


def pde_loss_lateral(R2: torch.Tensor) -> torch.Tensor:
    """
    ℒ₂ = (1/N_f) * Σ ||d²M/dx² - d/dx[N·dw/dx] - q(Wg - w)||²

    Args:
        R2 : lateral PDE residual (N_col,)

    Returns:
        scalar loss
    """
    return torch.mean(R2 ** 2)


def bc_loss(
    network: nn.Module,
    X_bc: torch.Tensor,
    u_bc_target: torch.Tensor,
    w_bc_target: torch.Tensor,
    include_rotation: bool = True,
) -> torch.Tensor:
    """
    ℒ_BC = (1/N_u) * Σ [ ||û(X_bc) - u_bc||² + ||ŵ(X_bc) - w_bc||²
                         + (||ûₓ(X_bc)||² + ||ŵₓ(X_bc)||²)  if include_rotation ]

    Boundary conditions: zero displacements and zero rotation (slope) at pipe ends
    far from the ground movement zone. Both x=0 and x=L ends.

    Args:
        network          : PINN model
        X_bc             : boundary collocation points (N_bc, 6), requires_grad=True
        u_bc_target      : target axial displacement at BCs (N_bc,), typically zeros
        w_bc_target      : target lateral displacement at BCs (N_bc,), typically zeros
        include_rotation : if True, also penalise dw/dx ≠ 0 at ends

    Returns:
        scalar BC loss
    """
    if not X_bc.requires_grad:
        X_bc = X_bc.detach().requires_grad_(True)

    out = network(X_bc)
    u_hat = out[:, 0]
    w_hat = out[:, 1]

    loss = torch.mean((u_hat - u_bc_target) ** 2) + \
           torch.mean((w_hat - w_bc_target) ** 2)

    if include_rotation:
        # Compute ∂w/∂x at boundary points
        grad_w = torch.autograd.grad(
            w_hat, X_bc,
            grad_outputs=torch.ones_like(w_hat),
            create_graph=True, retain_graph=True
        )[0]
        w_x_bc = grad_w[:, 0]
        loss = loss + torch.mean(w_x_bc ** 2)

        # Also penalise du/dx at boundaries (zero axial rotation)
        grad_u = torch.autograd.grad(
            u_hat, X_bc,
            grad_outputs=torch.ones_like(u_hat),
            create_graph=True, retain_graph=True
        )[0]
        u_x_bc = grad_u[:, 0]
        loss = loss + torch.mean(u_x_bc ** 2)

    return loss


def total_loss(
    R1: torch.Tensor,
    R2: torch.Tensor,
    network: nn.Module,
    X_bc: torch.Tensor,
    u_bc_target: torch.Tensor,
    w_bc_target: torch.Tensor,
    w1: float = 1.0,
    w2: float = 1.0,
    w_bc: float = 10.0,
    include_rotation: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Weighted total PINN loss.

        ℒ_total = w1*ℒ₁ + w2*ℒ₂ + w_bc*ℒ_BC

    Args:
        R1, R2          : PDE residuals from compute_all_residuals()
        network         : PINN model (needed for BC loss autodiff)
        X_bc            : boundary points
        u_bc_target     : target u at boundaries
        w_bc_target     : target w at boundaries
        w1, w2, w_bc    : loss weights
        include_rotation: whether to penalise slope at boundaries

    Returns:
        total  : scalar total loss (graph retained for backprop)
        losses : dict of individual loss values (detached floats for logging)
    """
    L1  = pde_loss_axial(R1)
    L2  = pde_loss_lateral(R2)
    Lbc = bc_loss(network, X_bc, u_bc_target, w_bc_target, include_rotation)

    total = w1 * L1 + w2 * L2 + w_bc * Lbc

    losses = {
        "L_axial"   : L1.item(),
        "L_lateral" : L2.item(),
        "L_BC"      : Lbc.item(),
        "L_total"   : total.item(),
    }
    return total, losses


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Loss Weighting (NTK-inspired gradient balancing)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveLossWeights:
    """
    Adaptive loss weights based on gradient magnitude balancing.

    After each epoch, rescale weights so all loss terms contribute
    equally to the total gradient norm. This prevents one term from
    dominating early training.

    Usage:
        alw = AdaptiveLossWeights(alpha=0.9)
        ...
        # after computing individual losses:
        w1, w2, wbc = alw.update(model, [L1, L2, Lbc])
    """

    def __init__(self, alpha: float = 0.9, n_terms: int = 3):
        """
        Args:
            alpha  : exponential moving average factor for weight smoothing
            n_terms: number of loss terms
        """
        self.alpha = alpha
        self.weights = torch.ones(n_terms)

    def update(
        self,
        model: nn.Module,
        losses: list,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """
        Compute gradient norms for each loss term and rescale weights.

        Args:
            model         : PINN model
            losses        : list of scalar loss tensors [L1, L2, Lbc, ...]
            retain_graph  : whether to retain computation graph

        Returns:
            updated weights tensor (n_terms,)
        """
        grad_norms = []
        for loss in losses:
            grads = torch.autograd.grad(
                loss, model.parameters(),
                retain_graph=retain_graph, allow_unused=True
            )
            norm = sum(g.norm() ** 2 for g in grads if g is not None) ** 0.5
            grad_norms.append(norm.item() + 1e-12)

        grad_norms = torch.tensor(grad_norms)
        mean_norm = grad_norms.mean()
        new_weights = mean_norm / grad_norms

        # EMA smoothing
        self.weights = self.alpha * self.weights + (1 - self.alpha) * new_weights
        return self.weights
