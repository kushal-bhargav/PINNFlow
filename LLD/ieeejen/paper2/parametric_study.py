"""
Parametric study: systematic variation of PGD magnitude δ and crossing angle β.

Maps the influence of (δ, β) on:
  - Peak tensile strain at outer fibre
  - Peak compressive strain at inner fibre
  - Location of maximum bending moment

Outputs a 2-D response surface over the δ-β parameter space.

References:
    Chen et al. (2025), PINNs + Transfer Learning for Pipe Responses
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import torch.nn as nn


@torch.no_grad()
def parametric_sweep(
    model: nn.Module,
    L: float = 300.0,
    delta_values: np.ndarray = None,
    beta_values: np.ndarray = None,
    n_x: int = 300,
    D: float = 0.508,
    t: float = 0.0095,
    device: str = "cpu",
) -> Dict:
    """
    Evaluate the trained PINN over a grid of (δ, β) values and
    extract peak strains and moment location along the pipe.

    Args:
        model        : trained ElasticPIPENN
        L            : pipeline length (m)
        delta_values : 1D array of ground displacement magnitudes (m)
        beta_values  : 1D array of crossing angles (degrees)
        n_x          : number of spatial points along pipe
        D, t         : pipe outer diameter and wall thickness
        device       : torch device

    Returns:
        results dict with keys:
            delta_grid, beta_grid : meshgrid arrays (n_delta, n_beta)
            eps_tension   : peak tensile strain (n_delta, n_beta)
            eps_compression : peak compressive strain (n_delta, n_beta)
            x_max_moment  : x-location of maximum moment (n_delta, n_beta)
    """
    if delta_values is None:
        delta_values = np.linspace(0.05, 0.5, 20)
    if beta_values is None:
        beta_values = np.linspace(0.0, 90.0, 19)

    model.eval()
    x_grid = torch.linspace(0.0, L, n_x, device=device)
    z_outer =  D / 2.0
    z_inner = -D / 2.0

    n_d = len(delta_values)
    n_b = len(beta_values)

    eps_t   = np.zeros((n_d, n_b))
    eps_c   = np.zeros((n_d, n_b))
    x_Mmax  = np.zeros((n_d, n_b))

    for i, delta in enumerate(delta_values):
        for j, beta in enumerate(beta_values):
            # Build X: (n_x, 3) = [x, δ, β]
            d_col = torch.full((n_x, 1), delta, device=device)
            b_col = torch.full((n_x, 1), beta,  device=device)
            X = torch.cat([x_grid.unsqueeze(1), d_col, b_col], dim=1)
            X.requires_grad_(True)

            out = model(X)
            u_hat = out[:, 0]
            w_hat = out[:, 1]

            # Derivatives via autograd
            g_u = torch.autograd.grad(u_hat, X, grad_outputs=torch.ones_like(u_hat),
                                      create_graph=True)[0]
            u_x = g_u[:, 0]
            g_w = torch.autograd.grad(w_hat, X, grad_outputs=torch.ones_like(w_hat),
                                      create_graph=True)[0]
            w_x = g_w[:, 0]
            g_wx = torch.autograd.grad(w_x, X, grad_outputs=torch.ones_like(w_x),
                                       create_graph=True)[0]
            w_xx = g_wx[:, 0]

            # Strain at extreme fibres (elastic, no initial strain for simplicity)
            eps_outer = (u_x + 0.5 * w_x ** 2 - z_outer * w_xx).detach().cpu().numpy()
            eps_inner = (u_x + 0.5 * w_x ** 2 - z_inner * w_xx).detach().cpu().numpy()

            # Bending moment M = EI * w_xx (elastic)
            M_arr = w_xx.detach().cpu().numpy()

            eps_t[i, j]  = float(eps_outer.max())
            eps_c[i, j]  = float(eps_inner.min())
            x_Mmax[i, j] = float(x_grid.cpu().numpy()[np.argmax(np.abs(M_arr))])

    delta_grid, beta_grid = np.meshgrid(delta_values, beta_values, indexing="ij")

    return {
        "delta_grid":      delta_grid,
        "beta_grid":       beta_grid,
        "eps_tension":     eps_t,
        "eps_compression": eps_c,
        "x_max_moment":    x_Mmax,
    }


def print_parametric_table(results: Dict, n_delta: int = 5, n_beta: int = 5):
    """Print a summary table of peak strains for a subset of δ-β combinations."""
    d_idx = np.linspace(0, results["delta_grid"].shape[0]-1, n_delta, dtype=int)
    b_idx = np.linspace(0, results["delta_grid"].shape[1]-1, n_beta, dtype=int)

    deltas = results["delta_grid"][d_idx, 0]
    betas  = results["beta_grid"][0, b_idx]

    print(f"\nPeak tensile strain (%) — δ rows × β columns")
    header = "δ\\β " + "  ".join(f"{b:>6.1f}°" for b in betas)
    print(header)
    for i, idx_d in enumerate(d_idx):
        row = f"{deltas[i]:.3f}m  " + "  ".join(
            f"{results['eps_tension'][idx_d, b_idx[j]]*100:>7.3f}" for j in range(n_beta)
        )
        print(row)
