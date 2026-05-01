"""
Limit state function for pipeline strain-based failure.

    g(ξ) = ε_allow - ε_l_max(x; δ, c, φ, γ, H)

Failure if g(ξ) ≤ 0.

Supports:
  - Tensile failure criterion
  - Compressive failure criterion
  - Combined (most critical of both)

The max strain is evaluated using the trained PINN surrogate via fast forward pass.

References:
    Haghighat et al. (2025), PINN-RA for Buried Pipeline Reliability Analysis
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Strain allowables (ASME / CSA)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ALLOWABLES = {
    "tensile_strain_limit":     0.03,    # 3% tensile — ASME B31.8 / CSA Z662
    "compressive_strain_limit": 0.005,   # 0.5% compressive — more conservative
}


# ─────────────────────────────────────────────────────────────────────────────
# Max strain extraction from PINN
# ─────────────────────────────────────────────────────────────────────────────

def build_pinn_query_points(
    xi_samples: torch.Tensor,
    x_grid: torch.Tensor,
    pipe_params: dict,
) -> torch.Tensor:
    """
    Construct PINN input tensor X from parameter samples and a spatial grid.

    For each sample ξ^i = [c, φ, γ, δ, H], evaluate the PINN at N_x spatial
    locations x and find the maximum strain along the pipe.

    Args:
        xi_samples  : (N_mc, 5) samples [c, φ, γ, δ, H]
        x_grid      : (N_x,) spatial evaluation points
        pipe_params : dict with 'P', 'delta_T', 'D', 't', 'beta_deg', etc.

    Returns:
        X_query : (N_mc * N_x, 6) input tensor [x, δ, c, φ, γ, H]
                  (note: column order matches PINN input convention)
    """
    N_mc = xi_samples.shape[0]
    N_x  = x_grid.shape[0]
    device = xi_samples.device

    # Repeat each sample N_x times (for all spatial positions)
    # xi_samples: (N_mc, 5) → (N_mc * N_x, 5)
    xi_rep = xi_samples.repeat_interleave(N_x, dim=0)  # (N_mc*N_x, 5)

    # Tile x_grid N_mc times
    x_rep = x_grid.repeat(N_mc).unsqueeze(1)             # (N_mc*N_x, 1)

    # PINN input order: [x, δ, c, φ, γ, H]
    # xi_samples columns:     [c, φ, γ, δ, H]  → indices 0,1,2,3,4
    c_col     = xi_rep[:, 0:1]    # c
    phi_col   = xi_rep[:, 1:2]    # φ
    gamma_col = xi_rep[:, 2:3]    # γ
    delta_col = xi_rep[:, 3:4]    # δ
    H_col     = xi_rep[:, 4:5]    # H

    # Assemble: [x, δ, c, φ, γ, H]
    X_query = torch.cat([x_rep, delta_col, c_col, phi_col, gamma_col, H_col], dim=1)
    return X_query


@torch.no_grad()
def evaluate_max_strain_pinn(
    network: nn.Module,
    xi_samples: torch.Tensor,
    x_grid: torch.Tensor,
    cross_section_integrator,
    pipe_params: dict,
    batch_size: int = 10_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate maximum tensile and compressive longitudinal strains along the pipe
    for each parameter sample ξ^i, using the trained PINN as surrogate.

    This is called inside MCS; the PINN forward pass replaces expensive FEA.

    Args:
        network                 : trained PINNNetwork
        xi_samples              : (N_mc, 5) samples [c, φ, γ, δ, H]
        x_grid                  : (N_x,) spatial grid for evaluation
        cross_section_integrator: CrossSectionIntegrator (for strain recovery)
        pipe_params             : dict with pipe geometry and loading parameters
        batch_size              : evaluate in chunks to avoid OOM

    Returns:
        eps_max_tension     : (N_mc,) maximum tensile strain per sample
        eps_max_compression : (N_mc,) most negative compressive strain per sample
    """
    from ..material_model import eps_GM_full, initial_strain

    N_mc   = xi_samples.shape[0]
    N_x    = x_grid.shape[0]
    device = xi_samples.device

    network.eval()

    D       = pipe_params.get("D", 0.914)
    t       = pipe_params.get("t", 0.0127)
    P       = pipe_params.get("P", 10e6)
    delta_T = pipe_params.get("delta_T", 0.0)
    E       = pipe_params.get("E", 207e9)
    nu      = pipe_params.get("nu", 0.3)
    alpha_T = pipe_params.get("alpha_T", 1.17e-5)

    # Extreme fibre positions
    z_outer = D / 2.0
    z_inner = -D / 2.0

    eps_tension_all     = torch.zeros(N_mc, device=device)
    eps_compression_all = torch.zeros(N_mc, device=device)

    # Process in mini-batches
    for i in range(0, N_mc, batch_size):
        xi_batch = xi_samples[i : i + batch_size]
        n_batch  = xi_batch.shape[0]

        X_query = build_pinn_query_points(xi_batch, x_grid, pipe_params).to(device)
        X_query.requires_grad_(True)

        out    = network(X_query)
        u_hat  = out[:, 0]
        w_hat  = out[:, 1]

        # Spatial derivatives via autograd
        grad_u = torch.autograd.grad(u_hat, X_query,
                                     grad_outputs=torch.ones_like(u_hat),
                                     create_graph=False)[0]
        grad_w = torch.autograd.grad(w_hat, X_query,
                                     grad_outputs=torch.ones_like(w_hat),
                                     create_graph=False)[0]
        u_x   = grad_u[:, 0]
        w_x   = grad_w[:, 0]

        grad_wx = torch.autograd.grad(w_x, X_query,
                                      grad_outputs=torch.ones_like(w_x),
                                      create_graph=False)[0]
        w_xx = grad_wx[:, 0]

        # Strain at extreme fibres
        eps_init = initial_strain(P, D, t, E, nu, alpha_T, delta_T)
        eps_outer = eps_init + eps_GM_full(u_x, w_x, w_xx, z_outer)    # (n_batch*N_x,)
        eps_inner = eps_init + eps_GM_full(u_x, w_x, w_xx, z_inner)

        # Reshape and find max/min along spatial axis
        eps_outer = eps_outer.reshape(n_batch, N_x)   # (n_batch, N_x)
        eps_inner = eps_inner.reshape(n_batch, N_x)

        eps_tension_all[i : i + n_batch]     = eps_outer.max(dim=1).values
        eps_compression_all[i : i + n_batch] = eps_inner.min(dim=1).values

    return eps_tension_all, eps_compression_all


# ─────────────────────────────────────────────────────────────────────────────
# Limit state function
# ─────────────────────────────────────────────────────────────────────────────

def limit_state_tensile(
    eps_max_tension: torch.Tensor,
    eps_allow_t: float = DEFAULT_ALLOWABLES["tensile_strain_limit"],
) -> torch.Tensor:
    """
    g(ξ) = ε_allow_t - ε_max_tension   (failure if g ≤ 0)

    Args:
        eps_max_tension : (N_mc,) maximum tensile strain per sample
        eps_allow_t     : allowable tensile strain limit

    Returns:
        g_t : (N_mc,) limit state values
    """
    return eps_allow_t - eps_max_tension


def limit_state_compressive(
    eps_max_compression: torch.Tensor,
    eps_allow_c: float = DEFAULT_ALLOWABLES["compressive_strain_limit"],
) -> torch.Tensor:
    """
    g(ξ) = ε_allow_c - |ε_max_compression|   (failure if g ≤ 0)

    Args:
        eps_max_compression : (N_mc,) most negative (compressive) strain
        eps_allow_c         : allowable compressive strain magnitude

    Returns:
        g_c : (N_mc,) limit state values
    """
    return eps_allow_c - torch.abs(eps_max_compression)


def limit_state_combined(
    eps_max_tension: torch.Tensor,
    eps_max_compression: torch.Tensor,
    eps_allow_t: float = DEFAULT_ALLOWABLES["tensile_strain_limit"],
    eps_allow_c: float = DEFAULT_ALLOWABLES["compressive_strain_limit"],
) -> torch.Tensor:
    """
    Combined limit state: failure if EITHER tensile OR compressive limit exceeded.

        g(ξ) = min(g_t, g_c)

    Args:
        eps_max_tension     : (N_mc,)
        eps_max_compression : (N_mc,)

    Returns:
        g : (N_mc,) combined limit state (min of tensile and compressive)
    """
    g_t = limit_state_tensile(eps_max_tension, eps_allow_t)
    g_c = limit_state_compressive(eps_max_compression, eps_allow_c)
    return torch.minimum(g_t, g_c)
