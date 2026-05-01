"""
Numerical integration of internal forces (N) and moments (M) over the pipe
cross-section, discretised into (n_r x n_theta) patches.

    N = Σ σ_ij * A_ij
    M = -Σ z_ij * σ_ij * A_ij

References:
    Haghighat et al. (2025), PINN-RA for Buried Pipeline Reliability Analysis
"""

import torch
import numpy as np
from .material_model import (
    biaxial_yield_stresses,
    menegotto_pinto_stress,
    eps_GM_full,
    initial_strain,
    hoop_stress,
    X65_DEFAULTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Section Patch Discretisation
# ─────────────────────────────────────────────────────────────────────────────

def build_cross_section_patches(D, t, n_r=4, n_theta=16):
    """
    Discretise the annular pipe cross-section into (n_r x n_theta) patches.

    Patches are defined by mid-point radius r_ij and angle theta_ij.
    Returns patch centroids (y_ij, z_ij) and areas A_ij.

    Convention:
        z  = vertical axis (positive upward)
        y  = horizontal axis
        Bending is in the x-z plane; z is the fibre coordinate for bending.

    Args:
        D      : outer diameter (m)
        t      : wall thickness (m)
        n_r    : number of radial divisions
        n_theta: number of circumferential divisions

    Returns:
        z_ij : fibre height from neutral axis for each patch (n_r * n_theta,)
        A_ij : patch area (n_r * n_theta,)
        r_ij : patch mid-radius (n_r * n_theta,)
    """
    r_outer = D / 2.0
    r_inner = r_outer - t

    r_edges = np.linspace(r_inner, r_outer, n_r + 1)
    theta_edges = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)

    r_mid     = 0.5 * (r_edges[:-1] + r_edges[1:])       # shape (n_r,)
    theta_mid = 0.5 * (theta_edges[:-1] + theta_edges[1:])  # shape (n_theta,)

    # Areas: dA = r * dr * dtheta
    dr     = np.diff(r_edges)           # shape (n_r,)
    dtheta = np.diff(theta_edges)       # shape (n_theta,)

    # Meshgrid
    r_grid, theta_grid = np.meshgrid(r_mid, theta_mid, indexing="ij")   # (n_r, n_theta)
    dr_grid, dtheta_grid = np.meshgrid(dr, dtheta, indexing="ij")

    A_grid = r_grid * dr_grid * dtheta_grid                              # (n_r, n_theta)

    # Centroid positions
    z_grid = r_grid * np.cos(theta_grid)                                  # vertical fibre height
    y_grid = r_grid * np.sin(theta_grid)                                  # horizontal

    # Flatten to 1-D arrays of length n_r * n_theta
    z_ij = z_grid.flatten()
    y_ij = y_grid.flatten()
    A_ij = A_grid.flatten()
    r_ij = r_grid.flatten()

    return z_ij, y_ij, A_ij, r_ij


class CrossSectionIntegrator:
    """
    Pre-computes patch geometry for a given pipe cross-section and provides
    methods to integrate N and M given strain field inputs from the PINN.

    Usage:
        integrator = CrossSectionIntegrator(D, t, n_r=4, n_theta=16)
        N, M = integrator.integrate(u_x, w_x, w_xx, pipe_params)
    """

    def __init__(self, D, t, n_r=4, n_theta=16, material_params=None):
        """
        Args:
            D, t           : outer diameter and wall thickness (m)
            n_r, n_theta   : cross-section discretisation resolution
            material_params: dict, defaults to X65_DEFAULTS
        """
        self.D = D
        self.t = t
        self.n_r = n_r
        self.n_theta = n_theta
        self.params = material_params or X65_DEFAULTS

        z_np, y_np, A_np, r_np = build_cross_section_patches(D, t, n_r, n_theta)

        # Store as float32 tensors for GPU compatibility
        self.z_ij = torch.tensor(z_np, dtype=torch.float32)   # (n_patches,)
        self.y_ij = torch.tensor(y_np, dtype=torch.float32)
        self.A_ij = torch.tensor(A_np, dtype=torch.float32)
        self.r_ij = torch.tensor(r_np, dtype=torch.float32)

        self.n_patches = len(self.A_ij)

    def to(self, device):
        """Move patch tensors to specified device."""
        self.z_ij = self.z_ij.to(device)
        self.y_ij = self.y_ij.to(device)
        self.A_ij = self.A_ij.to(device)
        self.r_ij = self.r_ij.to(device)
        return self

    def integrate(self, u_x, w_x, w_xx, P, delta_T):
        """
        Compute internal axial force N and bending moment M at each
        collocation point along the pipe.

        Args:
            u_x    : ∂u/∂x, shape (N_col,)  — from PINN autodiff
            w_x    : ∂w/∂x, shape (N_col,)
            w_xx   : ∂²w/∂x², shape (N_col,)
            P      : internal pressure (Pa, scalar or tensor broadcastable to N_col)
            delta_T: temperature change (°C, scalar or tensor)

        Returns:
            N_force : axial force N(x), shape (N_col,)
            M_moment: bending moment M(x), shape (N_col,)
        """
        device  = u_x.device
        z_ij    = self.z_ij.to(device)     # (n_patches,)
        A_ij    = self.A_ij.to(device)

        params  = self.params
        D, t    = self.D, self.t
        E, nu   = params["E"], params["nu"]
        alpha_T = params["alpha_T"]

        # Hoop stress and biaxial yield (scalar approximation — same across section)
        sigma_h = hoop_stress(P, D, t)
        sigma_y = params["sigma_y"]
        sigma_yT, sigma_yC = biaxial_yield_stresses(sigma_y, sigma_h)

        # Initial strain (scalar)
        eps_init = initial_strain(P, D, t, E, nu, alpha_T, delta_T)

        # Expand collocation points for broadcasting over patches
        # u_x: (N_col,)  →  (N_col, 1)
        u_x_  = u_x.unsqueeze(-1)     # (N_col, 1)
        w_x_  = w_x.unsqueeze(-1)
        w_xx_ = w_xx.unsqueeze(-1)
        z_    = z_ij.unsqueeze(0)     # (1, n_patches)

        # Geometric-mechanical strain at each fibre: (N_col, n_patches)
        eps_gm = eps_GM_full(u_x_, w_x_, w_xx_, z_)
        eps_l  = eps_init + eps_gm    # total longitudinal strain

        # Stress via Menegotto-Pinto (vectorised over all patches)
        sigma_l = menegotto_pinto_stress(eps_l, sigma_yT, sigma_yC, params)

        # Numerical integration over patches
        # N = Σ σ_ij * A_ij
        A_ = A_ij.unsqueeze(0)        # (1, n_patches)
        N_force  = (sigma_l * A_).sum(dim=-1)               # (N_col,)

        # M = -Σ z_ij * σ_ij * A_ij
        M_moment = -(sigma_l * z_ * A_).sum(dim=-1)         # (N_col,)

        return N_force, M_moment


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    D, t = 0.914, 0.0127
    integrator = CrossSectionIntegrator(D, t, n_r=6, n_theta=24)
    print(f"n_patches = {integrator.n_patches}")
    print(f"Total cross-section area = {integrator.A_ij.sum().item()*1e4:.3f} cm²")

    # Analytical annular area for comparison
    A_annulus = np.pi / 4.0 * (D**2 - (D - 2*t)**2)
    print(f"Analytical area          = {A_annulus*1e4:.3f} cm²")

    # Test integration with simple uniform strain
    N_col = 10
    u_x   = torch.full((N_col,), 0.001)     # 0.1% axial strain
    w_x   = torch.zeros(N_col)
    w_xx  = torch.zeros(N_col)
    P     = 10e6     # 10 MPa
    dT    = 50.0     # 50°C

    N_force, M_moment = integrator.integrate(u_x, w_x, w_xx, P, dT)
    print(f"N (uniform strain, no bending): {N_force[0].item()/1e6:.3f} MN")
    print(f"M (uniform strain, no bending): {M_moment[0].item()/1e3:.3f} kNm")
