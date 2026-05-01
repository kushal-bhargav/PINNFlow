"""
Soil spring force-displacement relationships per ALA (American Lifelines Alliance) guidelines.
Implements elastic-perfectly-plastic springs smoothed with tanh for autodiff compatibility.

References:
    Haghighat et al. (2025), PINN-RA for Buried Pipeline Reliability Analysis
    ALA Guidelines for the Design of Buried Steel Pipe (2001)
"""

import torch
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# ALA Ultimate Soil Resistance Calculations
# ─────────────────────────────────────────────────────────────────────────────

def ultimate_axial_resistance(c, phi, gamma, H, D, alpha=0.5, delta_factor=0.5):
    """
    Compute ultimate axial soil resistance Tu per ALA guidelines.

    Tu = pi * D * (alpha*c + gamma*H*(1+K0)/2 * tan(delta))

    Args:
        c      : soil cohesion (Pa or kPa, consistent units)
        phi    : soil friction angle (degrees)
        gamma  : soil unit weight (kN/m³ or N/m³)
        H      : burial depth to pipe centreline (m)
        D      : pipe outer diameter (m)
        alpha  : adhesion factor (dimensionless), default 0.5
        delta_factor : ratio delta/phi (interface friction), default 0.5

    Returns:
        Tu : ultimate axial resistance per unit length (same force/length units)
    """
    phi_rad   = phi * np.pi / 180.0
    delta_rad = delta_factor * phi_rad
    K0        = 1.0 - np.sin(phi_rad)          # at-rest earth pressure coefficient
    sigma_v0  = gamma * H                       # vertical effective stress at pipe centre

    Tu = np.pi * D * (alpha * c + sigma_v0 * (1.0 + K0) / 2.0 * np.tan(delta_rad))
    return Tu


def ultimate_lateral_resistance(c, phi, gamma, H, D, Nch=6.0, Nqh=None):
    """
    Compute ultimate lateral (horizontal) soil resistance Pu per ALA guidelines.

    Pu = (Nch*c + Nqh*gamma*H) * D

    Args:
        c    : soil cohesion
        phi  : soil friction angle (degrees)
        gamma: soil unit weight
        H    : burial depth to pipe centreline (m)
        D    : pipe outer diameter (m)
        Nch  : bearing capacity factor for cohesion (default 6.0 for H/D >= 2)
        Nqh  : bearing capacity factor for surcharge (computed from phi if None)

    Returns:
        Pu : ultimate lateral resistance per unit length
    """
    if Nqh is None:
        # Approximate Nqh from friction angle (ALA Table)
        Nqh = _compute_Nqh(phi)
    Pu = (Nch * c + Nqh * gamma * H) * D
    return Pu


def _compute_Nqh(phi):
    """
    Approximate horizontal bearing capacity factor Nqh from friction angle.
    Linear interpolation of ALA Table values.
    """
    phi_table = np.array([0, 10, 20, 30, 40, 45])
    Nqh_table = np.array([1.0, 2.0, 4.0, 10.0, 20.0, 35.0])
    return float(np.interp(phi, phi_table, Nqh_table))


# ─────────────────────────────────────────────────────────────────────────────
# Yield Displacement Estimates (ALA)
# ─────────────────────────────────────────────────────────────────────────────

def axial_yield_displacement(D):
    """
    Axial yield displacement Delta_t (m) — ALA recommendation.
    Delta_t = 3mm for dense sand; 5mm for loose sand; 8–10mm for clay.
    Conservative default: 0.005 m (5 mm).
    """
    return 0.005 * torch.ones_like(D) if isinstance(D, torch.Tensor) else 0.005


def lateral_yield_displacement(H, D):
    """
    Lateral yield displacement Delta_p (m) — ALA recommendation.
    Delta_p = 0.04*(H + D/2) for sand (loose to dense range).
    """
    return 0.04 * (H + D / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tanh-Smoothed Spring Force (differentiable)
# ─────────────────────────────────────────────────────────────────────────────

def axial_spring_force(Ug, u, Tu, Delta_t):
    """
    Elastic-perfectly-plastic axial spring force, smoothed with tanh.

        h(Ug - u) = Tu * tanh((Ug - u) / Delta_t)

    Args:
        Ug      : ground axial displacement (tensor)
        u       : pipe axial displacement (tensor, network output)
        Tu      : ultimate axial resistance (scalar or tensor)
        Delta_t : axial yield displacement (scalar or tensor)

    Returns:
        h : axial spring force per unit length (tensor), differentiable w.r.t. u
    """
    relative_disp = Ug - u
    return Tu * torch.tanh(relative_disp / Delta_t)


def lateral_spring_force(Wg, w, Pu, Delta_p):
    """
    Elastic-perfectly-plastic lateral spring force, smoothed with tanh.

        q(Wg - w) = Pu * tanh((Wg - w) / Delta_p)

    Args:
        Wg      : ground lateral displacement (tensor)
        w       : pipe lateral displacement (tensor, network output)
        Pu      : ultimate lateral resistance (scalar or tensor)
        Delta_p : lateral yield displacement (scalar or tensor)

    Returns:
        q : lateral spring force per unit length (tensor), differentiable w.r.t. w
    """
    relative_disp = Wg - w
    return Pu * torch.tanh(relative_disp / Delta_p)


# ─────────────────────────────────────────────────────────────────────────────
# Ground Displacement Profile (Rectangular pattern)
# ─────────────────────────────────────────────────────────────────────────────

def ground_displacement_profile(x, delta, beta_deg, x_start, x_end, smooth=True, k=50.0):
    """
    Rectangular (or smoothed) ground displacement profile.

    Inside the affected zone [x_start, x_end]:
        Ug = delta * cos(beta)
        Wg = delta * sin(beta)
    Outside: zero.

    Args:
        x        : pipe coordinate tensor, shape (N,)
        delta    : ground displacement magnitude (m)
        beta_deg : inclination angle to pipe axis (degrees)
        x_start  : start of affected zone (m)
        x_end    : end of affected zone (m)
        smooth   : if True, use sigmoid-based smooth step instead of hard cutoff
        k        : smoothness parameter for sigmoid transition

    Returns:
        Ug, Wg : axial and lateral ground displacement tensors, shape (N,)
    """
    beta_rad = beta_deg * torch.pi / 180.0

    if smooth:
        # Smooth rectangular pulse: sigmoid(k*(x - x_start)) - sigmoid(k*(x - x_end))
        window = torch.sigmoid(k * (x - x_start)) - torch.sigmoid(k * (x - x_end))
    else:
        window = ((x >= x_start) & (x <= x_end)).float()

    Ug = delta * torch.cos(beta_rad) * window
    Wg = delta * torch.sin(beta_rad) * window
    return Ug, Wg


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute all soil spring parameters from raw soil properties
# ─────────────────────────────────────────────────────────────────────────────

def compute_spring_params(c, phi, gamma, H, D):
    """
    Compute all spring parameters from soil/geometry inputs.
    Inputs can be scalars (numpy) or batched tensors.

    Returns dict with Tu, Pu, Delta_t, Delta_p.
    """
    if isinstance(c, torch.Tensor):
        # Tensor path — keep on same device
        phi_rad   = phi * torch.pi / 180.0
        delta_rad = 0.5 * phi_rad
        K0        = 1.0 - torch.sin(phi_rad)
        sigma_v0  = gamma * H

        Tu = torch.pi * D * (0.5 * c + sigma_v0 * (1.0 + K0) / 2.0 * torch.tan(delta_rad))

        # Simplified Nqh for tensor path (use fixed approximate formula)
        Nqh = torch.exp(0.18 * phi - 2.5).clamp(min=1.0, max=40.0)
        Pu  = (6.0 * c + Nqh * gamma * H) * D

        Delta_t = torch.full_like(Tu, 0.005)
        Delta_p = 0.04 * (H + D / 2.0)
    else:
        Tu      = ultimate_axial_resistance(c, phi, gamma, H, D)
        Pu      = ultimate_lateral_resistance(c, phi, gamma, H, D)
        Delta_t = axial_yield_displacement(D)
        Delta_p = lateral_yield_displacement(H, D)

    return {"Tu": Tu, "Pu": Pu, "Delta_t": Delta_t, "Delta_p": Delta_p}


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: X65 steel pipe buried in cohesionless sand
    c     = 0.0          # kPa (sand — no cohesion)
    phi   = 35.0         # degrees
    gamma = 18.0         # kN/m³
    H     = 1.5          # m
    D     = 0.914        # m (36 inch pipe)

    Tu = ultimate_axial_resistance(c, phi, gamma, H, D)
    Pu = ultimate_lateral_resistance(c, phi, gamma, H, D)
    print(f"Tu = {Tu:.3f} kN/m,  Pu = {Pu:.3f} kN/m")

    # Tanh spring force
    Ug = torch.tensor([0.0, 0.002, 0.005, 0.01, 0.05])
    u  = torch.zeros(5)
    h  = axial_spring_force(Ug, u, Tu=Tu, Delta_t=0.005)
    print(f"Axial spring forces: {h}")
