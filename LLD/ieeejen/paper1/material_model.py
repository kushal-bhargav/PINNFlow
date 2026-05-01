"""
Material model for buried steel pipeline analysis.

Implements:
  - Biaxial Von Mises yield criterion for modified yield stresses
  - Modified Menegotto-Pinto (M-P) smooth stress-strain model
  - Total longitudinal strain decomposition (initial + geometric-mechanical)

References:
    Haghighat et al. (2025), PINN-RA for Buried Pipeline Reliability Analysis
    Menegotto & Pinto (1973), Hysteretic behaviour of steel sections
"""

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Default X65 Steel Material Parameters
# ─────────────────────────────────────────────────────────────────────────────

X65_DEFAULTS = {
    "E"     : 207e9,    # Young's modulus (Pa)
    "sigma_y": 450e6,   # Nominal uniaxial yield stress (Pa)
    "nu"    : 0.3,      # Poisson's ratio
    "alpha_T": 1.17e-5, # Thermal expansion coefficient (1/°C)
    "bT"    : 0.02,     # Ratio plastic/elastic modulus in tension
    "bC"    : 0.02,     # Ratio plastic/elastic modulus in compression
    "R"     : 20.0,     # Transition smoothness (higher → sharper knee)
    "omega" : 500.0,    # Tensile-compressive switching sharpness
}


# ─────────────────────────────────────────────────────────────────────────────
# Biaxial Yield Stress via Von Mises Criterion
# ─────────────────────────────────────────────────────────────────────────────

def hoop_stress(P, D, t):
    """
    Hoop stress from internal pressure.

        sigma_h = P * (D - 2t) / (2t)

    Args:
        P : internal pressure (Pa)
        D : outer diameter (m)
        t : wall thickness (m)

    Returns:
        sigma_h : hoop stress (Pa), tensor or float
    """
    return P * (D - 2.0 * t) / (2.0 * t)


def biaxial_yield_stresses(sigma_y, sigma_h):
    """
    Modified yield stresses in tension and compression under biaxial state.

    From Von Mises criterion with σ₁ = σ_l (longitudinal), σ₂ = σ_h (hoop):
        σ_y^T = 0.5*(σ_h + sqrt(4*σ_y² - 3*σ_h²))
        σ_y^C = 0.5*(σ_h - sqrt(4*σ_y² - 3*σ_h²))

    Args:
        sigma_y : uniaxial yield stress (Pa)
        sigma_h : hoop stress (Pa)

    Returns:
        sigma_yT : biaxial tensile yield stress (Pa)
        sigma_yC : biaxial compressive yield stress (Pa, negative value)
    """
    discriminant = 4.0 * sigma_y ** 2 - 3.0 * sigma_h ** 2
    # Clamp to avoid sqrt of negative (physically: hoop stress must be < 2/sqrt(3)*sigma_y)
    if isinstance(discriminant, torch.Tensor):
        discriminant = torch.clamp(discriminant, min=0.0)
        sqrt_term = torch.sqrt(discriminant)
    else:
        discriminant = max(discriminant, 0.0)
        sqrt_term = np.sqrt(discriminant)

    sigma_yT = 0.5 * (sigma_h + sqrt_term)
    sigma_yC = 0.5 * (sigma_h - sqrt_term)
    return sigma_yT, sigma_yC


# ─────────────────────────────────────────────────────────────────────────────
# Menegotto-Pinto Smooth Stress-Strain Model
# ─────────────────────────────────────────────────────────────────────────────

def mp_branch(eps, E_branch, b_branch, sigma_y_branch, R):
    """
    Single M-P branch (tension OR compression).

        σ = E * ε * [b + (1 - b) / (1 + (E*ε / σ_y)^R)^(1/R)]

    This is a smooth approximation to elastic-hardening-plastic behaviour.

    Args:
        eps           : longitudinal strain (tensor)
        E_branch      : elastic modulus for this branch (Pa)
        b_branch      : hardening ratio (plastic slope / elastic slope)
        sigma_y_branch: yield stress for this branch (Pa)
        R             : curvature parameter (smoothness of knee)

    Returns:
        sigma : longitudinal stress (Pa)
    """
    x = E_branch * eps / (sigma_y_branch + 1e-12)    # normalised strain
    if isinstance(x, torch.Tensor):
        denom = (1.0 + torch.abs(x) ** R) ** (1.0 / R)
    else:
        denom = (1.0 + np.abs(x) ** R) ** (1.0 / R)
    sigma = E_branch * eps * (b_branch + (1.0 - b_branch) / denom)
    return sigma


def menegotto_pinto_stress(eps_l, sigma_yT, sigma_yC, params=None):
    """
    Full Modified Menegotto-Pinto stress as a function of longitudinal strain,
    including smooth switching between tension and compression branches via tanh.

        σ_l^T  = mp_branch(eps, E, bT, sigma_yT, R)
        σ_l^C  = mp_branch(eps, E, bC, sigma_yC, R)
        σ_l    = [σ_l^T - σ_l^C] / [1 + exp(-ω * eps)] + σ_l^C

    Args:
        eps_l    : longitudinal strain (tensor or float)
        sigma_yT : biaxial tensile yield stress (Pa)
        sigma_yC : biaxial compressive yield stress (Pa)
        params   : dict of material parameters (uses X65_DEFAULTS if None)

    Returns:
        sigma_l : longitudinal stress (Pa)
    """
    if params is None:
        params = X65_DEFAULTS

    E     = params["E"]
    bT    = params["bT"]
    bC    = params["bC"]
    R     = params["R"]
    omega = params["omega"]

    sigma_T = mp_branch(eps_l, E, bT, sigma_yT, R)
    sigma_C = mp_branch(eps_l, E, bC, sigma_yC, R)

    if isinstance(eps_l, torch.Tensor):
        blend = torch.sigmoid(omega * eps_l)          # ≈1 for tension, ≈0 for compression
    else:
        blend = 1.0 / (1.0 + np.exp(-omega * eps_l))

    sigma_l = (sigma_T - sigma_C) * blend + sigma_C
    return sigma_l


# ─────────────────────────────────────────────────────────────────────────────
# Total Longitudinal Strain (decomposed)
# ─────────────────────────────────────────────────────────────────────────────

def initial_strain(P, D, t, E, nu, alpha_T, delta_T):
    """
    Initial longitudinal strain from internal pressure and temperature.

        ε_initial = ν * P * (D - 2t) / (2t*E) - α * ΔT

    Note: minus sign for temperature if pipe is in operation (constrained expansion
    treated as compressive strain convention). Sign follows ALA/ASME convention.

    Args:
        P       : internal pressure (Pa)
        D       : outer diameter (m)
        t       : wall thickness (m)
        E       : Young's modulus (Pa)
        nu      : Poisson's ratio
        alpha_T : thermal expansion coefficient (1/°C)
        delta_T : temperature rise from installation to operation (°C)

    Returns:
        eps_initial : initial strain (dimensionless)
    """
    eps_pressure = nu * P * (D - 2.0 * t) / (2.0 * t * E)
    eps_thermal  = alpha_T * delta_T
    return eps_pressure + eps_thermal


def geometric_mechanical_strain(u_x, w_x, z):
    """
    Geometric-mechanical longitudinal strain at fibre position z.

    Includes:
      - Membrane axial strain: u_x (first-order)
      - Geometric nonlinearity: 0.5 * w_x²
      - Bending strain: -z * w_xx

    NOTE: w_xx must be provided by the caller (second derivative of lateral displacement).
    This function takes u_x, w_x, w_xx as arguments (all computed via autodiff in PINNs).

    Args:
        u_x  : du/dx (axial strain, tensor)
        w_x  : dw/dx (lateral slope, tensor)
        z    : fibre distance from neutral axis (m, scalar or tensor)
             (z > 0 for outer tension fibre)

    Returns:
        eps_GM : geometric-mechanical strain (tensor)
    """
    # w_xx must be passed separately; computed by autodiff in caller
    # Here signature keeps z and w_xx separate for clarity
    # Actual call: eps_GM(u_x, w_x, w_xx, z)
    raise NotImplementedError("Call eps_GM_full(u_x, w_x, w_xx, z) instead.")


def eps_GM_full(u_x, w_x, w_xx, z):
    """
    Full geometric-mechanical longitudinal strain at fibre depth z.

        ε_GM = u_x + 0.5 * w_x² - z * w_xx

    Args:
        u_x  : ∂u/∂x (tensor)
        w_x  : ∂w/∂x (tensor)
        w_xx : ∂²w/∂x² (tensor)
        z    : fibre height from neutral axis (m, scalar or tensor)

    Returns:
        eps_GM : strain tensor
    """
    return u_x + 0.5 * w_x ** 2 - z * w_xx


def total_longitudinal_strain(u_x, w_x, w_xx, z, P, D, t, E, nu, alpha_T, delta_T):
    """
    Total longitudinal strain = initial + geometric-mechanical.

    Args:
        (see individual functions above)

    Returns:
        eps_l : total longitudinal strain at fibre z (tensor)
    """
    eps_init = initial_strain(P, D, t, E, nu, alpha_T, delta_T)
    eps_gm   = eps_GM_full(u_x, w_x, w_xx, z)
    return eps_init + eps_gm


# ─────────────────────────────────────────────────────────────────────────────
# Von Mises Equivalent Stress
# ─────────────────────────────────────────────────────────────────────────────

def von_mises_stress(sigma_l, sigma_h):
    """
    Von Mises equivalent stress for biaxial state (pipe under combined axial + hoop).

        σ_vM = sqrt(σ_l² + σ_h² - σ_l * σ_h)

    Args:
        sigma_l : longitudinal stress (Pa)
        sigma_h : hoop stress (Pa)

    Returns:
        sigma_vM : Von Mises stress (Pa, always non-negative)
    """
    if isinstance(sigma_l, torch.Tensor):
        return torch.sqrt(sigma_l ** 2 + sigma_h ** 2 - sigma_l * sigma_h + 1e-20)
    else:
        return np.sqrt(sigma_l ** 2 + sigma_h ** 2 - sigma_l * sigma_h + 1e-20)


# ─────────────────────────────────────────────────────────────────────────────
# High-level: compute max strain over pipe cross-section
# ─────────────────────────────────────────────────────────────────────────────

def max_longitudinal_strain(u_x, w_x, w_xx, D, P, t, E, nu, alpha_T, delta_T):
    """
    Maximum longitudinal strain over all fibres of the pipe cross-section.

    For a circular pipe, the extreme fibres are at z = ±(D/2 - t) (mid-wall).
    Returns max(|ε_l|) at outer tension fibre (z = +D/2) and outer compression
    fibre (z = -D/2).

    Args:
        u_x, w_x, w_xx : displacement derivatives (tensors, same shape)
        D, t           : diameter and wall thickness (m)
        P              : internal pressure (Pa)
        E, nu          : material properties
        alpha_T        : thermal expansion coefficient
        delta_T        : temperature change (°C)

    Returns:
        eps_max_tension    : max tensile longitudinal strain
        eps_max_compression: max compressive longitudinal strain (most negative)
    """
    z_outer = D / 2.0        # outer fibre (tension side)
    z_inner = -D / 2.0       # inner fibre (compression side)

    eps_outer = total_longitudinal_strain(u_x, w_x, w_xx, z_outer, P, D, t, E, nu, alpha_T, delta_T)
    eps_inner = total_longitudinal_strain(u_x, w_x, w_xx, z_inner, P, D, t, E, nu, alpha_T, delta_T)

    if isinstance(eps_outer, torch.Tensor):
        eps_tension    = torch.max(eps_outer)
        eps_compression = torch.min(eps_inner)
    else:
        eps_tension    = float(np.max(eps_outer))
        eps_compression = float(np.min(eps_inner))

    return eps_tension, eps_compression


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    params = X65_DEFAULTS

    # Biaxial yield stresses
    P, D, t = 10e6, 0.914, 0.0127      # 10 MPa pressure, 914mm, 12.7mm
    sigma_h = hoop_stress(P, D, t)
    sigma_y = params["sigma_y"]

    sigma_yT, sigma_yC = biaxial_yield_stresses(sigma_y, sigma_h)
    print(f"sigma_h  = {sigma_h/1e6:.2f} MPa")
    print(f"sigma_yT = {sigma_yT/1e6:.2f} MPa")
    print(f"sigma_yC = {sigma_yC/1e6:.2f} MPa")

    # Stress-strain curve (tensor path)
    eps = torch.linspace(-0.03, 0.03, 200)
    sigma_l = menegotto_pinto_stress(eps, sigma_yT, sigma_yC, params)
    print(f"Stress range: [{sigma_l.min().item()/1e6:.2f}, {sigma_l.max().item()/1e6:.2f}] MPa")
