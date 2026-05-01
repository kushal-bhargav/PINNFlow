"""
Beam PDE residuals for dynamic pipe load estimation.

Euler-Bernoulli:
    EI * ∂⁴w/∂x⁴ + ρA * ∂²w/∂t² = f(x,t)

Timoshenko (two coupled PDEs):
    ρA * ∂²w/∂t² - kGA*(∂²w/∂x² - ∂ψ/∂x) = f(x,t)
    ρI * ∂²ψ/∂t² - EI*∂²ψ/∂x² - kGA*(∂w/∂x - ψ) = 0

Used in the inverse problem: residuals drive the PINN to recover
unknown dynamic forcing f(x,t) from sparse sensor data.

References:
    Patel et al. (2025), PINNs for Dynamic Load Estimation in Pipework (IJFMR)
"""

import torch
from typing import Tuple


def grad1(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """First derivative of y w.r.t. x (for vector y)."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]


# ─────────────────────────────────────────────────────────────────────────────
# Euler-Bernoulli beam PDE
# ─────────────────────────────────────────────────────────────────────────────

def eb_pde_residual(
    w_net: "callable",
    f_net: "callable",
    X: torch.Tensor,
    E: float,
    I: float,
    rho: float,
    A: float,
) -> torch.Tensor:
    """
    Euler-Bernoulli PDE residual at collocation points X = (x, t).

        R = EI * ∂⁴w/∂x⁴ + ρA * ∂²w/∂t² - f̂(x,t)

    Args:
        w_net : network predicting w(x,t)
        f_net : network predicting f(x,t) (load network)
        X     : (N, 2) collocation points [x, t], requires_grad=True
        E,I   : Young's modulus and second moment of area
        rho,A : density and cross-sectional area

    Returns:
        R : residual (N,)
    """
    w   = w_net(X)[:, 0]
    f   = f_net(X)[:, 0]

    # Spatial derivatives: ∂⁴w/∂x⁴
    wx   = grad1(w,   X)[:, 0]
    wxx  = grad1(wx,  X)[:, 0]
    wxxx = grad1(wxx, X)[:, 0]
    wxxxx= grad1(wxxx,X)[:, 0]

    # Temporal derivatives: ∂²w/∂t²
    wt   = grad1(w,  X)[:, 1]
    wtt  = grad1(wt, X)[:, 1]

    R = E * I * wxxxx + rho * A * wtt - f
    return R


# ─────────────────────────────────────────────────────────────────────────────
# Timoshenko beam PDE
# ─────────────────────────────────────────────────────────────────────────────

def timoshenko_pde_residuals(
    w_net: "callable",
    psi_net: "callable",
    f_net: "callable",
    X: torch.Tensor,
    E: float,
    I: float,
    rho: float,
    A: float,
    G: float,
    k_s: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Timoshenko beam PDE residuals for two coupled PDEs.

    PDE 1:  ρA * ∂²w/∂t² - kGA*(∂²w/∂x² - ∂ψ/∂x) = f(x,t)
    PDE 2:  ρI * ∂²ψ/∂t² - EI*∂²ψ/∂x² - kGA*(∂w/∂x - ψ) = 0

    Args:
        w_net  : network predicting w(x,t), output shape (N,1) or (N,)
        psi_net: network predicting ψ(x,t) (cross-section rotation)
        f_net  : network predicting f(x,t)
        X      : (N, 2) collocation points [x, t], requires_grad=True
        E      : Young's modulus (Pa)
        I      : second moment of area (m⁴)
        rho    : density (kg/m³)
        A      : cross-sectional area (m²)
        G      : shear modulus (Pa)
        k_s    : Timoshenko shear correction factor

    Returns:
        R1, R2 : residuals of PDE 1 and PDE 2 (N,)
    """
    kGA = k_s * G * A

    w   = w_net(X)[:, 0]
    psi = psi_net(X)[:, 0]
    f   = f_net(X)[:, 0]

    # ∂²w/∂t²
    wt   = grad1(w,  X)[:, 1]
    wtt  = grad1(wt, X)[:, 1]

    # ∂²w/∂x², ∂w/∂x
    wx   = grad1(w,  X)[:, 0]
    wxx  = grad1(wx, X)[:, 0]

    # ∂ψ/∂x, ∂²ψ/∂x², ∂²ψ/∂t²
    psix  = grad1(psi,  X)[:, 0]
    psixx = grad1(psix, X)[:, 0]
    psit  = grad1(psi,  X)[:, 1]
    psitt = grad1(psit, X)[:, 1]

    # PDE 1
    R1 = rho * A * wtt - kGA * (wxx - psix) - f

    # PDE 2
    R2 = rho * I * psitt - E * I * psixx - kGA * (wx - psi)

    return R1, R2
