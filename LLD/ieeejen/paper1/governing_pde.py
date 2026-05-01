"""
PDE residual computations for the Euler-Bernoulli beam on Winkler foundation,
implemented with PyTorch automatic differentiation.

Governing PDEs:
    Eq. 1a:  dN/dx + h(Ug - u) = 0          [axial equilibrium]
    Eq. 1b:  d²M/dx² - d/dx[N·dw/dx] - q(Wg - w) = 0   [lateral equilibrium]

Derivatives of N and M w.r.t. x are computed via autodiff through the PINN outputs.

References:
    Haghighat et al. (2025), PINN-RA
"""

import torch


def compute_grads(output, input_tensor, create_graph=True):
    """
    Compute ∂output/∂input_tensor via autograd.

    Args:
        output        : scalar or tensor (N,)
        input_tensor  : the input w.r.t. which we differentiate (requires_grad=True)
        create_graph  : True to allow higher-order derivatives

    Returns:
        grad : same shape as output
    """
    grad = torch.autograd.grad(
        outputs=output,
        inputs=input_tensor,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    return grad


def get_displacement_derivatives(network, X_col):
    """
    Forward pass through PINN and compute all required spatial derivatives
    of u and w with respect to x using automatic differentiation.

    Args:
        network : PINN model, outputs (u, w) from input X_col
        X_col   : collocation points, shape (N_col, 6), requires_grad=True on x-component

    Returns:
        dict with keys:
            u, w           : displacements (N_col,)
            u_x, w_x, w_xx: first/second spatial derivatives (N_col,)
    """
    # Ensure gradients flow through x (column 0 of X_col)
    # X_col[:, 0] = x is the spatial coordinate
    out = network(X_col)           # (N_col, 2)
    u   = out[:, 0]
    w   = out[:, 1]

    # First derivatives w.r.t. x
    u_x  = compute_grads(u, X_col)[:, 0]     # du/dx
    w_x  = compute_grads(w, X_col)[:, 0]     # dw/dx

    # Second derivative of w w.r.t. x
    w_xx = compute_grads(w_x, X_col)[:, 0]   # d²w/dx²

    return {"u": u, "w": w, "u_x": u_x, "w_x": w_x, "w_xx": w_xx}


def pde_residual_axial(network, X_col, N_force, dN_dx, h_spring):
    """
    Residual of axial equilibrium PDE:

        R1 = dN/dx + h(Ug - u) = 0

    Args:
        network  : PINN model
        X_col    : collocation points (N_col, 6), x in col 0
        N_force  : internal axial force N(x) (N_col,), from cross-section integrator
        dN_dx    : dN/dx (N_col,), computed via autodiff on N_force
        h_spring : axial spring force h(Ug - u) (N_col,)

    Returns:
        R1 : residual tensor (N_col,)
    """
    R1 = dN_dx + h_spring
    return R1


def pde_residual_lateral(dM_xx, N_force, w_x, dNwx_dx, q_spring):
    """
    Residual of lateral equilibrium PDE:

        R2 = d²M/dx² - d/dx[N·dw/dx] - q(Wg - w) = 0

    Args:
        dM_xx    : d²M/dx² (N_col,), second derivative of bending moment
        N_force  : N(x) (N_col,)
        w_x      : dw/dx (N_col,)
        dNwx_dx  : d/dx[N·dw/dx] (N_col,)
        q_spring : lateral spring force q(Wg - w) (N_col,)

    Returns:
        R2 : residual tensor (N_col,)
    """
    R2 = dM_xx - dNwx_dx - q_spring
    return R2


def compute_all_residuals(network, X_col, cross_section_integrator,
                          spring_params, ground_disp_fn, pipe_params):
    """
    Compute both PDE residuals at collocation points X_col.

    This is the main entry point called during PINN training.

    Args:
        network                 : PINN model
        X_col                   : (N_col, 6) collocation points [x, δ, c, φ, γ, H]
                                  with requires_grad=True
        cross_section_integrator: CrossSectionIntegrator instance
        spring_params           : dict with Tu, Pu, Delta_t, Delta_p (computed from inputs)
        ground_disp_fn          : function(x, delta, beta) → (Ug, Wg)
        pipe_params             : dict with P, delta_T, D, t etc.

    Returns:
        R1 : axial equilibrium residual (N_col,)
        R2 : lateral equilibrium residual (N_col,)
    """
    from .soil_springs import axial_spring_force, lateral_spring_force

    # ── Extract collocation components ──────────────────────────────────────
    x     = X_col[:, 0]
    delta = X_col[:, 1]
    c     = X_col[:, 2]
    phi   = X_col[:, 3]
    gamma = X_col[:, 4]
    H     = X_col[:, 5]

    # ── Forward pass + first spatial derivatives ─────────────────────────────
    out  = network(X_col)
    u    = out[:, 0]
    w    = out[:, 1]

    u_x  = compute_grads(u, X_col)[:, 0]
    w_x  = compute_grads(w, X_col)[:, 0]
    w_xx = compute_grads(w_x, X_col)[:, 0]

    # ── Cross-section integration → N, M ────────────────────────────────────
    P      = pipe_params.get("P", 10e6)
    dT     = pipe_params.get("delta_T", 0.0)
    N_force, M_moment = cross_section_integrator.integrate(u_x, w_x, w_xx, P, dT)

    # ── Derivatives of N and M w.r.t. x ─────────────────────────────────────
    dN_dx = compute_grads(N_force, X_col)[:, 0]        # dN/dx
    dM_dx = compute_grads(M_moment, X_col)[:, 0]       # dM/dx
    dM_xx = compute_grads(dM_dx, X_col)[:, 0]          # d²M/dx²

    # ── d/dx[N·dw/dx] ────────────────────────────────────────────────────────
    Nwx    = N_force * w_x
    dNwx_dx = compute_grads(Nwx, X_col)[:, 0]

    # ── Ground displacement profile ──────────────────────────────────────────
    beta_deg = pipe_params.get("beta_deg", 45.0)
    x_start  = pipe_params.get("x_start", 140.0)
    x_end    = pipe_params.get("x_end", 160.0)
    Ug, Wg   = ground_disp_fn(x, delta, beta_deg, x_start, x_end)

    # ── Soil spring parameters (per-sample, may vary with c, phi, gamma, H) ─
    from .soil_springs import compute_spring_params
    sp = compute_spring_params(c, phi, gamma, H, pipe_params.get("D", 0.914))
    Tu      = sp["Tu"]
    Pu      = sp["Pu"]
    Delta_t = sp["Delta_t"]
    Delta_p = sp["Delta_p"]

    h_spring = axial_spring_force(Ug, u, Tu, Delta_t)
    q_spring = lateral_spring_force(Wg, w, Pu, Delta_p)

    # ── PDE residuals ────────────────────────────────────────────────────────
    R1 = pde_residual_axial(network, X_col, N_force, dN_dx, h_spring)
    R2 = pde_residual_lateral(dM_xx, N_force, w_x, dNwx_dx, q_spring)

    return R1, R2
