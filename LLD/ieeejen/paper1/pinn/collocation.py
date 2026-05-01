"""
Collocation point sampling for the 6-dimensional parametric PINN.

Samples points in the joint space [x, δ, c, φ, γ, H] for:
  - Interior PDE residual evaluation (X_col)
  - Boundary condition enforcement (X_bc)

Supports:
  - Uniform random sampling
  - Latin Hypercube Sampling (LHS) for better space coverage
  - Fixed boundary points at x=0 and x=L

References:
    Haghighat et al. (2025), PINN-RA for Buried Pipeline Reliability Analysis
"""

import torch
import numpy as np
from scipy.stats import qmc
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DomainBounds:
    """
    Defines the 6-dimensional parameter domain for collocation sampling.

    Dimensions: [x, δ, c, φ, γ, H]
    """
    # Spatial
    x_lb:     float = 0.0
    x_ub:     float = 300.0    # pipeline length (m)

    # Ground displacement magnitude
    delta_lb: float = 0.01
    delta_ub: float = 0.5      # metres

    # Soil cohesion (Pa)
    c_lb:     float = 0.0
    c_ub:     float = 50e3

    # Friction angle (degrees)
    phi_lb:   float = 20.0
    phi_ub:   float = 45.0

    # Unit weight (N/m³)
    gamma_lb: float = 15e3
    gamma_ub: float = 22e3

    # Burial depth (m)
    H_lb:     float = 0.9
    H_ub:     float = 2.5

    def lb_array(self) -> np.ndarray:
        return np.array([self.x_lb, self.delta_lb, self.c_lb,
                         self.phi_lb, self.gamma_lb, self.H_lb])

    def ub_array(self) -> np.ndarray:
        return np.array([self.x_ub, self.delta_ub, self.c_ub,
                         self.phi_ub, self.gamma_ub, self.H_ub])

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lb = torch.tensor(self.lb_array(), dtype=torch.float32)
        ub = torch.tensor(self.ub_array(), dtype=torch.float32)
        return lb, ub


def sample_collocation_uniform(
    bounds: DomainBounds,
    n_col: int,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample N collocation points uniformly at random in the 6-D domain.

    Args:
        bounds : DomainBounds instance
        n_col  : number of collocation points
        device : 'cpu' or 'cuda'
        seed   : random seed for reproducibility

    Returns:
        X_col : (n_col, 6) tensor with requires_grad=True
    """
    if seed is not None:
        torch.manual_seed(seed)

    lb = torch.tensor(bounds.lb_array(), dtype=torch.float32)
    ub = torch.tensor(bounds.ub_array(), dtype=torch.float32)

    X = lb + (ub - lb) * torch.rand(n_col, 6)
    X = X.to(device).requires_grad_(True)
    return X


def sample_collocation_lhs(
    bounds: DomainBounds,
    n_col: int,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample N collocation points using Latin Hypercube Sampling (LHS).

    LHS provides better coverage of the parameter space compared to uniform
    random sampling, especially important for high-dimensional spaces.

    Args:
        bounds : DomainBounds instance
        n_col  : number of collocation points
        device : 'cpu' or 'cuda'
        seed   : random seed for reproducibility

    Returns:
        X_col : (n_col, 6) tensor with requires_grad=True
    """
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    sample  = sampler.random(n=n_col)                       # (n_col, 6) in [0,1]^6

    lb = bounds.lb_array()
    ub = bounds.ub_array()
    X_np = qmc.scale(sample, lb, ub).astype(np.float32)    # scale to physical domain

    X = torch.tensor(X_np, device=device).requires_grad_(True)
    return X


def sample_boundary_points(
    bounds: DomainBounds,
    n_bc: int,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample boundary condition points at x=0 and x=L.

    For each BC point, the parameter dimensions (δ, c, φ, γ, H) are sampled
    uniformly from their respective domains (the PINN must satisfy BCs for all
    parameter values, not just specific ones).

    Args:
        bounds : DomainBounds instance
        n_bc   : total number of BC points (split equally between x=0 and x=L)
        device : device string
        seed   : random seed

    Returns:
        X_bc        : (n_bc, 6) boundary point tensor, requires_grad=True
        u_bc_target : (n_bc,) zeros for u BC
        w_bc_target : (n_bc,) zeros for w BC
    """
    if seed is not None:
        np.random.seed(seed)

    n_each = n_bc // 2
    lb = bounds.lb_array()
    ub = bounds.ub_array()

    # Parameter dimensions only (cols 1–5, excluding x at col 0)
    def random_params(n):
        P = np.random.uniform(lb[1:], ub[1:], size=(n, 5)).astype(np.float32)
        return P

    # x = 0 boundary
    params_0  = random_params(n_each)
    x_left    = np.zeros((n_each, 1), dtype=np.float32)
    X_left    = np.hstack([x_left, params_0])

    # x = L boundary
    params_L  = random_params(n_bc - n_each)
    x_right   = np.full((n_bc - n_each, 1), bounds.x_ub, dtype=np.float32)
    X_right   = np.hstack([x_right, params_L])

    X_bc_np   = np.vstack([X_left, X_right])
    X_bc      = torch.tensor(X_bc_np, device=device).requires_grad_(True)

    u_bc_target = torch.zeros(n_bc, device=device)
    w_bc_target = torch.zeros(n_bc, device=device)

    return X_bc, u_bc_target, w_bc_target


def resample_collocation(
    bounds: DomainBounds,
    n_col: int,
    method: str = "lhs",
    device: str = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Convenience function that samples collocation points using specified method.

    Args:
        bounds : DomainBounds
        n_col  : number of points
        method : 'lhs' or 'uniform'
        device : device string
        seed   : random seed

    Returns:
        X_col : (n_col, 6) requires_grad=True tensor
    """
    if method == "lhs":
        return sample_collocation_lhs(bounds, n_col, device=device, seed=seed)
    elif method == "uniform":
        return sample_collocation_uniform(bounds, n_col, device=device, seed=seed)
    else:
        raise ValueError(f"Unknown sampling method '{method}'. Use 'lhs' or 'uniform'.")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bounds = DomainBounds()
    X_col  = sample_collocation_lhs(bounds, n_col=5000, seed=42)
    print(f"X_col shape: {X_col.shape}")
    print(f"x range:     [{X_col[:,0].min():.1f}, {X_col[:,0].max():.1f}]")
    print(f"delta range: [{X_col[:,1].min():.4f}, {X_col[:,1].max():.4f}]")

    X_bc, u_t, w_t = sample_boundary_points(bounds, n_bc=200, seed=0)
    print(f"\nX_bc shape: {X_bc.shape}")
    print(f"x values at BC: {X_bc[:5,0]}, {X_bc[-5:,0]}")
