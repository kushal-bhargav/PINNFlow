"""
Physics-derived state-space model for isothermal gas pipeline segments.

Governing PDEs (linearised around steady-state (p₀, q₀)):
    Continuity: ∂ρ/∂t + ∂(ρv)/∂x = 0
    Momentum:   ρ∂v/∂t + ∂P/∂x + λρv|v|/(2D) = 0
    EOS:        P = ρZRT/M_g   →  c² = ZRT/M_g

After spatial discretisation of a single pipe segment of length Δx,
linearisation yields a discrete-time state-space model:

    [p_{k+1}]   =   A(θ) * [p_k]  +  B(θ) * [q_k]
    [q_{k+1}]           [q_k]

where θ = {λ, D, L, Z, T, M_g} are physical parameters.

The matrices A(θ) and B(θ) are implemented as differentiable PyTorch
operations so that θ can be learned by backpropagation through the PIRN.

References:
    Wang / Liu et al. (2025), Physics-Informed Recurrent Network for
    Gas Pipeline State-Space Modeling (arXiv:2502.07230)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

R_UNIV = 8.314   # universal gas constant (J/(mol·K))


# ─────────────────────────────────────────────────────────────────────────────
# Speed of sound and steady-state auxiliary quantities
# ─────────────────────────────────────────────────────────────────────────────

def speed_of_sound_squared(Z: torch.Tensor, T: float,
                            M_g: torch.Tensor) -> torch.Tensor:
    """
    c² = Z * R * T / M_g   (isothermal speed of sound squared, m²/s²)

    Args:
        Z   : compressibility factor (scalar or tensor)
        T   : temperature (K, scalar)
        M_g : molar mass of gas (kg/mol, tensor or scalar)

    Returns:
        c2 : tensor
    """
    return Z * R_UNIV * T / M_g


def darcy_friction_coefficient(lam: torch.Tensor, rho0: torch.Tensor,
                                v0: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Linearised friction coefficient R_f = λ * ρ₀ * |v₀| / (D) for the
    momentum equation (coefficient of the velocity perturbation).

    Args:
        lam  : Darcy friction factor (dimensionless)
        rho0 : steady-state density (kg/m³)
        v0   : steady-state velocity (m/s)
        D    : pipe inner diameter (m)

    Returns:
        R_f : friction coefficient (kg/(m³·s))
    """
    return lam * rho0 * torch.abs(v0) / D


# ─────────────────────────────────────────────────────────────────────────────
# Single-pipe state-space matrices A(θ), B(θ)
# ─────────────────────────────────────────────────────────────────────────────

class PipeSegmentStateSpace(nn.Module):
    """
    Computes the discrete-time state-space matrices A(θ) and B(θ) for a
    single pipe segment, parameterised by learnable physical parameters θ.

    State vector: x = [p_in, p_out, q_in, q_out]^T  (pressures and flows)
    Input vector: u = [q_source] (compressor or demand input)

    The matrices are derived from the finite-difference discretisation of
    the linearised gas flow PDEs, then discretised in time via forward Euler
    or the matrix exponential method.

    Args:
        lam_init  : initial estimate of Darcy friction factor
        D         : pipe inner diameter (m, fixed geometry from design)
        L         : pipe length (m, fixed geometry)
        Z_init    : initial gas compressibility estimate
        T         : operating temperature (K)
        M_g       : molar mass of gas (kg/mol)
        p0        : nominal operating pressure (Pa)
        q0        : nominal mass flow rate (kg/s)
        dt        : time step for discretisation (s)
        n_segments: number of spatial segments for finite-difference discretisation
    """

    def __init__(
        self,
        lam_init: float = 0.01,
        D:        float = 0.5,
        L:        float = 100e3,
        Z_init:   float = 0.9,
        T:        float = 288.0,
        M_g:      float = 0.01604,   # methane
        p0:       float = 5e6,
        q0:       float = 100.0,
        dt:       float = 60.0,
        n_segments: int = 10,
    ):
        super().__init__()

        self.D   = D
        self.L   = L
        self.T   = T
        self.M_g = M_g
        self.p0  = p0
        self.q0  = q0
        self.dt  = dt
        self.N   = n_segments

        # Learnable physical parameters (initialised from engineering estimates)
        self.lam = nn.Parameter(torch.tensor(lam_init, dtype=torch.float64))
        self.Z   = nn.Parameter(torch.tensor(Z_init,   dtype=torch.float64))

        # Fixed geometry parameters (not learned)
        self.register_buffer("D_t",   torch.tensor(D,  dtype=torch.float64))
        self.register_buffer("L_t",   torch.tensor(L,  dtype=torch.float64))
        self.register_buffer("M_g_t", torch.tensor(M_g,dtype=torch.float64))

        # Cross-section area and other derived quantities
        self.A_cross = np.pi * (D / 2.0) ** 2   # pipe bore area (m²)

    def _build_K_matrix(self) -> torch.Tensor:
        """
        Build the network-level system matrix K(θ) from physical parameters.

        For a single-pipe, N-segment finite difference discretisation:
          K is a (2N × 2N) tridiagonal system matrix derived from the
          linearised continuity and momentum PDEs.

        Returns:
            K : (2N, 2N) system matrix (float64 for numerical stability)
        """
        N   = self.N
        dx  = self.L_t / N
        dt  = self.dt

        # Speed of sound squared (differentiable)
        c2  = speed_of_sound_squared(self.Z, self.T, self.M_g_t)

        # Gas density at operating point
        rho0 = self.p0 * self.M_g_t / (self.Z * R_UNIV * self.T)
        v0   = torch.tensor(self.q0 / (rho0.detach().item() * self.A_cross),
                            dtype=torch.float64)

        # Friction coefficient
        R_f = darcy_friction_coefficient(self.lam, rho0, v0, self.D_t)

        # Build block-tridiagonal K using sparse structure
        # State vector ordering: [p_1, ..., p_N, q_1, ..., q_N]
        K = torch.zeros(2 * N, 2 * N, dtype=torch.float64)

        # Continuity block: dp/dt + (c²/A) * dq/dx = 0
        # Discretised: (p_{k+1,i} - p_{k,i})/dt = -(c²/A)*(q_{k,i} - q_{k,i-1})/dx
        c2_over_Adx = c2 / (self.A_cross * dx)
        for i in range(N):
            K[i, i]   = 1.0 / dt                    # p_i diagonal
            if i > 0:
                K[i, N + i - 1] = -c2_over_Adx      # -q_{i-1}
            K[i, N + i]     =  c2_over_Adx           # +q_i

        # Momentum block: dq/dt + (A/ρ₀)*dp/dx + R_f*q = 0
        # Discretised: (q_{k+1,i} - q_{k,i})/dt = -(A/ρ₀)*(p_{k,i+1}-p_{k,i})/dx - R_f*q_{k,i}
        A_over_rho0dx = torch.tensor(self.A_cross, dtype=torch.float64) / (rho0 * dx)
        for i in range(N):
            K[N + i, N + i] = 1.0 / dt + R_f        # q_i diagonal
            K[N + i, i]     = -A_over_rho0dx          # -p_i
            if i < N - 1:
                K[N + i, i + 1] = A_over_rho0dx      # +p_{i+1}

        return K

    def forward(self, x_k: torch.Tensor) -> torch.Tensor:
        """
        One-step state update: x_{k+1} = A(θ) * x_k

        Uses K(θ)⁻¹ factorisation to compute the update:
            K * x_{k+1} = x_k / dt  +  (rhs terms)

        For the fully linearised case, implemented as:
            x_{k+1} = (I - dt * F) * x_k

        where F encodes the linearised PDE operator (simplified here for clarity).

        Args:
            x_k : (n_state,) state vector at time k

        Returns:
            x_{k+1} : (n_state,) state vector at time k+1
        """
        K = self._build_K_matrix()
        dt = torch.tensor(self.dt, dtype=torch.float64)
        # Solve K * x_{k+1} = x_k / dt  (forward Euler rearranged)
        rhs = x_k.double() / dt
        x_next = torch.linalg.solve(K, rhs)
        return x_next.to(x_k.dtype)

    def get_AB_matrices(self) -> Dict[str, torch.Tensor]:
        """
        Return explicit A(θ) and B(θ) matrices (for analysis purposes).

        A = K⁻¹ / dt  (simplified; exact form depends on forcing structure)

        Returns:
            dict with 'A', 'B', 'K' tensors
        """
        K   = self._build_K_matrix()
        K_inv = torch.linalg.inv(K)
        A   = K_inv / self.dt
        B   = K_inv[:, self.N:]   # input mapping (from mass flows at boundaries)
        return {"A": A, "B": B, "K": K, "K_inv": K_inv}
