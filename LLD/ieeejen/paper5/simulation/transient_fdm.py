"""
Finite Difference Method (FDM) forward solver for transient gas pipeline simulation.

Generates synthetic pressure and flow time series from known true parameters θ_true.
These serve as the "measurement" data for PIRN training and evaluation.

The simulation uses explicit forward Euler time integration of the
linearised isothermal gas flow PDEs.

References:
    Wang / Liu et al. (2025), PIRN (arXiv:2502.07230)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

R_UNIV = 8.314   # J/(mol·K)


# ─────────────────────────────────────────────────────────────────────────────
# Single-pipe FDM solver
# ─────────────────────────────────────────────────────────────────────────────

class GasPipelineFDM:
    """
    Explicit FDM solver for 1-D isothermal compressible gas flow.

    Solves the linearised PDEs:
        ∂p/∂t + (c²/A) * ∂q/∂x = 0       [continuity]
        ∂q/∂t + A * ∂p/∂x + R_f * q = 0  [momentum + friction]

    where c² = ZRT/M_g, R_f = λρ₀|v₀|/D.

    Args:
        D         : pipe inner diameter (m)
        L         : pipe length (m)
        lam_true  : true Darcy friction factor
        Z         : compressibility factor
        T         : temperature (K)
        M_g       : molar mass (kg/mol)
        p0        : nominal operating pressure (Pa)
        q0        : nominal mass flow (kg/s)
        N_spatial : number of spatial nodes
        dt        : time step (s); must satisfy CFL condition
    """

    def __init__(
        self,
        D:        float = 0.5,
        L:        float = 100e3,
        lam_true: float = 0.01,
        Z:        float = 0.9,
        T:        float = 288.0,
        M_g:      float = 0.01604,
        p0:       float = 5e6,
        q0:       float = 50.0,
        N_spatial: int  = 50,
        dt:       float = 60.0,
    ):
        self.D   = D
        self.L   = L
        self.lam = lam_true
        self.Z   = Z
        self.T   = T
        self.M_g = M_g
        self.p0  = p0
        self.q0  = q0
        self.N   = N_spatial
        self.dt  = dt

        # Derived quantities
        self.A_cross = np.pi * (D / 2.0) ** 2   # m²
        self.dx = L / N_spatial

        # Speed of sound
        self.c2   = Z * R_UNIV * T / M_g        # m²/s²
        self.c    = np.sqrt(self.c2)

        # Steady-state density and velocity
        self.rho0 = p0 * M_g / (Z * R_UNIV * T)   # kg/m³
        self.v0   = q0 / (self.rho0 * self.A_cross)  # m/s

        # Friction coefficient (linearised)
        self.R_f  = lam_true * self.rho0 * abs(self.v0) / D

        # CFL number check
        CFL = self.c * dt / self.dx
        if CFL > 1.0:
            import warnings
            warnings.warn(f"CFL={CFL:.2f} > 1 — solver may be unstable. "
                          f"Reduce dt or increase N_spatial.")

    def simulate(
        self,
        T_total: float,
        p_inlet_fn: Optional[callable] = None,
        q_outlet_fn: Optional[callable] = None,
        noise_level: float = 0.0,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Run time-domain simulation.

        Boundary conditions:
          - Inlet (x=0): prescribed pressure p_inlet(t) or nominal p0
          - Outlet (x=L): prescribed mass flow q_outlet(t) or nominal q0

        Args:
            T_total      : total simulation time (s)
            p_inlet_fn   : callable(t) → inlet pressure (Pa); None = p0 constant
            q_outlet_fn  : callable(t) → outlet mass flow (kg/s); None = q0 constant
            noise_level  : relative Gaussian noise std (e.g. 0.01 = 1%)
            seed         : random seed for noise

        Returns:
            results dict with:
                t      : (N_t,) time array (s)
                p      : (N_spatial, N_t) pressure field (Pa)
                q      : (N_spatial, N_t) mass flow field (kg/s)
                p_inlet, p_outlet : (N_t,) boundary pressures
                q_inlet, q_outlet : (N_t,) boundary flows
        """
        rng    = np.random.default_rng(seed)
        N_t    = int(T_total / self.dt)
        t_arr  = np.arange(N_t) * self.dt

        # Initialise with steady-state
        p = np.full((self.N, N_t), self.p0)   # pressure at each node
        q = np.full((self.N, N_t), self.q0)   # mass flow at each node

        # Default BCs
        if p_inlet_fn is None:
            p_inlet_fn = lambda t: self.p0 * (1.0 + 0.05 * np.sin(2*np.pi*t/3600))
        if q_outlet_fn is None:
            q_outlet_fn = lambda t: self.q0 * (1.0 + 0.1 * np.sin(2*np.pi*t/7200))

        # Pre-compute BC time series
        p_inlet  = np.array([p_inlet_fn(t)  for t in t_arr])
        q_outlet = np.array([q_outlet_fn(t) for t in t_arr])

        # Time integration (explicit forward Euler)
        for k in range(1, N_t):
            p_k = p[:, k-1].copy()
            q_k = q[:, k-1].copy()

            dp = np.zeros(self.N)
            dq = np.zeros(self.N)

            # Interior nodes
            for i in range(1, self.N - 1):
                # Continuity: dp/dt = -(c²/A) * dq/dx
                dq_dx = (q_k[i] - q_k[i-1]) / self.dx
                dp[i] = -(self.c2 / self.A_cross) * dq_dx

                # Momentum: dq/dt = -A * dp/dx - R_f * q
                dp_dx = (p_k[i+1] - p_k[i]) / self.dx
                dq[i] = -self.A_cross * dp_dx - self.R_f * q_k[i]

            # Apply BCs
            p[0,  k] = p_inlet[k]                        # inlet pressure BC
            q[-1, k] = q_outlet[k]                       # outlet flow BC

            # Update interior
            p[1:-1, k] = p_k[1:-1] + self.dt * dp[1:-1]
            q[0:-1, k] = q_k[0:-1] + self.dt * dq[0:-1]

        # Add measurement noise
        if noise_level > 0:
            p_noisy = p + noise_level * p * rng.standard_normal(p.shape)
            q_noisy = q + noise_level * q * rng.standard_normal(q.shape)
        else:
            p_noisy, q_noisy = p.copy(), q.copy()

        return {
            "t":         t_arr,
            "p":         p_noisy,
            "q":         q_noisy,
            "p_clean":   p,
            "q_clean":   q,
            "p_inlet":   p_inlet,
            "q_outlet":  q_outlet,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Network-level simulation (chain of pipes)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_network(
    network,
    lam_true_list: List[float],
    T_total:       float = 86400.0,
    dt:            float = 60.0,
    noise_level:   float = 0.01,
    seed:          int   = 0,
) -> Dict:
    """
    Simulate a gas network (GasLibNetwork) forward in time using known parameters.

    Each pipe is simulated independently with matching boundary conditions
    (pressure continuity at junctions).

    Args:
        network       : GasLibNetwork instance
        lam_true_list : list of true friction factors for each pipe
        T_total       : total simulation time (s)
        dt            : time step (s)
        noise_level   : measurement noise level (relative)
        seed          : random seed

    Returns:
        results dict with:
            t            : (N_t,) time array
            p_nodes      : (n_nodes, N_t) nodal pressures (training data)
            q_edges      : (n_pipes, N_t) pipe flows
            y_terminal   : (n_terminal_sensors, N_t) terminal measurements only
    """
    rng   = np.random.default_rng(seed)
    N_t   = int(T_total / dt)
    t_arr = np.arange(N_t) * dt

    p_nodes = np.zeros((network.n_nodes, N_t))
    q_edges = np.zeros((network.n_pipes, N_t))

    # Initialise all nodes at nominal pressure
    p0_nom = 5e6
    q0_nom = 50.0
    p_nodes[:, :] = p0_nom

    # Simulate each pipe and collect terminal pressures
    for j, (i_from, i_to) in enumerate(network.edges):
        props = network.pipe_props[j]
        solver = GasPipelineFDM(
            D=props["D"], L=props["L"],
            lam_true=lam_true_list[j],
            Z=props["Z"],
            T=288.0, M_g=0.01604,
            p0=p0_nom, q0=q0_nom,
            N_spatial=10, dt=dt,
        )
        results = solver.simulate(T_total, noise_level=noise_level, seed=seed + j)
        # Store terminal pressures at pipe endpoints
        p_nodes[i_from, :] = (p_nodes[i_from, :] + results["p"][0,  :]) / 2.0
        p_nodes[i_to,   :] = (p_nodes[i_to,   :] + results["p"][-1, :]) / 2.0
        q_edges[j, :]      = results["q"][5, :]   # mid-pipe flow

    # Extract terminal sensor measurements
    terminal_nodes = sorted(set(network.entry_nodes + network.exit_nodes))
    y_terminal     = p_nodes[terminal_nodes, :]

    return {
        "t":          t_arr,
        "p_nodes":    p_nodes,
        "q_edges":    q_edges,
        "y_terminal": y_terminal,
        "terminal_node_indices": terminal_nodes,
    }
